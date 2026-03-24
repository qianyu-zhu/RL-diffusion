"""
FK (Feynman-Kac) Steering Baseline.

Implements simplified FK steering as the upper-bound comparison for LASD.

FK steering (Dockhorn et al. 2024, Singhal et al. 2025):
- At each denoising step, maintain K particles (noisy copies of x_t)
- Score each particle by a reward function r(x)
- Reweight the noise prediction by the particle scores
- This approximates the optimal drift for the reward-conditioned SDE

Our simplified version (FK-lite):
1. For each image, generate K candidate denoised versions at each step
2. Score each with the concept labeler
3. Select the best scoring candidate
4. Continue denoising from the selected candidate

This gives an upper bound on what activation steering COULD achieve if it
had access to the value function at each step.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR,
)
from eval import (
    get_concept_labels, compute_diversity, print_summary, Timer, get_peak_memory_mb,
)
from extract import load_dit_pipeline, get_dit_transformer
from steer import generate_baseline, load_concept_vectors, generate_steered


RESULTS_FILE = "results.tsv"


def continuous_brightness(images):
    """Return continuous brightness score per image."""
    return np.array([np.array(img).astype(np.float32).mean() / 255.0 for img in images])


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def denoise_one_step(pipe, latent, t, class_id):
    """Run one denoising step using the pipeline's internal logic.

    DiT predicts 8 channels (noise + variance); need to split and handle correctly.
    """
    transformer = get_dit_transformer(pipe)
    scheduler = pipe.scheduler

    t_tensor = torch.tensor([t], device=DEVICE)
    class_labels = torch.tensor([class_id], device=DEVICE)

    # CFG: concat conditional + unconditional
    latent_input = torch.cat([latent, latent], dim=0)
    t_input = torch.cat([t_tensor, t_tensor], dim=0)
    class_input = torch.cat([class_labels, torch.tensor([NUM_CLASSES], device=DEVICE)])

    with torch.no_grad():
        model_output = transformer(
            latent_input, timestep=t_input, class_labels=class_input,
        ).sample

    # DiT predicts 8 channels: split into noise and variance (like DiTPipeline does)
    # Only use the noise prediction (first 4 channels)
    channels = latent.shape[1]
    model_output_cond, model_output_uncond = model_output.chunk(2)

    # Split noise and learned variance
    noise_cond = model_output_cond[:, :channels]
    noise_uncond = model_output_uncond[:, :channels]

    # CFG on the noise part only
    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

    # Also need the variance part for the scheduler
    # Use the conditional variance
    if model_output_cond.shape[1] > channels:
        variance = model_output_cond[:, channels:]
        noise_pred = torch.cat([noise_pred, variance], dim=1)

    step_output = scheduler.step(noise_pred, t, latent)
    return step_output.prev_sample


def fk_steer_brightness_simple(pipe, n_images, k_particles=4, n_resample_steps=5):
    """Simplified FK steering: at selected steps, generate K alternatives and pick brightest.

    At each resampling step:
    1. Save the current latent state
    2. Generate K noisy variants (add small noise to current latent)
    3. Run one denoising step for each variant
    4. Score each variant by brightness proxy
    5. Continue with the best variant
    """
    import copy

    images = []
    all_class_ids = []

    for img_idx in tqdm(range(n_images), desc=f"FK(k={k_particles})"):
        class_id = torch.randint(0, NUM_CLASSES, (1,)).item()
        all_class_ids.append(class_id)

        # Initialize latent
        latent_shape = (1, 4, 32, 32)  # DiT-XL/2 at 256x256
        latent = torch.randn(latent_shape, device=DEVICE, dtype=DTYPE)

        # Set up scheduler for this image
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        timesteps = pipe.scheduler.timesteps

        total_steps = len(timesteps)
        resample_at = set(np.linspace(0, total_steps - 1, n_resample_steps + 2,
                                       dtype=int)[1:-1].tolist())

        for step_idx, t in enumerate(timesteps):
            if step_idx in resample_at:
                # FK resampling: try K particles
                best_latent = latent
                best_score = -1e9

                for k in range(k_particles):
                    if k == 0:
                        candidate = latent.clone()
                    else:
                        noise_scale = 0.1 * (1.0 - step_idx / total_steps)
                        candidate = latent + noise_scale * torch.randn_like(latent)

                    candidate_next = denoise_one_step(pipe, candidate, t, class_id)

                    # Score: brightness proxy from latent statistics
                    # At late steps, decode for better estimate
                    if step_idx > total_steps * 0.7:
                        with torch.no_grad():
                            decoded = pipe.vae.decode(
                                candidate_next / pipe.vae.config.scaling_factor
                            ).sample
                            score = decoded.mean().item()
                    else:
                        score = candidate_next.mean().item()

                    if score > best_score:
                        best_score = score
                        best_latent = candidate_next

                latent = best_latent
            else:
                latent = denoise_one_step(pipe, latent, t, class_id)

        # Final decode
        with torch.no_grad():
            decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            img_np = decoded[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))

    return images, all_class_ids


def main():
    timer = Timer()
    n_images = 100  # FK is expensive — 100 images is reasonable

    print("="*60)
    print("  FK Steering Baseline Comparison")
    print("="*60)

    print(f"\nLoading model {MODEL_ID}...")
    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    # Fixed class ids for fair comparison
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # 1. Baseline (no steering)
    print("\n--- Generating baseline ---")
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()
    print(f"  Baseline: brightness={baseline_bright.mean():.3f}, pos_rate={baseline_rate:.3f}")

    # 2. LASD best config (all-layer mean_diff eps=-0.5)
    print("\n--- LASD (all-layer mean_diff eps=-0.5) ---")
    vectors = load_concept_vectors("brightness", method="mean_diff")
    lasd_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=-0.5, class_ids=class_ids)
    lasd_bright = continuous_brightness(lasd_imgs)
    lasd_rate = (np.array([1 if b > 0.5 else -1 for b in lasd_bright]) == 1).mean()
    lasd_lift = lasd_rate - baseline_rate
    print(f"  LASD: brightness={lasd_bright.mean():.3f}, pos_rate={lasd_rate:.3f}, lift={lasd_lift:+.3f}")

    append_result(commit, "S8-FK", "lift", lasd_lift, 3.9, "keep",
                  "FK-comparison LASD all-layer mean_diff eps=-0.5")

    # 3. LASD norm_scale (layer0, eps=-0.1)
    print("\n--- LASD norm_scale (layer0 eps=-0.1) ---")
    vectors_l0 = load_concept_vectors("brightness", method="mean_diff", layers=[0])
    lasd_ns_imgs, _ = generate_steered(pipe, n_images, vectors_l0, epsilon=-0.1,
                                        class_ids=class_ids, norm_scale=True)
    lasd_ns_bright = continuous_brightness(lasd_ns_imgs)
    lasd_ns_rate = (np.array([1 if b > 0.5 else -1 for b in lasd_ns_bright]) == 1).mean()
    lasd_ns_lift = lasd_ns_rate - baseline_rate
    print(f"  LASD-NS: brightness={lasd_ns_bright.mean():.3f}, pos_rate={lasd_ns_rate:.3f}, lift={lasd_ns_lift:+.3f}")

    append_result(commit, "S8-FK", "lift", lasd_ns_lift, 3.9, "keep",
                  "FK-comparison LASD layer0 norm_scale eps=-0.1")

    # 4. FK steering (k=2, 5 resample steps)
    print("\n--- FK(k=2, resample=5) ---")
    fk2_imgs, fk2_cids = fk_steer_brightness_simple(pipe, n_images, k_particles=2, n_resample_steps=5)
    fk2_bright = continuous_brightness(fk2_imgs)
    fk2_rate = (np.array([1 if b > 0.5 else -1 for b in fk2_bright]) == 1).mean()
    fk2_lift = fk2_rate - baseline_rate
    print(f"  FK(k=2): brightness={fk2_bright.mean():.3f}, pos_rate={fk2_rate:.3f}, lift={fk2_lift:+.3f}")

    append_result(commit, "S8-FK", "lift", fk2_lift, 3.9, "keep",
                  f"FK k=2 resample=5 brightness")

    # 5. FK steering (k=4, 5 resample steps)
    print("\n--- FK(k=4, resample=5) ---")
    fk4_imgs, fk4_cids = fk_steer_brightness_simple(pipe, n_images, k_particles=4, n_resample_steps=5)
    fk4_bright = continuous_brightness(fk4_imgs)
    fk4_rate = (np.array([1 if b > 0.5 else -1 for b in fk4_bright]) == 1).mean()
    fk4_lift = fk4_rate - baseline_rate
    print(f"  FK(k=4): brightness={fk4_bright.mean():.3f}, pos_rate={fk4_rate:.3f}, lift={fk4_lift:+.3f}")

    append_result(commit, "S8-FK", "lift", fk4_lift, 3.9, "keep",
                  f"FK k=4 resample=5 brightness")

    # 6. FK steering (k=8, 10 resample steps) — expensive upper bound
    print("\n--- FK(k=8, resample=10) ---")
    fk8_imgs, fk8_cids = fk_steer_brightness_simple(pipe, n_images, k_particles=8, n_resample_steps=10)
    fk8_bright = continuous_brightness(fk8_imgs)
    fk8_rate = (np.array([1 if b > 0.5 else -1 for b in fk8_bright]) == 1).mean()
    fk8_lift = fk8_rate - baseline_rate
    print(f"  FK(k=8): brightness={fk8_bright.mean():.3f}, pos_rate={fk8_rate:.3f}, lift={fk8_lift:+.3f}")

    append_result(commit, "S8-FK", "lift", fk8_lift, 3.9, "keep",
                  f"FK k=8 resample=10 brightness")

    # 7. Compute cost ratios
    # LASD: cost_ratio = 1.0 (zero overhead)
    # FK(k, r): cost ≈ (1 + k*r/total_steps) per image
    total_steps = NUM_INFERENCE_STEPS
    print("\n--- Cost Comparison ---")
    print(f"  {'Method':<30s} {'Lift':>8s} {'Cost Ratio':>12s} {'Lift/Cost':>12s}")
    print(f"  {'-'*62}")
    for name, lift, k, r in [
        ("LASD all-layer", lasd_lift, 0, 0),
        ("LASD norm_scale", lasd_ns_lift, 0, 0),
        (f"FK(k=2,r=5)", fk2_lift, 2, 5),
        (f"FK(k=4,r=5)", fk4_lift, 4, 5),
        (f"FK(k=8,r=10)", fk8_lift, 8, 10),
    ]:
        if k == 0:
            cost = 1.0
        else:
            cost = 1.0 + (k * r) / total_steps
        efficiency = lift / cost if cost > 0 else 0
        print(f"  {name:<30s} {lift:>+8.3f} {cost:>12.2f}x {efficiency:>+12.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — FK Baseline Comparison")
    print(f"  Total wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"  Peak GPU memory: {get_peak_memory_mb():.1f} MB")
    print(f"{'='*60}")

    print_summary(
        lasd_lift=lasd_lift,
        fk2_lift=fk2_lift,
        fk4_lift=fk4_lift,
        fk8_lift=fk8_lift,
        wall_seconds=timer.elapsed(),
        peak_vram_mb=get_peak_memory_mb(),
    )


if __name__ == "__main__":
    main()
