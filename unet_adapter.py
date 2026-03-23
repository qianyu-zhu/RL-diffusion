"""
U-Net adapter for LASD experiments.
Extends the pipeline to support Stable Diffusion v1.5 (U-Net architecture)
in addition to DiT-XL/2.

Key differences from DiT:
  - U-Net has spatial (conv) activations, not patch tokens
  - Hookable locations: resnet outputs, cross-attn outputs, mid block
  - "h-space" = mid block output (bottleneck) — most studied in literature
  - Text conditioning via cross-attention, not class embeddings
  - Supports callback_on_step_end (unlike DiTPipeline)
"""
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, DTYPE, NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR, CONCEPTS,
)
from eval import get_concept_labels, Timer, get_peak_memory_mb

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_IMAGE_SIZE = 512

# Hookable block locations in the U-Net
# Format: (block_type, block_idx, sub_idx)
UNET_HOOK_POINTS = {
    "down_0": ("down", 0, -1),   # after down_block 0 (320-d)
    "down_1": ("down", 1, -1),   # after down_block 1 (640-d)
    "down_2": ("down", 2, -1),   # after down_block 2 (1280-d)
    "down_3": ("down", 3, -1),   # after down_block 3 (1280-d)
    "mid":    ("mid", 0, -1),    # h-space bottleneck (1280-d)
    "up_0":   ("up", 0, -1),     # after up_block 0 (1280-d)
    "up_1":   ("up", 1, -1),     # after up_block 1 (1280-d)
    "up_2":   ("up", 2, -1),     # after up_block 2 (640-d)
    "up_3":   ("up", 3, -1),     # after up_block 3 (320-d)
}


def load_sd_pipeline():
    """Load Stable Diffusion v1.5 pipeline."""
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    pipe.safety_checker = None  # disable for research
    return pipe


def get_unet(pipe):
    """Extract UNet from SD pipeline."""
    return pipe.unet


class UNetActivationCollector:
    """Collect activations from U-Net blocks.

    Hooks into resnet block outputs at specified locations.
    Activations are spatially averaged (global average pooling) to get
    a (batch, channels) vector per block.
    """

    def __init__(self, unet, hook_points=None):
        self.unet = unet
        self.activations = {}
        self.hooks = []
        self.hook_points = hook_points or list(UNET_HOOK_POINTS.keys())
        self._register_hooks()

    def _register_hooks(self):
        for point_name in self.hook_points:
            if point_name not in UNET_HOOK_POINTS:
                continue
            block_type, block_idx, sub_idx = UNET_HOOK_POINTS[point_name]

            if block_type == "down":
                block = self.unet.down_blocks[block_idx]
                target = block.resnets[sub_idx]
            elif block_type == "mid":
                target = self.unet.mid_block.resnets[sub_idx]
            elif block_type == "up":
                block = self.unet.up_blocks[block_idx]
                target = block.resnets[sub_idx]
            else:
                continue

            hook = target.register_forward_hook(self._make_hook(point_name))
            self.hooks.append(hook)

    def _make_hook(self, point_name):
        def hook_fn(module, input, output):
            # output shape: (batch, channels, height, width)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # With CFG, batch is doubled
            batch_size = hidden.shape[0]
            if batch_size > 1 and batch_size % 2 == 0:
                hidden = hidden[:batch_size // 2]

            # Global average pooling: (batch, C, H, W) -> (batch, C)
            pooled = hidden.mean(dim=[2, 3]).detach().cpu()
            self.activations[point_name] = pooled
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class UNetSteeringHook:
    """Inject concept vectors into U-Net activations during denoising.

    Key insight from Kwon et al. (ICLR 2023): editing is only effective
    during the first ~30% of denoising steps (high-noise, semantic region).
    Later steps control fine details and should not be perturbed.
    """

    def __init__(self, unet, concept_vectors, epsilon=0.1, norm_clip=1.5,
                 steer_fraction=0.3):
        """
        Args:
            unet: UNet2DConditionModel
            concept_vectors: dict of {hook_point_name: (C,) tensor}
            epsilon: steering strength
            norm_clip: max norm ratio (0 = no clip)
            steer_fraction: fraction of early denoising steps to steer (0.3 = first 30%)
        """
        self.unet = unet
        self.concept_vectors = concept_vectors
        self.epsilon = epsilon
        self.norm_clip = norm_clip
        self.steer_fraction = steer_fraction
        self.current_step = 0
        self.total_steps = NUM_INFERENCE_STEPS
        self.hooks = []
        self._register_hooks()

    def step_callback(self, pipe, step, timestep, callback_kwargs):
        """SD pipeline callback to track current step."""
        self.current_step = step + 1
        return callback_kwargs

    @property
    def _should_steer(self):
        """Only steer during the semantic editing window (first N% of steps)."""
        if self.steer_fraction >= 1.0:
            return True
        return self.current_step < int(self.total_steps * self.steer_fraction)

    def _register_hooks(self):
        for point_name, vec in self.concept_vectors.items():
            if point_name not in UNET_HOOK_POINTS:
                continue
            block_type, block_idx, sub_idx = UNET_HOOK_POINTS[point_name]

            if block_type == "down":
                target = self.unet.down_blocks[block_idx].resnets[sub_idx]
            elif block_type == "mid":
                target = self.unet.mid_block.resnets[sub_idx]
            elif block_type == "up":
                target = self.unet.up_blocks[block_idx].resnets[sub_idx]
            else:
                continue

            hook = target.register_forward_hook(self._make_hook(vec))
            self.hooks.append(hook)

    def _make_hook(self, concept_vec):
        def hook_fn(module, input, output):
            if not self._should_steer:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # hidden: (batch, C, H, W)
            v = concept_vec.to(hidden.device, hidden.dtype)
            # Broadcast: (C,) -> (1, C, 1, 1)
            v_spatial = v.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            orig_norm = hidden.norm(dim=1, keepdim=True)
            hidden = hidden + self.epsilon * v_spatial

            if self.norm_clip > 0:
                new_norm = hidden.norm(dim=1, keepdim=True)
                max_norm = orig_norm * self.norm_clip
                scale = torch.clamp(max_norm / new_norm.clamp(min=1e-8), max=1.0)
                hidden = hidden * scale

            if rest is not None:
                return (hidden,) + rest
            return hidden
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def generate_sd_and_collect(pipe, n_samples, collector, prompts=None):
    """Generate images with SD and collect activations.

    Args:
        prompts: list of text prompts. If None, uses generic prompts.

    Returns:
        images, activations, prompts_used
    """
    if prompts is None:
        # Use generic prompts that generate diverse images
        base_prompts = [
            "a photograph", "a painting", "a landscape",
            "a portrait", "an animal", "a building",
            "a scene", "nature photography", "digital art",
            "a still life",
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(n_samples)]

    images = []
    all_activations = {}
    prompts_used = []

    for i in tqdm(range(0, n_samples, EVAL_BATCH_SIZE), desc="Generating (SD)"):
        batch_size = min(EVAL_BATCH_SIZE, n_samples - i)
        batch_prompts = prompts[i:i + batch_size]

        collector.clear()

        with torch.no_grad():
            output = pipe(
                batch_prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
            )

        images.extend(output.images)
        prompts_used.extend(batch_prompts)

        for point_name, act in collector.activations.items():
            if point_name not in all_activations:
                all_activations[point_name] = []
            all_activations[point_name].append(act)

    stacked = {}
    for point_name, act_list in all_activations.items():
        stacked[point_name] = torch.cat(act_list, dim=0)

    return images, stacked, prompts_used


def generate_sd_steered(pipe, n_images, concept_vectors, epsilon=0.1,
                        prompts=None, norm_clip=1.5, steer_fraction=0.3):
    """Generate steered images with SD.

    Args:
        steer_fraction: fraction of early denoising steps to steer (0.3 = first 30%,
                        per Kwon et al. ICLR 2023). Set to 1.0 to steer all steps.
    """
    unet = get_unet(pipe)
    steering = UNetSteeringHook(unet, concept_vectors, epsilon=epsilon,
                                norm_clip=norm_clip, steer_fraction=steer_fraction)

    if prompts is None:
        base_prompts = [
            "a photograph", "a painting", "a landscape",
            "a portrait", "an animal", "a building",
            "a scene", "nature photography", "digital art",
            "a still life",
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(n_images)]

    images = []
    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc=f"Steering SD (eps={epsilon})"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)
        batch_prompts = prompts[i:i + batch_size]

        steering.current_step = 0  # reset step counter per batch

        with torch.no_grad():
            output = pipe(
                batch_prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
                callback_on_step_end=steering.step_callback,
            )
        images.extend(output.images)

    steering.remove_hooks()
    return images, prompts


def generate_sd_baseline(pipe, n_images, prompts=None):
    """Generate baseline SD images (no steering)."""
    if prompts is None:
        base_prompts = [
            "a photograph", "a painting", "a landscape",
            "a portrait", "an animal", "a building",
            "a scene", "nature photography", "digital art",
            "a still life",
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(n_images)]

    images = []
    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc="Baseline (SD)"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)
        batch_prompts = prompts[i:i + batch_size]
        with torch.no_grad():
            output = pipe(
                batch_prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
            )
        images.extend(output.images)

    return images, prompts
