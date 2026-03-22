"""One-time setup: download DiT-S/2 checkpoint and verify dependencies."""
import subprocess
import sys


def main():
    print("=== LASD Setup ===\n")

    # 1. Check dependencies
    print("1. Checking dependencies...")
    required = ["torch", "diffusers", "transformers", "accelerate", "scipy", "sklearn", "tqdm"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} — MISSING")
            missing.append(pkg)

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Run: uv sync")
        return

    # 2. Check device
    import torch
    if torch.backends.mps.is_available():
        print(f"\n2. Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        print(f"\n2. Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"\n2. Device: CPU (WARNING: will be very slow)")

    # 3. Download model
    print("\n3. Downloading DiT-XL/2-256 (675M params)...")
    from diffusers import DiTPipeline
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
    print("   ✓ DiT-XL/2-256 downloaded and cached")

    # 4. Quick smoke test
    print("\n4. Smoke test: generating 1 image...")
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = pipe.to(device)
    with torch.no_grad():
        output = pipe([207], num_inference_steps=10, guidance_scale=4.0, output_type="pil")
    output.images[0].save("smoke_test.png")
    print("   ✓ Generated smoke_test.png")

    # 5. Create directories
    import os
    os.makedirs("vectors", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("\n5. Created directories: vectors/, results/")

    print("\n=== Setup complete! ===")


if __name__ == "__main__":
    main()
