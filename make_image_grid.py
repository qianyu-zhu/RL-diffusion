"""Generate qualitative image grids from saved steering results."""
import os
import numpy as np
from PIL import Image

def make_grid(image_dir, prefix_pairs, n_cols=5, cell_size=256):
    """Create a comparison grid: rows of (baseline, steered) pairs."""
    rows = []
    for base_prefix, steer_prefix, label in prefix_pairs:
        base_files = sorted([f for f in os.listdir(image_dir) if f.startswith(base_prefix)])[:n_cols]
        steer_files = sorted([f for f in os.listdir(image_dir) if f.startswith(steer_prefix)])[:n_cols]

        if not base_files or not steer_files:
            continue

        # Make row of baseline images
        base_row = []
        for f in base_files:
            img = Image.open(os.path.join(image_dir, f)).resize((cell_size, cell_size))
            base_row.append(np.array(img))

        # Make row of steered images
        steer_row = []
        for f in steer_files:
            img = Image.open(os.path.join(image_dir, f)).resize((cell_size, cell_size))
            steer_row.append(np.array(img))

        if base_row and steer_row:
            n = min(len(base_row), len(steer_row))
            rows.append(("Baseline", np.concatenate(base_row[:n], axis=1)))
            rows.append((label, np.concatenate(steer_row[:n], axis=1)))

    if not rows:
        print(f"  No images found in {image_dir}")
        return None

    # Add row labels
    from PIL import ImageDraw, ImageFont
    full_height = sum(r[1].shape[0] for r in rows) + 30 * len(rows)
    full_width = rows[0][1].shape[1]
    canvas = Image.new('RGB', (full_width, full_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for label, arr in rows:
        draw.text((5, y + 5), label, fill=(0, 0, 0))
        y += 25
        canvas.paste(Image.fromarray(arr), (0, y))
        y += arr.shape[0] + 5

    return canvas


def main():
    os.makedirs('figures', exist_ok=True)

    # Semantic fix images
    sem_dir = 'results/semantic_fix'
    if os.path.exists(sem_dir):
        grid = make_grid(sem_dir, [
            ('baseline_', 'bright_steered_', 'Brightness steered'),
            ('baseline_', 'animal_steered_', 'Animal steered (ε=-5)'),
            ('baseline_', 'animal_extreme_', 'Animal extreme (ε=-20)'),
        ], n_cols=5)
        if grid:
            grid.save('figures/qualitative_semantic.png')
            print("Saved figures/qualitative_semantic.png")

    # Check other result dirs
    for d in sorted(os.listdir('results')):
        full = os.path.join('results', d)
        if os.path.isdir(full):
            files = os.listdir(full)
            base = [f for f in files if f.startswith('baseline')]
            steer = [f for f in files if f.startswith('steered')]
            if base and steer:
                grid = make_grid(full, [
                    ('baseline_', 'steered_', d),
                ], n_cols=5)
                if grid:
                    outname = f'figures/qualitative_{d.replace("/", "_")}.png'
                    grid.save(outname)
                    print(f"Saved {outname}")

    print("Done generating image grids.")


if __name__ == "__main__":
    main()
