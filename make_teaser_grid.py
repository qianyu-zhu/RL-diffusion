"""Make a publication-quality teaser grid from teaser_v2 images."""
import os, numpy as np
from PIL import Image, ImageDraw

def main():
    d = 'figures/teaser_v2'
    out = 'figures/teaser_grid.png'
    cell = 200
    n_cols = 5  # images per row
    gap = 4
    label_h = 20

    # Pick 5 diverse examples (indices with different classes)
    indices = [0, 2, 4, 6, 8]

    rows_data = [
        ("Baseline", "baseline"),
        ("+ Brightness (ε=−0.3)", "bright"),
        ("+ Warmth (ε=−0.3)", "warm"),
        ("+ Animal (ε=+2)", "animal_mild"),
        ("+ Natural (ε=−2)", "natural_mild"),
        ("+ Bright+Warm (ε=−0.3)", "compose_bw"),
    ]

    total_w = n_cols * cell + (n_cols - 1) * gap
    total_h = len(rows_data) * (cell + label_h + gap)

    canvas = Image.new('RGB', (total_w + 60, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row_idx, (label, prefix) in enumerate(rows_data):
        y = row_idx * (cell + label_h + gap)
        draw.text((2, y + 2), label, fill=(0, 0, 0))

        for col_idx, img_idx in enumerate(indices):
            fname = f"{prefix}_{img_idx:02d}.png"
            if prefix == "baseline":
                # Find the baseline with matching index
                candidates = [f for f in os.listdir(d) if f.startswith(f"baseline_{img_idx:02d}")]
                if candidates:
                    fname = candidates[0]

            path = os.path.join(d, fname)
            if os.path.exists(path):
                img = Image.open(path).resize((cell, cell), Image.LANCZOS)
                x = 60 + col_idx * (cell + gap)
                canvas.paste(img, (x, y + label_h))

    canvas.save(out)
    print(f"Saved {out} ({canvas.size})")

if __name__ == "__main__":
    main()
