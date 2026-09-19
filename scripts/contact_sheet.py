"""Tile the PNGs written by render_slides.ps1 into one overview image:  python scripts/contact_sheet.py <preview dir> [columns]"""
import sys
from pathlib import Path

from PIL import Image

folder, cols = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
files = sorted(folder.glob("slide_*.png"))
thumbs = [Image.open(f).convert("RGB") for f in files]
w = 640
thumbs = [t.resize((w, round(t.height * w / t.width))) for t in thumbs]
h = thumbs[0].height
rows = -(-len(thumbs) // cols)
sheet = Image.new("RGB", (cols * (w + 8) + 8, rows * (h + 8) + 8), (120, 120, 120))
for i, t in enumerate(thumbs):
    sheet.paste(t, (8 + (i % cols) * (w + 8), 8 + (i // cols) * (h + 8)))
sheet.save(folder / "contact_sheet.png")
print(folder / "contact_sheet.png")
