"""Generates assets/icon.png and a multi-size assets/icon.ico.  Run:  python tools/make_icon.py
Design: gradient rounded tile, white head-and-shoulders silhouette, viewfinder corners, green scan line."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "assets"
S = 1024                                           # master canvas; everything is drawn in 0..1 units


def render(stroke: float = 1.0, glow: bool = True) -> Image.Image:
    u = lambda v: v * S
    # background: diagonal gradient (indigo -> cyan) clipped to a rounded tile
    y, x = np.mgrid[0:S, 0:S] / (S - 1)
    t = np.clip((x * 0.55 + y * 0.85) / 1.4, 0, 1)[..., None]
    c0, c1 = np.array([67, 56, 202]), np.array([14, 165, 233])
    bg = Image.fromarray((c0 + (c1 - c0) * t).astype(np.uint8), "RGB").convert("RGBA")
    tile = Image.new("L", (S, S), 0)
    ImageDraw.Draw(tile).rounded_rectangle((u(.02), u(.02), u(.98), u(.98)), radius=u(.22), fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(bg, (0, 0), tile)

    # soft highlight on the top-left of the tile
    hl = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    ImageDraw.Draw(hl).ellipse((u(-.25), u(-.35), u(.75), u(.45)), fill=(255, 255, 255, 38))
    hl.putalpha(Image.composite(hl.getchannel("A"), Image.new("L", (S, S), 0), tile))
    img = Image.alpha_composite(img, hl)

    # head + shoulders (white), shoulders cut flat so they sit inside the viewfinder
    sil = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(sil)
    d.ellipse((u(.5 - .125), u(.41 - .125), u(.5 + .125), u(.41 + .125)), fill=255)            # head
    d.ellipse((u(.5 - .235), u(.60), u(.5 + .235), u(.60 + .40)), fill=255)                    # shoulders
    d.rectangle((0, u(.775), S, S), fill=0)
    shadow = Image.new("RGBA", (S, S), (20, 20, 90, 0))
    shadow.putalpha(sil.filter(ImageFilter.GaussianBlur(u(.012))).point(lambda v: int(v * .35)))
    img = Image.alpha_composite(img, shadow.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, -u(.012))))
    img = Image.alpha_composite(img, Image.merge("RGBA", (*[Image.new("L", (S, S), 255)] * 3, sil)))

    # viewfinder corners
    w, L, a, b = u(.038 * stroke), u(.15), u(.17), u(.83)
    br = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(br)
    col = (255, 255, 255, 255)
    for cx, cy, sx, sy in ((a, a, 1, 1), (b, a, -1, 1), (a, b, 1, -1), (b, b, -1, -1)):
        d.line((cx, cy, cx + sx * L, cy), fill=col, width=int(w))
        d.line((cx, cy, cx, cy + sy * L), fill=col, width=int(w))
        for px, py in ((cx, cy), (cx + sx * L, cy), (cx, cy + sy * L)):
            d.ellipse((px - w / 2, py - w / 2, px + w / 2, py + w / 2), fill=col)
    img = Image.alpha_composite(img, br)

    # scan line (green) with glow
    ln = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(ln).rounded_rectangle((u(.12), u(.505 - .018 * stroke), u(.88), u(.505 + .018 * stroke)),
                                         radius=u(.02), fill=(52, 211, 153, 255))
    if glow:
        img = Image.alpha_composite(img, ln.filter(ImageFilter.GaussianBlur(u(.03))))
    return Image.alpha_composite(img, ln)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    big, small = render(1.0, True), render(1.7, False)             # small sizes get bolder strokes, no glow
    big.resize((512, 512), Image.LANCZOS).save(OUT / "icon.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [(small if s <= 32 else big).resize((s, s), Image.LANCZOS) for s in sizes]
    frames[-1].save(OUT / "icon.ico", format="ICO", sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    print("wrote", OUT / "icon.png", "and", OUT / "icon.ico")
