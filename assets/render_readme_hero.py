"""Render a conceptual, seamless README hero for BuildNet3D-Uncertainty.

Run with Pillow: python render_readme_hero.py [output_directory]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H = 1152, 480
N_FRAMES = 48
FRAME_MS = 100
BG = (252, 252, 249)
WHITE = (28, 33, 33)  # Ink; retained name to keep text calls concise.
MUTED = (104, 111, 109)
CYAN = (35, 127, 115)  # Restrained scan color on the paper background.
CYAN_SOFT = (112, 149, 139)
ORANGE = (222, 105, 38)
VIOLET = (156, 150, 137)  # Warm roof gray.
WALL = (197, 213, 200)
ROOF = (184, 182, 170)
WINDOW = (35, 127, 115)
SCAN = (17, 83, 76)

FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            return ImageFont.load_default(size=size)


F_SMALL = font(15, True)
F_MED = font(17, True)
F_BIG = font(27, True)
F_MID = font(16)


def rgba(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (*rgb, alpha)


def background() -> Image.Image:
    im = Image.new("RGBA", (W, H), (*BG, 255))
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(32, W, 32):
        for y in range(28, H, 32):
            d.ellipse((x, y, x + 1, y + 1), fill=(230, 232, 226, 255))
    d.ellipse((680, 42, 1180, 510), outline=(232, 234, 228, 255), width=2)
    d.ellipse((732, 90, 1128, 458), outline=(239, 240, 235, 255), width=2)
    return im


STATIC_BG = background()


def draw_input(d: ImageDraw.ImageDraw, phase: float) -> None:
    # A small stack of image planes suggests sequential observations.
    d.rounded_rectangle((85, 127, 360, 342), radius=12, fill=(237, 238, 233, 255), outline=(200, 204, 197, 255), width=2)
    d.rounded_rectangle((73, 139, 348, 354), radius=12, fill=(245, 246, 241, 255), outline=(197, 202, 194, 255), width=2)
    d.rounded_rectangle((61, 151, 336, 366), radius=12, fill=(255, 255, 252, 255), outline=(139, 149, 141, 255), width=2)

    # Sky and a deliberately imperfect observation of a facade.
    d.rounded_rectangle((68, 158, 329, 358), radius=7, fill=(247, 247, 241, 255))
    d.polygon([(78, 253), (126, 214), (276, 214), (318, 250), (318, 351), (78, 351)], fill=(204, 209, 202, 255))
    d.polygon([(78, 253), (126, 214), (276, 214), (318, 250)], fill=(178, 182, 173, 255))
    for row in range(2):
        for col in range(5):
            x = 105 + col * 39
            y = 269 + row * 40
            fill = (90, 119, 110, 255) if (row, col) not in {(0, 3), (1, 1)} else rgba(ORANGE, 255)
            d.rounded_rectangle((x, y, x + 21, y + 28), radius=2, fill=(238, 240, 234, 255), outline=fill, width=2)
            d.line((x + 11, y + 2, x + 11, y + 26), fill=(168, 179, 168, 255), width=1)

    # Occlusion / label noise: compact jagged shapes rather than a literal photo.
    tree = [(226, 353), (225, 323), (213, 322), (218, 304), (203, 298), (215, 286), (207, 269), (223, 273), (232, 255), (241, 276), (257, 273), (252, 292), (271, 304), (256, 316), (266, 342), (255, 353)]
    d.polygon(tree, fill=(85, 92, 84, 255))
    d.line((235, 351, 234, 303), fill=(63, 69, 64, 255), width=6)
    for x, y in [(259, 271), (276, 292), (267, 307), (235, 285), (212, 303)]:
        d.rectangle((x, y, x + 6, y + 6), fill=rgba(ORANGE, 255))

    # Scanning trace moves continuously across the image card.
    scan_y = 166 + ((phase + 0.08) % 1) * 183
    d.line((70, scan_y, 326, scan_y), fill=rgba(CYAN, 175), width=2)
    d.rounded_rectangle((61, 151, 336, 366), radius=12, outline=(139, 149, 141, 255), width=2)
    d.text((73, 162), "RGB  /  SEMANTICS", font=F_SMALL, fill=WHITE)


def draw_field(d: ImageDraw.ImageDraw, phase: float) -> None:
    cx, cy = 554, 260
    for i, (rx, ry) in enumerate(((114, 87), (88, 62), (61, 40))):
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (phase - i * 0.14))
        ring = int(221 - 23 * pulse)
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(ring, ring + 3, ring - 1, 255), width=2)
    d.arc((cx - 85, cy - 97, cx + 85, cy + 97), 25, 154, fill=rgba(CYAN, 255), width=3)
    d.arc((cx - 85, cy - 97, cx + 85, cy + 97), 205, 337, fill=rgba(CYAN, 255), width=3)
    d.polygon([(cx, cy - 54), (cx + 52, cy - 26), (cx + 52, cy + 27), (cx, cy + 54), (cx - 52, cy + 27), (cx - 52, cy - 26)], outline=rgba(CYAN_SOFT, 255))
    d.line((cx, cy - 54, cx, cy + 54), fill=rgba(CYAN_SOFT, 255), width=2)
    d.line((cx - 52, cy - 26, cx + 52, cy + 27), fill=rgba(CYAN_SOFT, 255), width=2)
    d.line((cx + 52, cy - 26, cx - 52, cy + 27), fill=rgba(CYAN_SOFT, 255), width=2)
    d.text((cx - 26, cy - 18), "SDF", font=F_BIG, fill=WHITE)
    d.text((cx - 56, cy + 105), "RGB σ²  ·  SEM σ²", font=F_MID, fill=ORANGE)


def building_polygons() -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    front = [(747, 218), (969, 218), (969, 383), (747, 383)]
    side = [(969, 218), (1069, 171), (1069, 337), (969, 383)]
    roof = [(747, 218), (846, 171), (1069, 171), (969, 218)]
    return front, side, roof


def draw_building(d: ImageDraw.ImageDraw, phase: float) -> None:
    front, side, roof = building_polygons()
    d.polygon(side, fill=rgba(WALL, 255), outline=(82, 100, 91, 255))
    d.polygon(front, fill=rgba(WALL, 255), outline=(82, 100, 91, 255))
    d.polygon(roof, fill=rgba(ROOF, 255), outline=(101, 107, 98, 255))
    # Layered wall lines convey a reconstructed surface, not a screenshot.
    for y in (275, 329):
        d.line((748, y, 968, y), fill=(175, 184, 173, 255), width=1)
        d.line((970, y, 1068, y - 47), fill=(164, 177, 166, 255), width=1)
    for x in (791, 835, 879, 923):
        d.line((x, 219, x, 382), fill=(184, 191, 180, 255), width=1)

    for row in range(3):
        for col in range(5):
            x = 763 + col * 42
            y = 238 + row * 44
            d.rounded_rectangle((x, y, x + 26, y + 30), radius=2, fill=rgba(WINDOW, 255), outline=(246, 248, 241, 255), width=2)
            d.line((x + 13, y + 2, x + 13, y + 28), fill=(192, 218, 206, 255), width=1)
    for row in range(3):
        for col in range(2):
            x = 984 + col * 39
            y = 232 + row * 44 - col * 18
            d.polygon([(x, y), (x + 23, y - 10), (x + 23, y + 19), (x, y + 29)], fill=rgba(WINDOW, 255), outline=(246, 248, 241, 255))

    d.text((750, 144), "SEMANTIC SURFACE", font=F_SMALL, fill=WHITE)
    for x, color, label in ((750, WALL, "WALL"), (829, ROOF, "ROOF"), (909, WINDOW, "WINDOW"), (1010, ORANGE, "UNCERTAIN")):
        d.rounded_rectangle((x, 410, x + 11, 421), radius=2, fill=rgba(color, 255))
        d.text((x + 17, 405), label, font=font(13, True), fill=MUTED)


def draw_flow(d: ImageDraw.ImageDraw, phase: float) -> None:
    for start, end in (((352, 244), (436, 244)), ((663, 244), (735, 244))):
        d.line((*start, *end), fill=(145, 165, 153, 255), width=2)
        d.polygon([(end[0], end[1]), (end[0] - 8, end[1] - 5), (end[0] - 8, end[1] + 5)], fill=rgba(CYAN_SOFT, 255))
    for offset in (0, 0.28, 0.56, 0.84):
        t = (phase + offset) % 1
        x = 354 + 80 * t
        d.ellipse((x - 2, 242, x + 2, 246), fill=rgba(CYAN, 220))
        x2 = 663 + 69 * t
        d.ellipse((x2 - 3, 241, x2 + 3, 247), fill=rgba(CYAN, 220))
    d.text((400, 216), "OBSERVE", font=font(12, True), fill=MUTED)
    d.text((675, 216), "REBUILD", font=font(12, True), fill=MUTED)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def draw_scan_and_uncertainty(im: Image.Image, phase: float) -> Image.Image:
    """Sweep several cyan probes downward; reveal orange estimates behind them."""
    # Roof edges, facade seams, and side-wall seams are scanned in a staggered
    # top-to-bottom pass. Each point fades in/out at the ends of its path.
    paths = (
        ((846, 171), (747, 218), 0.00, 0.32),
        ((1069, 171), (969, 218), 0.035, 0.32),
        ((747, 218), (747, 383), 0.11, 0.76),
        ((791, 218), (791, 383), 0.14, 0.75),
        ((835, 218), (835, 383), 0.17, 0.74),
        ((879, 218), (879, 383), 0.20, 0.73),
        ((923, 218), (923, 383), 0.23, 0.72),
        ((969, 218), (969, 383), 0.12, 0.76),
        ((1015, 196), (1015, 358), 0.17, 0.76),
        ((1069, 171), (1069, 337), 0.08, 0.82),
    )
    scan_points: list[tuple[float, float, float, float, float]] = []
    for (x0, y0), (x1, y1), delay, length in paths:
        local = (phase - delay) / length
        if 0.0 <= local <= 1.0:
            opacity = math.sin(math.pi * local) ** 0.55
            scan_points.append((x0 + (x1 - x0) * local, y0 + (y1 - y0) * local, x0, y0, opacity))

    # Edge uncertainty follows the diagonal roof boundary and the left/lower
    # frame of a window on the side wall. A separate patch on the front facade
    # represents uncertain reconstruction behind the occluding tree in the input.
    roof_sites = tuple(
        (x, round(218 - 0.47 * (x - 969)), 0.14 + i * 0.03, -0.906, 0.423)
        for i, x in enumerate((1048, 1040, 1032, 1024, 1016, 1008, 1000))
    )
    side_window_sites = (
        (1023, 259, 0.52, 0.0, 1.0), (1023, 265, 0.55, 0.0, 1.0),
        (1023, 271, 0.58, 0.0, 1.0), (1023, 277, 0.61, 0.0, 1.0),
        (1023, 283, 0.64, 0.0, 1.0), (1029, 284, 0.67, 0.918, -0.399),
        (1035, 282, 0.70, 0.918, -0.399), (1041, 279, 0.73, 0.918, -0.399),
        (1046, 277, 0.76, 0.918, -0.399),
    )
    occlusion_sites = (
        (880, 286, 0.49), (893, 294, 0.54), (902, 305, 0.59),
        (890, 315, 0.64), (907, 322, 0.68),
    )
    end_fade = smoothstep((1.0 - phase) / 0.16)
    revealed: list[tuple[int, int, float, float, float]] = []
    for x, y, threshold, tx, ty in roof_sites + side_window_sites:
        strength = smoothstep((phase - threshold) / 0.12) * end_fade
        if strength > 0.001:
            revealed.append((x, y, tx, ty, strength))
    occlusion: list[tuple[int, int, float]] = []
    for x, y, threshold in occlusion_sites:
        strength = smoothstep((phase - threshold) / 0.17) * end_fade
        if strength > 0.001:
            occlusion.append((x, y, strength))

    glow = Image.new("RGBA", (W, H))
    gd = ImageDraw.Draw(glow, "RGBA")
    for x, y, _x0, _y0, strength in scan_points:
        gd.ellipse((x - 9, y - 9, x + 9, y + 9), fill=rgba(SCAN, int(105 * strength)))
    for x, y, _tx, _ty, strength in revealed:
        radius = 4 + 3 * strength
        gd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba(ORANGE, int(90 * strength)))
    for x, y, strength in occlusion:
        radius = 9 + 5 * strength
        gd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba(ORANGE, int(105 * strength)))
    im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(im, "RGBA")
    for x, y, x0, y0, strength in scan_points:
        # A short tracer makes the direction of motion legible in a still frame.
        dx, dy = x - x0, y - y0
        distance = math.hypot(dx, dy) or 1.0
        trail = 13.0
        d.line((x - dx / distance * trail, y - dy / distance * trail, x, y), fill=rgba(SCAN, int(130 * strength)), width=2)
        r = 2.3 + 1.8 * strength
        d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(SCAN, int(245 * strength)))
    for x, y, tx, ty, strength in revealed:
        half_dash = 2.4 + 1.0 * strength
        d.line((x - tx * half_dash, y - ty * half_dash, x + tx * half_dash, y + ty * half_dash), fill=rgba(ORANGE, int(230 * strength)), width=2)
        r = 1.1 + 1.1 * strength
        d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(ORANGE, int(245 * strength)))
    for x, y, strength in occlusion:
        r = 1.5 + 2.4 * strength
        d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(ORANGE, int(245 * strength)))
    return im


def make_frame(index: int) -> Image.Image:
    phase = index / N_FRAMES
    im = STATIC_BG.copy()
    glow = Image.new("RGBA", (W, H))
    gd = ImageDraw.Draw(glow, "RGBA")
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * phase)
    for x, y, radius in ((238, 297, 30),):
        r = radius + 11 * pulse
        gd.ellipse((x - r, y - r, x + r, y + r), fill=(235, 133, 58, int(37 + 28 * pulse)))
    im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle((51, 38, 85, 72), radius=7, fill=rgba(WHITE, 255))
    d.polygon([(61, 63), (68, 48), (75, 63)], fill=(*BG, 255))
    d.text((99, 36), "BUILDNET3D", font=F_BIG, fill=WHITE)
    d.text((100, 75), "UNCERTAINTY-AWARE RECONSTRUCTION", font=F_SMALL, fill=CYAN)
    d.line((53, 101, W - 53, 101), fill=(215, 219, 211, 255), width=1)
    draw_input(d, phase)
    draw_flow(d, phase)
    draw_field(d, phase)
    draw_building(d, phase)
    im = draw_scan_and_uncertainty(im, phase)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((64, 409), "IMPERFECT OBSERVATIONS", font=F_SMALL, fill=MUTED)
    d.text((458, 409), "LEARNED RELIABILITY", font=F_SMALL, fill=MUTED)
    d.line((53, 447, W - 53, 447), fill=(215, 219, 211, 255), width=1)
    d.text((53, 454), "CONCEPTUAL VISUALIZATION", font=font(12, True), fill=MUTED)
    return im.convert("RGB")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(i) for i in range(N_FRAMES)]
    frames[40].save(out / "readme-hero-poster.png", optimize=True)
    # One palette for all frames keeps the loop compact and prevents color flashes.
    samples = Image.new("RGB", (W * 2, H * 2))
    for n, i in enumerate((0, 12, 24, 36)):
        samples.paste(frames[i], ((n % 2) * W, (n // 2) * H))
    palette = samples.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif_frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    gif_frames[0].save(
        out / "readme-hero.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(out / "readme-hero.gif")


if __name__ == "__main__":
    main()
