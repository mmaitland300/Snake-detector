from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a public-safe placeholder dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/public_demo_raw"))
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def generate_demo_dataset(
    output_dir: Path,
    samples_per_class: int = 60,
    image_size: int = 160,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    for class_name in ("snake", "no_snake"):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(samples_per_class):
            image = _make_background(image_size, rng)
            draw = ImageDraw.Draw(image)
            if class_name == "snake":
                _draw_snake_like_shape(draw, image_size, rng)
            else:
                _draw_non_snake_scene(draw, image_size, rng)
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.2)))
            image.save(class_dir / f"{class_name}_{idx:03d}.png")


def _make_background(image_size: int, rng: random.Random) -> Image.Image:
    base = Image.new(
        "RGB",
        (image_size, image_size),
        (
            rng.randint(180, 235),
            rng.randint(180, 235),
            rng.randint(170, 230),
        ),
    )
    draw = ImageDraw.Draw(base)
    for _ in range(18):
        x0 = rng.randint(0, image_size)
        y0 = rng.randint(0, image_size)
        x1 = x0 + rng.randint(12, 42)
        y1 = y0 + rng.randint(12, 42)
        color = (
            rng.randint(120, 220),
            rng.randint(120, 220),
            rng.randint(120, 220),
        )
        draw.ellipse((x0, y0, x1, y1), fill=color)
    return base


def _draw_snake_like_shape(draw: ImageDraw.ImageDraw, image_size: int, rng: random.Random) -> None:
    points: list[tuple[int, int]] = []
    amplitude = rng.randint(image_size // 10, image_size // 5)
    start_y = rng.randint(image_size // 4, 3 * image_size // 4)
    for step in range(7):
        x = int((step / 6) * (image_size - 24)) + 12
        y = int(start_y + math.sin(step * 0.85 + rng.random()) * amplitude)
        points.append((x, max(12, min(image_size - 12, y))))

    thickness = rng.randint(12, 22)
    body_color = (
        rng.randint(40, 90),
        rng.randint(90, 150),
        rng.randint(30, 90),
    )
    stripe_color = (
        min(body_color[0] + 50, 255),
        min(body_color[1] + 40, 255),
        min(body_color[2] + 50, 255),
    )
    draw.line(points, fill=body_color, width=thickness, joint="curve")
    for point in points[1:-1]:
        draw.ellipse(
            (
                point[0] - thickness // 3,
                point[1] - thickness // 3,
                point[0] + thickness // 3,
                point[1] + thickness // 3,
            ),
            fill=stripe_color,
        )

    head_x, head_y = points[-1]
    draw.ellipse(
        (
            head_x - thickness,
            head_y - thickness,
            head_x + thickness,
            head_y + thickness,
        ),
        fill=body_color,
    )
    eye_offset = max(2, thickness // 5)
    draw.ellipse(
        (
            head_x + eye_offset,
            head_y - eye_offset,
            head_x + eye_offset + 3,
            head_y - eye_offset + 3,
        ),
        fill="white",
    )


def _draw_non_snake_scene(draw: ImageDraw.ImageDraw, image_size: int, rng: random.Random) -> None:
    for _ in range(rng.randint(5, 8)):
        shape = rng.choice(["rectangle", "ellipse", "line"])
        color = (
            rng.randint(20, 230),
            rng.randint(20, 230),
            rng.randint(20, 230),
        )
        if shape == "rectangle":
            x0 = rng.randint(0, image_size - 30)
            y0 = rng.randint(0, image_size - 30)
            x1 = x0 + rng.randint(16, 60)
            y1 = y0 + rng.randint(16, 60)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=rng.randint(4, 16), fill=color)
        elif shape == "ellipse":
            x0 = rng.randint(0, image_size - 30)
            y0 = rng.randint(0, image_size - 30)
            x1 = x0 + rng.randint(14, 52)
            y1 = y0 + rng.randint(14, 52)
            draw.ellipse((x0, y0, x1, y1), fill=color)
        else:
            points = [
                (rng.randint(0, image_size), rng.randint(0, image_size))
                for _ in range(rng.randint(2, 4))
            ]
            draw.line(points, fill=color, width=rng.randint(4, 9))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_demo_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Generated placeholder dataset at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
