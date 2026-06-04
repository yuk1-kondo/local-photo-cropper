#!/usr/bin/env python3
"""
Batch-crop student ID photos to a 3cm x 4cm JPEG under a size limit.

The script keeps all processing local:
  - detects faces with OpenCV YuNet
  - crops from the original high-resolution image
  - writes review cases separately
  - exports a CSV report and contact-sheet previews
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class FaceDetection:
    x: float
    y: float
    w: float
    h: float
    score: float


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def cm_to_px(cm: float, dpi: int) -> int:
    return int(round(cm / 2.54 * dpi))


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_images(input_dir: Path, recursive: bool, excluded_dirs: list[Path]) -> Iterable[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in sorted(paths):
        resolved_path = path.resolve(strict=False)
        if any(is_relative_to(resolved_path, excluded) for excluded in excluded_dirs):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def relative_output_path(source_path: Path, input_dir: Path, output_dir: Path, suffix: str) -> Path:
    relative = source_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{source_path.stem}{suffix}"


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def create_detector(model_path: Path, input_size: tuple[int, int], score_threshold: float):
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m pip install opencv-python"
        )
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "This OpenCV build does not include FaceDetectorYN. "
            "Install a recent opencv-python package."
        )
    if not model_path.exists():
        raise RuntimeError(
            f"YuNet model not found: {model_path}\n"
            "Download it with:\n"
            "curl -L -o models/face_detection_yunet_2023mar.onnx "
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
    return cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        input_size,
        score_threshold,
        0.3,
        5000,
    )


def detect_faces(
    image: Image.Image,
    model_path: Path,
    detect_max_side: int,
    score_threshold: float,
) -> list[FaceDetection]:
    src_w, src_h = image.size
    scale = min(1.0, detect_max_side / max(src_w, src_h))
    det_w = max(1, int(round(src_w * scale)))
    det_h = max(1, int(round(src_h * scale)))

    if scale < 1.0:
        detect_image = image.resize((det_w, det_h), Image.Resampling.LANCZOS)
    else:
        detect_image = image

    rgb = np.array(detect_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector = create_detector(model_path, (det_w, det_h), score_threshold)
    _, faces = detector.detect(bgr)

    if faces is None:
        return []

    inv_scale = 1.0 / scale
    detections: list[FaceDetection] = []
    for row in faces:
        x, y, w, h, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[14])
        detections.append(
            FaceDetection(
                x=x * inv_scale,
                y=y * inv_scale,
                w=w * inv_scale,
                h=h * inv_scale,
                score=score,
            )
        )
    detections.sort(key=lambda face: face.score, reverse=True)
    return detections


def make_crop_box(
    face: FaceDetection,
    image_size: tuple[int, int],
    output_size: tuple[int, int],
    face_height_ratio: float,
    headroom_ratio: float,
) -> CropBox:
    img_w, img_h = image_size
    out_w, out_h = output_size
    aspect = out_w / out_h

    crop_h = face.h / face_height_ratio
    crop_w = crop_h * aspect

    # Keep the face center horizontally. Put more room above the detector box
    # because the box may not include all hair volume.
    center_x = face.x + face.w / 2
    left = center_x - crop_w / 2
    top = face.y - face.h * headroom_ratio

    box = CropBox(
        left=int(round(left)),
        top=int(round(top)),
        right=int(round(left + crop_w)),
        bottom=int(round(top + crop_h)),
    )

    # Do not silently pad. If the desired proof-photo frame does not fit in the
    # original image, review is safer than inventing background.
    if box.left < 0 or box.top < 0 or box.right > img_w or box.bottom > img_h:
        return box
    return box


def is_box_inside(box: CropBox, image_size: tuple[int, int]) -> bool:
    img_w, img_h = image_size
    return box.left >= 0 and box.top >= 0 and box.right <= img_w and box.bottom <= img_h


def save_jpeg_under_limit(
    image: Image.Image,
    output_path: Path,
    max_kb: int,
    dpi: int,
    min_quality: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = max_kb * 1024
    low, high = min_quality, 95
    best_data: bytes | None = None
    best_quality = min_quality

    while low <= high:
        quality = (low + high) // 2
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
            dpi=(dpi, dpi),
        )
        data = buffer.getvalue()
        if len(data) <= max_bytes:
            best_data = data
            best_quality = quality
            low = quality + 1
        else:
            high = quality - 1

    if best_data is None:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=min_quality,
            optimize=True,
            progressive=True,
            subsampling=2,
            dpi=(dpi, dpi),
        )
        best_data = buffer.getvalue()
        best_quality = min_quality

    output_path.write_bytes(best_data)
    return best_quality, len(best_data)


def draw_review_image(
    image: Image.Image,
    output_path: Path,
    faces: list[FaceDetection],
    crop_box: CropBox | None,
    reason: str,
) -> None:
    preview = image.copy()
    preview.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height

    draw = ImageDraw.Draw(preview)
    for face in faces:
        rect = [
            face.x * scale_x,
            face.y * scale_y,
            (face.x + face.w) * scale_x,
            (face.y + face.h) * scale_y,
        ]
        draw.rectangle(rect, outline=(0, 180, 0), width=4)

    if crop_box is not None:
        rect = [
            crop_box.left * scale_x,
            crop_box.top * scale_y,
            crop_box.right * scale_x,
            crop_box.bottom * scale_y,
        ]
        draw.rectangle(rect, outline=(220, 0, 0), width=4)

    draw.rectangle([0, 0, preview.width, 36], fill=(255, 255, 255))
    draw.text((10, 10), reason, fill=(180, 0, 0))
    preview.save(output_path, format="JPEG", quality=88, optimize=True)


def make_contact_sheet(
    items: list[tuple[Path, Path, str]],
    output_path: Path,
    thumb_size: tuple[int, int] = (160, 214),
    columns: int = 4,
) -> None:
    if not items:
        return

    font = ImageFont.load_default()
    cell_w = thumb_size[0] * 2 + 24
    cell_h = thumb_size[1] + 48
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (before_path, after_path, label) in enumerate(items):
        row, col = divmod(index, columns)
        x0 = col * cell_w
        y0 = row * cell_h

        for offset, path in ((8, before_path), (thumb_size[0] + 16, after_path)):
            try:
                img = load_image(path)
            except Exception:
                continue
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            frame = Image.new("RGB", thumb_size, (245, 245, 245))
            px = (thumb_size[0] - img.width) // 2
            py = (thumb_size[1] - img.height) // 2
            frame.paste(img, (px, py))
            sheet.paste(frame, (x0 + offset, y0 + 8))

        display_label = label[:46]
        draw.text((x0 + 8, y0 + thumb_size[1] + 16), display_label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=85, optimize=True)


def safe_review_copy(source_path: Path, input_dir: Path, review_dir: Path, reason: str) -> Path:
    safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason)
    target = relative_output_path(
        source_path,
        input_dir,
        review_dir,
        f"__{safe_reason}{source_path.suffix.lower()}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return target


def process_one(
    source_path: Path,
    args: argparse.Namespace,
    output_size: tuple[int, int],
) -> dict[str, object]:
    input_dir = Path(args.input)
    ok_dir = Path(args.output_ok)
    review_dir = Path(args.output_review)
    review_dir.mkdir(parents=True, exist_ok=True)
    ok_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = load_image(source_path)
    except (UnidentifiedImageError, OSError) as exc:
        safe_review_copy(source_path, input_dir, review_dir, "read_error")
        return {
            "file": str(source_path.relative_to(input_dir)),
            "status": "review",
            "reason": f"read_error: {exc}",
            "faces": 0,
            "score": "",
            "quality": "",
            "bytes": "",
        }

    faces = detect_faces(
        image=image,
        model_path=Path(args.model),
        detect_max_side=args.detect_max_side,
        score_threshold=args.score_threshold,
    )

    crop_box: CropBox | None = None
    reason = ""
    if len(faces) == 0:
        reason = "no_face"
    elif len(faces) > 1 and not args.use_largest_when_multiple:
        reason = "multiple_faces"
    else:
        face = faces[0]
        crop_box = make_crop_box(
            face=face,
            image_size=image.size,
            output_size=output_size,
            face_height_ratio=args.face_height_ratio,
            headroom_ratio=args.headroom_ratio,
        )
        if face.h < args.min_face_px:
            reason = "face_too_small"
        elif not is_box_inside(crop_box, image.size):
            reason = "crop_out_of_bounds"

    if reason:
        review_original = safe_review_copy(source_path, input_dir, review_dir, reason)
        review_preview = relative_output_path(
            source_path,
            input_dir,
            review_dir,
            f"__{reason}_preview.jpg",
        )
        review_preview.parent.mkdir(parents=True, exist_ok=True)
        draw_review_image(image, review_preview, faces, crop_box, reason)
        return {
            "file": str(source_path.relative_to(input_dir)),
            "status": "review",
            "reason": reason,
            "faces": len(faces),
            "score": f"{faces[0].score:.4f}" if faces else "",
            "quality": "",
            "bytes": "",
            "review_file": str(review_original.relative_to(review_dir)),
        }

    assert crop_box is not None
    cropped = image.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))
    resized = cropped.resize(output_size, Image.Resampling.LANCZOS)
    output_path = relative_output_path(source_path, input_dir, ok_dir, ".jpg")
    quality, byte_count = save_jpeg_under_limit(
        resized,
        output_path,
        max_kb=args.max_kb,
        dpi=args.dpi,
        min_quality=args.min_quality,
    )

    status = "ok" if byte_count <= args.max_kb * 1024 else "review"
    reason = "" if status == "ok" else "size_limit_exceeded"
    if status == "review":
        review_path = relative_output_path(source_path, input_dir, review_dir, ".jpg")
        review_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_path), review_path)

    return {
        "file": str(source_path.relative_to(input_dir)),
        "status": status,
        "reason": reason,
        "faces": len(faces),
        "score": f"{faces[0].score:.4f}",
        "quality": quality,
        "bytes": byte_count,
        "output_file": str(output_path.relative_to(ok_dir)),
    }


def write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "status",
        "reason",
        "faces",
        "score",
        "quality",
        "bytes",
        "output_file",
        "review_file",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop student ID photos to 3cm x 4cm JPEG files under 100KB."
    )
    parser.add_argument("--input", default="input", help="Folder containing original photos.")
    parser.add_argument("--recursive", action="store_true", help="Process images inside subfolders too.")
    parser.add_argument("--output-ok", default="output_ok", help="Folder for successful cropped photos.")
    parser.add_argument("--output-review", default="output_review", help="Folder for review cases.")
    parser.add_argument("--preview", default="preview_sheet/preview.jpg", help="Before/after contact sheet path.")
    parser.add_argument("--report", default="report.csv", help="CSV processing report path.")
    parser.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx", help="YuNet ONNX model path.")
    parser.add_argument("--width-cm", type=float, default=3.0, help="Printed photo width in cm.")
    parser.add_argument("--height-cm", type=float, default=4.0, help="Printed photo height in cm.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI used to convert cm to pixels.")
    parser.add_argument("--max-kb", type=int, default=100, help="Maximum output JPEG size in KB.")
    parser.add_argument("--min-quality", type=int, default=35, help="Minimum JPEG quality during size search.")
    parser.add_argument("--detect-max-side", type=int, default=1280, help="Long side used for face detection.")
    parser.add_argument("--score-threshold", type=float, default=0.85, help="YuNet face confidence threshold.")
    parser.add_argument("--face-height-ratio", type=float, default=0.42, help="Face bbox height as a ratio of crop height.")
    parser.add_argument("--headroom-ratio", type=float, default=0.55, help="Space above face bbox, in face-height units.")
    parser.add_argument("--min-face-px", type=int, default=120, help="Review if detected face is smaller than this in original pixels.")
    parser.add_argument(
        "--use-largest-when-multiple",
        action="store_true",
        help="Crop the highest-score face when multiple faces are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        print(f"Created input folder: {input_dir}")
        print("Put photos there and run this script again.")
        return 1

    output_size = (cm_to_px(args.width_cm, args.dpi), cm_to_px(args.height_cm, args.dpi))
    print(
        f"Output: {args.width_cm:g}cm x {args.height_cm:g}cm, "
        f"{output_size[0]} x {output_size[1]} px, {args.dpi} dpi, <= {args.max_kb}KB"
    )

    rows: list[dict[str, object]] = []
    preview_items: list[tuple[Path, Path, str]] = []
    excluded_dirs = [
        Path(args.output_ok).resolve(strict=False),
        Path(args.output_review).resolve(strict=False),
        Path(args.preview).parent.resolve(strict=False),
        Path(args.model).parent.resolve(strict=False),
        Path(".venv").resolve(strict=False),
    ]
    image_paths = list(iter_images(input_dir, args.recursive, excluded_dirs))
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return 1

    for index, source_path in enumerate(image_paths, start=1):
        relative_name = source_path.relative_to(input_dir)
        print(f"[{index}/{len(image_paths)}] {relative_name}")
        row = process_one(source_path, args, output_size)
        rows.append(row)
        if row.get("status") == "ok":
            after = Path(args.output_ok) / str(row["output_file"])
            preview_items.append((source_path, after, f"OK {relative_name}"))

    write_report(rows, Path(args.report))
    make_contact_sheet(preview_items, Path(args.preview))

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    review_count = sum(1 for row in rows if row.get("status") == "review")
    print(f"Done. ok={ok_count}, review={review_count}")
    print(f"Report: {args.report}")
    print(f"Preview: {args.preview}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
