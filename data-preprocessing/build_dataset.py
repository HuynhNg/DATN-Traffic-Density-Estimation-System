"""
build_dataset.py
----------------
Tao dataset chuan tu 2 thu muc dau vao:
- Thu muc anh theo tung video
- Thu muc label tuong ung

Output:
    dataset/
    ├── images/
    └── labels/

Doi ten theo format:
    <video_name>_frame_XXXXX.jpg
    <video_name>_frame_XXXXX.txt
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

# ╔══════════════════════════════════════════════╗
# ║               C A U   H I N H               ║
# ╚══════════════════════════════════════════════╝

# Thu muc chua anh theo tung video (moi video la 1 thu muc con)
IMAGE_ROOT_DIR = r"D:\2026\DATN\Data\output_frames"

# Thu muc chua label theo tung video (moi video la 1 thu muc con)
LABEL_ROOT_DIR = r"D:\2026\DATN\Data\label"

# Thu muc dataset dau ra
OUTPUT_DATASET_DIR = r"D:\2026\DATN\Data\dataset"

# Dinh dang so thu tu frame
FRAME_NUMBER_WIDTH = 5

# Extensions hop le cho anh
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ╚══════════════════════════════════════════════╝


def _natural_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", name)
    if match:
        return (int(match.group(1)), name)
    return (0, name)


def _collect_images(video_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(video_dir.glob(f"*{ext}"))
    return sorted(files, key=lambda p: _natural_key(p.stem))


def _collect_labels(label_dir: Path) -> Dict[str, Path]:
    label_map: Dict[str, Path] = {}
    for label_path in label_dir.rglob("*.txt"):
        label_map[label_path.stem.lower()] = label_path
    return label_map


def _ensure_dirs(root: Path) -> Tuple[Path, Path]:
    images_out = root / "images"
    labels_out = root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out


def _video_has_labels(label_dir: Path) -> bool:
    return any(label_dir.rglob("*.txt"))


def _safe_copy(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _label_has_content(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    if label_path.stat().st_size == 0:
        return False
    try:
        return label_path.read_text(encoding="utf-8").strip() != ""
    except UnicodeDecodeError:
        return label_path.read_text(errors="ignore").strip() != ""


def build_dataset(image_root: Path, label_root: Path, output_root: Path) -> None:
    if not image_root.exists():
        print(f"[LOI] Khong ton tai thu muc anh: {image_root}")
        return
    if not label_root.exists():
        print(f"[LOI] Khong ton tai thu muc label: {label_root}")
        return

    images_out, labels_out = _ensure_dirs(output_root)

    video_dirs = [d for d in sorted(image_root.iterdir()) if d.is_dir()]
    if not video_dirs:
        print(f"[CANH BAO] Khong tim thay thu muc video trong: {image_root}")
        return

    total_pairs = 0
    skipped_videos = 0

    for video_dir in video_dirs:
        video_name = video_dir.name
        label_dir = label_root / video_name

        if not label_dir.exists() or not _video_has_labels(label_dir):
            print(f"[BO QUA] Video khong co label: {video_name}")
            skipped_videos += 1
            continue

        image_files = _collect_images(video_dir)
        if not image_files:
            print(f"[BO QUA] Khong co anh trong: {video_dir}")
            skipped_videos += 1
            continue

        label_map = _collect_labels(label_dir)
        if not label_map:
            print(f"[BO QUA] Khong co label hop le trong: {label_dir}")
            skipped_videos += 1
            continue

        frame_index = 1
        for image_path in image_files:
            label_path = label_map.get(image_path.stem.lower())
            if not label_path:
                print(f"  [THIEU] Khong tim thay label cho: {image_path.name}")
                continue

            if not _label_has_content(label_path):
                print(f"  [TRONG] Label rong: {label_path.name}")
                continue

            frame_str = str(frame_index).zfill(FRAME_NUMBER_WIDTH)
            image_out = images_out / f"{video_name}_frame_{frame_str}{image_path.suffix.lower()}"
            label_out = labels_out / f"{video_name}_frame_{frame_str}.txt"

            image_copied = _safe_copy(image_path, image_out)
            label_copied = _safe_copy(label_path, label_out)

            if image_copied and label_copied:
                total_pairs += 1
                frame_index += 1
            else:
                # Neu da ton tai, khong tang frame_index de tranh lech mapping
                print(f"  [TRUNG] Da ton tai: {image_out.name} hoac {label_out.name}")

        print(f"[XONG] {video_name}: {frame_index - 1} cap image/label")

    print("=" * 60)
    print(f"Tong cap duoc tao: {total_pairs}")
    print(f"Video bi bo qua: {skipped_videos}")
    print(f"Dataset output: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset(Path(IMAGE_ROOT_DIR), Path(LABEL_ROOT_DIR), Path(OUTPUT_DATASET_DIR))
