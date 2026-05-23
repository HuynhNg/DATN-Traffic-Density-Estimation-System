"""
extract_frames.py
-----------------
Cắt hình ảnh từ tất cả video MP4/WebM trong thư mục đầu vào,
với tần số mỗi INTERVAL_SEC giây một frame.

Cấu trúc output:
    output_frames/
    └── <tên_video>/
        ├── frame_000000s.jpg   ← t = 0s
        ├── frame_000020s.jpg   ← t = 20s
        ├── frame_000040s.jpg   ← t = 40s
        └── ...

Cách dùng:
    Chỉnh 3 biến ở phần CONFIG bên dưới rồi chạy:
        python extract_frames.py
"""

import cv2
from pathlib import Path

# ╔══════════════════════════════════════════════╗
# ║               C Ấ U   H Ì N H               ║
# ╚══════════════════════════════════════════════╝

# Thư mục chứa các file video (.mp4 / .webm)
INPUT_DIR    = r"D:\2026\DATN\Data\video"

# Thư mục lưu ảnh kết quả (tự tạo nếu chưa có)
OUTPUT_DIR   = r"D:\2026\DATN\Data\output_frames"

# Khoảng cách giữa hai frame cần lưu (giây)
INTERVAL_SEC = 20

# ╚══════════════════════════════════════════════╝


def extract_frames(video_path: Path, output_dir: Path, interval_sec: int):
    """
    Trích xuất frame từ một video theo khoảng thời gian cố định.

    Args:
        video_path   : Đường dẫn tới file video (.mp4 / .webm)
        output_dir   : Thư mục gốc lưu ảnh output
        interval_sec : Khoảng cách giữa hai frame (giây)

    Returns:
        int: Số frame đã lưu
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [LỖI] Không mở được: {video_path.name}")
        return 0

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Tạo thư mục con riêng cho mỗi video (tên thư mục = tên file không có đuôi)
    video_out_dir = output_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    interval_frames = int(fps * interval_sec)
    saved_count     = 0
    current_sec     = 0

    print(f"\n  Video   : {video_path.name}")
    print(f"  FPS     : {fps:.2f}  |  Tổng frame: {total_frames}  |  Thời lượng: {duration_sec:.1f}s")
    print(f"  Interval: mỗi {interval_sec}s (≈ {interval_frames} frame/bước)")
    print(f"  Output  : {video_out_dir}")

    while True:
        target_frame = int(current_sec * fps)
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        filename = video_out_dir / f"frame_{current_sec:06d}s.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    Đã lưu: {filename.name}  (t = {current_sec}s)")

        saved_count += 1
        current_sec += interval_sec

    cap.release()
    print(f"  → Tổng: {saved_count} frame đã lưu.")
    return saved_count


def process_folder():
    """Duyệt INPUT_DIR, tìm toàn bộ .mp4 / .webm và xử lý từng file."""
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[LỖI] Thư mục đầu vào không tồn tại: {input_path.resolve()}")
        return

    # Tìm đệ quy cả hai định dạng
    video_files = sorted(
        list(input_path.glob("**/*.mp4")) +
        list(input_path.glob("**/*.webm"))
    )

    if not video_files:
        print(f"[CẢNH BÁO] Không tìm thấy file .mp4 hoặc .webm nào trong: {input_path.resolve()}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Thư mục video   : {input_path.resolve()}")
    print(f"Thư mục output  : {output_path.resolve()}")
    print(f"Interval        : {INTERVAL_SEC} giây / frame")
    print(f"Số video tìm thấy: {len(video_files)}")
    print("=" * 60)

    total_saved = 0
    for idx, video_path in enumerate(video_files, 1):
        video_out_dir = output_path / video_path.stem
        existing_frames = list(video_out_dir.glob("*.jpg")) if video_out_dir.exists() else []

        if existing_frames:
            print(f"\n[{idx}/{len(video_files)}] Bỏ qua (đã có {len(existing_frames)} ảnh): {video_path.name}")
            continue

        print(f"\n[{idx}/{len(video_files)}] Đang xử lý...")
        total_saved += extract_frames(video_path, output_path, INTERVAL_SEC)

    print("\n" + "=" * 60)
    print(f"HOÀN THÀNH — {total_saved} frame từ {len(video_files)} video.")
    print(f"Ảnh lưu tại    : {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    process_folder()
