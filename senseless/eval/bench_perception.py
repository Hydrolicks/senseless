"""On-Pi benchmark: throughput (FPS) and CPU% for each perception backend.

Run this ON THE PI to confirm the perception-backend choice documented in
``senseless/sign/README.md``. It pushes frames from a camera or video file
through a backend plus the landmark normalization, and reports FPS and CPU%.

Examples
--------
    python -m senseless.eval.bench_perception --frames 300
    python -m senseless.eval.bench_perception --backend tasks --video clip.mp4
    python -m senseless.eval.bench_perception --backend holistic --source 0

CPU% is process CPU time over wall time, so 100% = one core fully busy; on the
4-core Pi what matters is the budget left for ASR + the classifier.
"""

from __future__ import annotations

import argparse
import time

from senseless.sign import capture, landmarks


def _bench_one(backend_name: str, frames: int, source: int, video: str | None) -> dict:
    src = (
        capture.OpenCVSource(video)
        if video is not None
        else capture.open_frame_source(source=source)
    )
    backend = landmarks.create_backend(backend_name)
    count = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with backend, src:
        for frame in src.frames():
            ts_ms = int((time.perf_counter() - wall_start) * 1000)
            raw = backend.extract(frame, ts_ms)
            landmarks.frame_landmarks_to_vector(raw)
            count += 1
            if count >= frames:
                break
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return {
        "backend": backend_name,
        "frames": count,
        "fps": count / wall if wall > 0 else 0.0,
        "cpu_pct": 100.0 * cpu / wall if wall > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark perception backends.")
    parser.add_argument(
        "--backend",
        choices=["tasks", "holistic"],
        default=None,
        help="Backend to bench; omit to bench both.",
    )
    parser.add_argument("--frames", type=int, default=300, help="Frames per backend.")
    parser.add_argument("--source", type=int, default=0, help="Camera index (OpenCV/auto).")
    parser.add_argument("--video", default=None, help="Video file path (forces OpenCV).")
    args = parser.parse_args()

    names = [args.backend] if args.backend else ["tasks", "holistic"]
    rows = []
    for name in names:
        print(f"Benchmarking {name} ...", flush=True)
        rows.append(_bench_one(name, args.frames, args.source, args.video))

    print(f"\n{'backend':<10}{'frames':>8}{'fps':>10}{'cpu%':>10}")
    for row in rows:
        print(f"{row['backend']:<10}{row['frames']:>8}{row['fps']:>10.1f}{row['cpu_pct']:>10.1f}")


if __name__ == "__main__":
    main()
