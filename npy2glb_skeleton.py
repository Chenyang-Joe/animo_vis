#!/usr/bin/env python3
"""CLI: Convert AniMo4D joint .npy files to animated .glb files."""

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from core.converter import convert


def gather_inputs(paths, mode):
    """Resolve input paths to a list of .npy files."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".npy":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.npy")))
        else:
            print(f"Warning: skipping {p} (not a .npy file or directory)", file=sys.stderr)
    return files


def convert_one(args):
    """Wrapper for ProcessPoolExecutor."""
    npy_path, output_path, mode, fps, sphere_radius, max_frames = args
    try:
        convert(npy_path, output_path, mode=mode, fps=fps,
                sphere_radius=sphere_radius, max_frames=max_frames)
        return str(npy_path), None
    except Exception as e:
        return str(npy_path), str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert AniMo4D joint .npy files to animated .glb"
    )
    parser.add_argument("input", nargs="+", help="Input .npy file(s) or directory")
    parser.add_argument("--mode", choices=["joints", "vecs"], default="joints",
                        help="'joints' reads (T,30,3) directly; 'vecs' decodes (T,359)")
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS (default: 30)")
    parser.add_argument("--radius", type=float, default=0.008,
                        help="Joint sphere radius (default: 0.008)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for .glb files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .glb files")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Truncate animations to N frames")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N files from input")

    args = parser.parse_args()

    files = gather_inputs(args.input, args.mode)
    if not files:
        print("No .npy files found.", file=sys.stderr)
        sys.exit(1)

    if args.sample is not None and args.sample < len(files):
        files = random.sample(files, args.sample)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for f in files:
        out_path = output_dir / f"{f.stem}.glb"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping {f.name} (exists, use --overwrite)")
            continue
        tasks.append((str(f), str(out_path), args.mode, args.fps, args.radius, args.max_frames))

    if not tasks:
        print("Nothing to do.")
        return

    print(f"Converting {len(tasks)} file(s) with {args.workers} worker(s)...")

    if args.workers <= 1:
        for task in tasks:
            name, err = convert_one(task)
            if err:
                print(f"FAIL: {Path(name).name}: {err}", file=sys.stderr)
            else:
                print(f"OK: {Path(name).name}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, t): t for t in tasks}
            for future in as_completed(futures):
                name, err = future.result()
                if err:
                    print(f"FAIL: {Path(name).name}: {err}", file=sys.stderr)
                else:
                    print(f"OK: {Path(name).name}")

    print("Done.")


if __name__ == "__main__":
    main()
