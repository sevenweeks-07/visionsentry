import os
import time
from pathlib import Path

def calculate_fps(stream_dir):
    p = Path(stream_dir)
    if not p.exists():
        print(f"Directory {stream_dir} does not exist.")
        return

    frames = sorted(list(p.glob("frame_*.jpg")))
    if not frames:
        print("No frames found.")
        return

    count = len(frames)
    first_time = frames[0].stat().st_mtime
    last_time = frames[-1].stat().st_mtime
    
    duration = last_time - first_time
    if duration <= 0:
        print(f"Found {count} frames, but duration is 0 (all written at same time?).")
        return

    fps = count / duration
    print(f"Stream: {stream_dir}")
    print(f"  Frames: {count}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Processed FPS: {fps:.2f}")

if __name__ == "__main__":
    calculate_fps("data/cache/frames/stream_0")
