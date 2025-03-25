#!/usr/bin/env python
"""
Simple script to test the OpenCVVideoWriter class.

This script creates a series of test frames with a moving gradient using parallel processing,
then uses the OpenCVVideoWriter to save them as a video file.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import sys
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.ERROR)

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.video.writers.opencv_writer import OpenCVVideoWriter


def create_frame(frame_index, width=640, height=480):
    """Create a single frame with a rotating gradient.
    
    Args:
        frame_index: Index of the frame to create
        width: Width of the frame
        height: Height of the frame
        
    Returns:
        numpy.ndarray: The created frame
    """
    # Create a blank RGB frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate angle for rotating gradient
    angle = frame_index * 6  # 6 degrees per frame
    angle_rad = np.radians(angle)
    
    # Create coordinate meshgrid for vectorized calculation
    y, x = np.mgrid[0:height, 0:width]
    
    # Normalize coordinates to [-1, 1]
    nx = 2 * x / width - 1
    ny = 2 * y / height - 1
    
    # Rotate coordinates
    rx = nx * np.cos(angle_rad) - ny * np.sin(angle_rad)
    ry = nx * np.sin(angle_rad) + ny * np.cos(angle_rad)
    
    # Create RGB gradient and clip to valid range [0, 255]
    r = np.clip(128 + 127 * rx, 0, 255).astype(np.uint8)
    g = np.clip(128 + 127 * ry, 0, 255).astype(np.uint8)
    b = np.clip(128 + 127 * np.sin(frame_index * 0.1), 0, 255).astype(np.uint8)
    
    # Assign to frame channels
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    
    return frame


def create_test_frames_parallel(num_frames=60, width=640, height=480, num_processes=None):
    """Create synthetic test frames with a moving gradient using parallel processing.
    
    Args:
        num_frames: Number of frames to generate
        width: Width of the frames
        height: Height of the frames
        num_processes: Number of processes to use. If None, uses cpu_count()
        
    Returns:
        list: List of numpy arrays containing the frames
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Generating {num_frames} frames using {num_processes} processes...")
    
    # Create a partial function with fixed width and height
    create_frame_with_size = partial(create_frame, width=width, height=height)
    
    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Use imap to get a generator that can be passed to tqdm
        frames = list(tqdm(
            pool.imap(create_frame_with_size, range(num_frames)),
            total=num_frames,
            desc="Generating frames"
        ))
    
    return frames


def main():
    """Main function to test the OpenCVVideoWriter."""
    # Create output directory if it doesn't exist
    output_dir = Path(project_root) / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate a test video with a rotating gradient.')
    parser.add_argument('--num-frames', type=int, default=60, help='Number of frames to generate')
    parser.add_argument('--width', type=int, default=640, help='Width of the video')
    parser.add_argument('--height', type=int, default=480, help='Height of the video')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second')
    parser.add_argument('--codec', type=str, default=None, help='Codec to use (e.g., mp4v, avc1, h264)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for frame generation')
    parser.add_argument('--output-path', type=str, default="test_gradient.mp4", help='Output path for the video')
    args = parser.parse_args()
    
    # Set output file path
    output_path = output_dir / f"{args.output_path}"
    
    # Create test frames using parallel processing
    frames = create_test_frames_parallel(
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_processes=args.processes
    )
    
    # Initialize writer with the requested codec
    print(f"Initializing video writer with codec '{args.codec}' to save to {output_path}...")
    
    writer = OpenCVVideoWriter(
        output_path=output_path,
        fps=args.fps,
        frame_size=(args.width, args.height),
        codec=args.codec
    )
    
    # Write frames
    print("Writing frames to video...")
    with writer:
        for frame in tqdm(frames, desc="Writing frames"):
            writer.write_frame(frame)
    
    # Verify output
    print("\nVideo writing complete!")
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Video saved to: {output_path} ({size_mb:.2f} MB)")
        print(f"Actual codec used: {writer.output_codec}")
        
        # Try to play the video using OpenCV to verify it works
        print("\nVerifying video can be read back...")
        cap = cv2.VideoCapture(str(output_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video properties:")
            print(f"  - Frame count: {frame_count}")
            print(f"  - FPS: {fps}")
            print(f"  - Resolution: {width}x{height}")
            
            # Read first frame to verify content
            ret, frame = cap.read()
            if ret:
                print("Successfully read the first frame.")
            else:
                print("Failed to read the first frame.")
            
            cap.release()
        else:
            print("Failed to open the video file for verification.")
    else:
        print(f"Error: Video file was not created at {output_path}")


if __name__ == "__main__":
    # Best practice for multiprocessing
    mp.set_start_method('spawn', force=True)
    main() 