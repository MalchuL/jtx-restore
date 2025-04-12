#!/usr/bin/env python3
"""
Example script demonstrating the use of the PracticalRIFEFrameInterpolator425 class
for frame interpolation.
"""

import argparse
import os
import logging
import cv2
import numpy as np
from typing import List, Optional

from src.core.video.processors.frame_interpolation.ai.rife import PracticalRIFEFrameInterpolator425
from src.core.video.processors.frame_interpolation.interpolated_frame import InterpolatedFrame
from src.core.video.processors.frame_interpolation.interpolators.opencv_interpolator import OpenCVFrameInterpolator


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[InterpolatedFrame]:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None for all frames)
        
    Returns:
        List of InterpolatedFrame objects
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading video from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create InterpolatedFrame
        interpolated_frame = InterpolatedFrame(
            data=frame_rgb,
            frame_id=float(frame_count)
        )
        frames.append(interpolated_frame)
        
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    
    return frames


def save_video(frames: List[InterpolatedFrame], output_path: str, fps: float = 30.0):
    """
    Save frames to a video file.
    
    Args:
        frames: List of InterpolatedFrame objects
        output_path: Path to save the video file
        fps: Frames per second for the output video
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving video to {output_path}")
    
    if not frames:
        logger.warning("No frames to save")
        return
    
    # Get frame dimensions
    height, width = frames[0].data.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        print(frame.frame_id, frame.dt)
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    logger.info(f"Saved video with {len(frames)} frames at {fps} FPS")


def interpolate_video(
    input_path: str,
    output_path: str,
    model_path: str,
    factor: float = 2.0,
    scale: float = 1.0,
    max_frames: Optional[int] = None,
    fps: float = 30.0
):
    """
    Interpolate frames in a video using the RIFE model.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video file
        model_path: Path to the RIFE model directory
        factor: Frame rate increase factor
        scale: Scale factor for the model
        max_frames: Maximum number of frames to process
        fps: Frames per second for the output video
    """
    logger = logging.getLogger(__name__)
    
    # Load frames from video
    frames = load_video_frames(input_path, max_frames)
    
    # Create interpolator
    interpolator = PracticalRIFEFrameInterpolator425(
        factor=factor,
        model_path=model_path,
        scale=scale
    )
    # interpolator = OpenCVFrameInterpolator(
    #     factor=factor,
    # )
    
    # Process frames
    interpolated_frames = []
    for i, frame in enumerate(frames):
        logger.info(f"Processing frame {i+1}/{len(frames)}")
        
        # Process frame
        result = interpolator(frame)
        
        # Add interpolated frames to output
        if result.ready:
            interpolated_frames.extend(result.frames)
        print(len(interpolated_frames))
    
    # Process remaining frames
    result = interpolator.finish()
    if result.ready:
        interpolated_frames.extend(result.frames)
    
    # Save interpolated video
    save_video(interpolated_frames, output_path, fps * factor)
    
    logger.info(f"Interpolation complete. Output saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interpolate frames in a video using RIFE")
    parser.add_argument("--input", "-i", required=True, help="Path to the input video file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the output video file")
    parser.add_argument("--model", "-m", required=True, help="Path to the RIFE model directory")
    parser.add_argument("--factor", "-f", type=float, default=2.0, help="Frame rate increase factor")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Scale factor for the model")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for the output video")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Interpolate video
    interpolate_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        factor=args.factor,
        scale=args.scale,
        max_frames=args.max_frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main() 