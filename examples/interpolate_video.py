#!/usr/bin/env python
"""
Example script demonstrating frame interpolation using OpenCVFrameInterpolator.

This script:
1. Reads a video file using OpenCVVideoReader
2. Applies frame interpolation to increase the frame rate
3. Writes the interpolated video to a new file using OpenCVVideoWriter
"""

import argparse
import logging
import time
from pathlib import Path
import numpy as np
import cv2

from src.core.video.readers.opencv_reader import OpenCVVideoReader
from src.core.video.writers.opencv_writer import OpenCVVideoWriter
from src.core.video.frame_interpolation.interpolators import OpenCVFrameInterpolator
from src.core.video.frame_interpolation.interpolated_frame import InterpolatedFrame

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_video(
    input_path: Path,
    output_path: Path,
    factor: float = 2.0,
    optical_flow_method: str = 'farneback',
    show_preview: bool = False
):
    """
    Process a video by applying frame interpolation.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video
        factor: Frame rate multiplier (e.g., 2.0 doubles the frame rate)
        optical_flow_method: Method to use for optical flow ('farneback' or 'dis')
        show_preview: Whether to show a preview window during processing
    """
    # Open the input video
    logger.info(f"Opening input video: {input_path}")
    reader = OpenCVVideoReader(input_path)
    
    # Get video metadata
    metadata = reader.metadata
    logger.info(f"Input video: {metadata.width}x{metadata.height}, {metadata.fps} fps, {metadata.frame_count} frames")
    
    # Calculate new FPS for the output video
    new_fps = metadata.fps * factor
    logger.info(f"Output video will have {new_fps} fps (factor: {factor}x)")
    
    # Create the writer
    logger.info(f"Creating output video: {output_path}")
    writer = OpenCVVideoWriter(
        output_path=output_path,
        fps=new_fps,
        frame_size=(metadata.width, metadata.height),
        codec=None,  # Use default codec based on file extension
        resize_frames=False
    )
    
    # Create the frame interpolator
    logger.info(f"Creating frame interpolator with method: {optical_flow_method}")
    interpolator = OpenCVFrameInterpolator(factor=factor, optical_flow_method=optical_flow_method)
    
    # Process the video
    start_time = time.time()
    frame_count = 0
    output_frame_count = 0
    
    # Open the reader
    reader.open()
    writer.open()
    
    # Create preview window if requested
    if show_preview:
        cv2.namedWindow("Interpolation Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Interpolation Preview", 960, 540)
    
    logger.info("Starting frame processing...")
    while not reader.is_finished:
        # Read the next frame
        frame = reader.read_frame()
        
        if frame is not None:
            # Convert to InterpolatedFrame
            interp_frame = InterpolatedFrame(
                data=frame,
                frame_id=frame_count,
                dt=0
            )
            
            # Process the frame through the interpolator
            result_frames = interpolator.process_frame(interp_frame)
            
            # Write frames if we have any
            if result_frames:
                for result_frame in result_frames:
                    print(f"Writing frame: {result_frame.frame_id}, {result_frame.dt}")
                    # Write the frame
                    writer.write_frame(result_frame.data)
                    output_frame_count += 1
                    
                    # Show preview if requested
                    if show_preview:
                        # Add info text to the preview
                        preview = result_frame.data.copy()
                        if result_frame.metadata.get('interpolated', False):
                            text = f"Interpolated (dt={result_frame.dt:.2f})"
                            color = (255, 120, 0)  # Orange for interpolated frames
                        else:
                            text = "Original"
                            color = (0, 255, 0)  # Green for original frames
                            
                        cv2.putText(
                            preview, 
                            text, 
                            (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            color, 
                            2
                        )
                        
                        # Convert RGB to BGR for OpenCV display
                        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Interpolation Preview", preview_bgr)
                        
                        # Process UI events and check for exit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            logger.info("Preview closed by user")
                            break
            
            frame_count += 1
            
            # Log progress periodically
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count}/{metadata.frame_count} frames ({fps:.2f} fps)")
    
    # Process remaining frames in the interpolator
    result_frames = interpolator.process_frame(None)
    if result_frames:
        for result_frame in result_frames:
            writer.write_frame(result_frame.data)
            output_frame_count += 1
    
    # Clean up
    reader.close()
    writer.close()
    if show_preview:
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"Processing complete. Elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Input frames: {frame_count}, Output frames: {output_frame_count}")
    logger.info(f"Output video saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Increase video frame rate using frame interpolation")
    parser.add_argument("input", type=Path, help="Path to input video file")
    parser.add_argument("output", type=Path, help="Path to output video file")
    parser.add_argument("--factor", type=float, default=2.0, help="Frame rate multiplier (default: 2.0)")
    parser.add_argument("--method", type=str, default="farneback", choices=["farneback", "dis"], 
                       help="Optical flow method (default: farneback)")
    parser.add_argument("--preview", action="store_true", help="Show preview window during processing")
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        input_path=args.input,
        output_path=args.output,
        factor=args.factor,
        optical_flow_method=args.method,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main() 