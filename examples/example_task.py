#!/usr/bin/env python
"""
Example script demonstrating the usage of OpenCVFFmpegPipeline with various frame processors.

This script shows how to:
1. Create a video processing pipeline
2. Add frame processors for denoising, upscaling, and color correction
3. Process a video file and save the result
"""

import argparse
import logging
from pathlib import Path
from typing import List

from src.pipelines.video.opencv_ffmpeg_pipeline import OpenCVFFmpegPipeline
from src.core.video.frames.processors import (
    ColorCorrectionProcessor,
    DenoiseProcessor,
    RealESRGANProcessor,
    FrameProcessor,
    PracticalRIFEFrameInterpolator425
)
from src.core.video.frames.processors import FBCNNProcessor
from src.tasks.video.enhance.video_enhance import VideoEnhanceTask

logging.basicConfig(level=logging.INFO)


def create_processor_pipeline() -> List[FrameProcessor]:
    """Create a pipeline of frame processors for video enhancement.

    Returns:
        List of frame processors to apply in sequence
    """
    return [
        # Denoise the video to remove noise and artifacts
        FBCNNProcessor(),
        PracticalRIFEFrameInterpolator425(),
        # Upscale the video using RealESRGAN
        RealESRGANProcessor(
            scale=2,  # 4x upscaling
            device="cuda",  # Use CUDA if available
            batch_size=32,  # RealESRGAN processes one frame at a time
        ),
        # Enhance colors and adjust visual parameters
        # ColorCorrectionProcessor(
        #     brightness=0.1,  # Slightly increase brightness
        #     contrast=1.1,  # Slightly increase contrast
        #     saturation=1.2,  # Increase color saturation
        #     white_balance=True,  # Apply automatic white balance
        #     gamma=1.1,  # Slightly adjust gamma
        #     auto_exposure=True,  # Apply automatic exposure correction
        #     num_workers=4,
        # ),
    ]


def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(
        description="Process a video file using the OpenCVFFmpegPipeline"
    )
    parser.add_argument("input_path", type=Path, help="Path to the input video file")
    parser.add_argument(
        "output_path", type=Path, help="Path where the processed video will be saved"
    )
    args = parser.parse_args()

    # Create the pipeline with our processors
    enhancer = VideoEnhanceTask(
        video_folder=args.input_path,
        output_folder=args.output_path,
        processors=create_processor_pipeline(),
    )

    # Process the video
    print(f"Processing video: {args.input_path}")
    enhancer.enhance()
    print(f"Processed video saved to: {args.output_path}")


if __name__ == "__main__":
    main()
