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

from src.core.video.pipeline.opencv_ffmpeg_pipeline import OpenCVFFmpegPipeline
from src.core.video.processors.enhancers import (
    ColorCorrectionProcessor,
    DenoiseProcessor,
)
from src.core.video.processors.ai.upscale.realesrgan import RealESRGANProcessor
from src.core.video.processors.processor import FrameProcessor

logging.basicConfig(level=logging.INFO)


def create_processor_pipeline() -> List[FrameProcessor]:
    """Create a pipeline of frame processors for video enhancement.

    Returns:
        List of frame processors to apply in sequence
    """
    return [
        # Denoise the video to remove noise and artifacts
        # DenoiseProcessor(
        #     strength=10.0,
        #     color_strength=10.0,
        #     template_window_size=7,
        #     search_window_size=21,
        #     use_fast_nl_means=True,
        #     num_workers=4,  # Use 4 worker processes for parallel processing
        # ),
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
    pipeline = OpenCVFFmpegPipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        processors=create_processor_pipeline(),
        upscale_coefficient=2.0,  # Match RealESRGAN's 4x upscaling
    )

    # Process the video
    print(f"Processing video: {args.input_path}")
    pipeline.run()
    print(f"Processed video saved to: {args.output_path}")


if __name__ == "__main__":
    main()
