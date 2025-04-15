#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path

from src.merger import FFMPEGVideoAudioMerger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("merger_example")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Merge video and audio files using FFMPEG")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--output", required=True, help="Path for the output file")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to the ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to the ffprobe executable")
    args = parser.parse_args()
    
    logger.info(f"Starting video-audio merge process")
    logger.info(f"Video: {args.video}")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Create the merger with all source and destination paths
        merger = FFMPEGVideoAudioMerger(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output,
            ffmpeg_path=args.ffmpeg,
            ffprobe_path=args.ffprobe,
            extra_args=["-shortest"]  # Use the shortest of the two inputs
        )
        
        # The merger already validated inputs and FFMPEG capability in __init__
        logger.info("FFMPEG configured successfully")
        
        # Perform the merge operation
        result_path = merger.merge()
        logger.info(f"Merge completed successfully to: {result_path}")
        
        # Get and display information about the merged file
        media_info = merger.probe_media_info(result_path)
        
        # Extract and display format information
        format_info = media_info.get('format', {})
        duration = format_info.get('duration', '0')
        size = int(format_info.get('size', 0))
        
        logger.info(f"Output file details:")
        logger.info(f"  Duration: {float(duration):.2f} seconds")
        logger.info(f"  Size: {size / (1024 * 1024):.2f} MB")
        
        # Display stream information
        for idx, stream in enumerate(media_info.get('streams', [])):
            codec_type = stream.get('codec_type', 'unknown')
            codec_name = stream.get('codec_name', 'unknown')
            
            if codec_type == 'video':
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                frame_rate = stream.get('avg_frame_rate', '0/0')
                
                try:
                    fps = eval(frame_rate)
                except:
                    fps = 0
                
                logger.info(f"  Video Stream: {codec_name} - {width}x{height} @ {fps} FPS")
            
            elif codec_type == 'audio':
                channels = stream.get('channels', 0)
                sample_rate = stream.get('sample_rate', 0)
                bit_rate = stream.get('bit_rate', 0)
                
                logger.info(f"  Audio Stream: {codec_name} - {channels} channels, {sample_rate} Hz")
                if bit_rate:
                    logger.info(f"    Bitrate: {int(bit_rate) / 1000} kbps")
        
        logger.info("Merge operation completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 2
    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 4

if __name__ == "__main__":
    sys.exit(main())