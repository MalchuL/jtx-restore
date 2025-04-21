#!/usr/bin/env python
"""Streamlit app for visualizing frame processor results.

This application allows users to upload a video, select frame processors to apply,
and visualize the intermediate results of each processor in the pipeline.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import rootutils
import streamlit as st
from PIL import Image

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.core.video.frames.readers.opencv_reader import OpenCVVideoReader
from src.core.video.frames.readers.video_reader import VideoMetadata, VideoReader

# Add parent directory to path to import project modules

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor import FrameProcessor
from src.core.video.frames.processors.visualizer.streamlit_processor_visualizer import StreamlitProcessorVisualizer

log = logging.getLogger(__name__)


@st.cache_resource(max_entries=3)  # Cache the reader for 3 videos, just in case
def _get_reader(video_path: str) -> VideoReader:
    return OpenCVVideoReader(video_path)

def get_video_metadata(video_path: str) -> VideoMetadata:
    reader = _get_reader(video_path)
    return reader.metadata


def set_frame_idx(video_path: str, frame_idx: int) -> None:
    reader = _get_reader(video_path)
    is_set = reader.set_frame_index(frame_idx)
    if not is_set:
        raise ValueError(f"Failed to set frame index to {frame_idx}")

def load_video_frame(video_path: str) -> np.ndarray:
    reader = _get_reader(video_path)
    frame = reader.read_frame()
    return frame

# Global variable to store the processor config, hydra main will return None
_processor_cfg = None

@hydra.main(config_path=str(root / "configs"), version_base="1.3", config_name="main.yaml")
def set_processors_config(cfg: DictConfig) -> Dict:
    global _processor_cfg
    _processor_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(_processor_cfg, type(_processor_cfg))
    
@st.cache_resource(max_entries=1)
def get_processors_config() -> Dict:
    if _processor_cfg is None:
        set_processors_config()
    return _processor_cfg



@st.cache_resource(max_entries=1)
def create_processor_instances(cfg: Dict) -> List[FrameProcessor]:
    """Create an instance of a processor with the given parameters.
    
    Args:
        processor_class: The processor class to instantiate
        **kwargs: Parameters to pass to the processor constructor
        
    Returns:
        An instance of the processor
    """
    # Filter kwargs to only include those accepted by the constructor
    processors = []
    for processor_name, processor_cfg in cfg["pipeline"]["processor"].items():
        log.info(
            f"Instantiating processor <{processor_name}> <{processor_cfg['_target_']}>"
        )
        processor = hydra.utils.instantiate(processor_cfg)
        processors.append(processor)
    return processors


def processor_parameter_ui(processor_name: str) -> Dict[str, Any]:
    """Generate UI for processor-specific parameters.
    
    Args:
        processor_name: Name of the processor
        
    Returns:
        Dictionary of parameter name to value
    """
    params = {}
    
    if processor_name == "Color Correction":
        st.subheader("Color Correction Parameters")
        params["brightness"] = st.slider("Brightness", -1.0, 1.0, 0.0, 0.05)
        params["contrast"] = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
        params["saturation"] = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
        params["gamma"] = st.slider("Gamma", 0.5, 2.0, 1.0, 0.05)
        params["white_balance"] = st.checkbox("Auto White Balance", False)
        params["auto_exposure"] = st.checkbox("Auto Exposure", False)
        params["num_workers"] = 1  # Use a single worker for Streamlit demo
        
    elif processor_name == "Denoise":
        st.subheader("Denoise Parameters")
        params["strength"] = st.slider("Strength", 1, 30, 10, 1)
        params["num_workers"] = 1  # Use a single worker for Streamlit demo
        
    elif processor_name == "Upscale":
        st.subheader("Upscale Parameters")
        params["scale_factor"] = st.slider("Scale Factor", 1.0, 4.0, 2.0, 0.5)
        params["interpolation"] = cv2.INTER_CUBIC
        params["num_workers"] = 1  # Use a single worker for Streamlit demo
        
    elif processor_name == "RIFE Frame Interpolation" and RIFE_AVAILABLE:
        st.subheader("RIFE Frame Interpolation Parameters")
        params["model_path"] = st.text_input("Model Path", "weights/practical_rife_4_25")
        params["scale"] = st.slider("Scale", 0.5, 1.0, 1.0, 0.1)
        
    return params


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Frame Processor Visualizer",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("Frame Processor Visualization")
    st.write("""
    This app allows you to visualize the effects of different frame processors on video frames.
    Upload a video, select processors to apply, and visualize the results.
    """)
    
    # Sidebar for processor selection
    st.sidebar.title("Processor Configuration")
    
    # Let user select processors to use
    # selected_processors = st.sidebar.multiselect(
    #     "Select Processors",
    #     list(available_processors.keys()),
    #     default=["Color Correction"],
    # )
    
    # # Create processor configuration UI based on selection
    # processor_params = {}
    # for processor_name in selected_processors:
    #     processor_params[processor_name] = processor_parameter_ui(processor_name)
    
    # Main area for video/frame display
    upload_col, preview_col = st.columns(2)
    
    
    video_path = None or "/media/ssd_stuff/video_enhance/out_video/01. Istoriya nachinaetsya_tmp.avi"
    with upload_col:
        # File uploader for video
        uploaded_file = st.text_input(
            "Path to video",
            value=video_path
        )
        
        # Example image selection if no video uploaded
        if uploaded_file:
            if not os.path.exists(uploaded_file):
                st.error(f"File {uploaded_file} does not exist")
                return
            video_path = uploaded_file
    
    with preview_col:
        if video_path:
            # Load the selected frame
            metadata = get_video_metadata(video_path)
            frame_idx = st.slider("Frame", 0, metadata.frame_count -1, 0)
            set_frame_idx(video_path, frame_idx)
            frame_data = load_video_frame(video_path)
            
            # Display frame metadata
            st.write(f"Frame dimensions: {metadata.width}x{metadata.height}")
            st.write(f"Video FPS: {metadata.fps:.2f}")
            st.write(f"Total frames: {metadata.frame_count}")
            
            # Create ProcessedFrame object
            original_frame = ProcessedFrame(
                data=frame_data,
                frame_id=frame_idx
            )
            
            # Display the original frame
            st.image(
                frame_data, 
                caption=f"Original Frame (Frame {frame_idx})",
                use_column_width=True
            )

    # Process button
    if st.button("Process Frame"):
        cfg = get_processors_config()
        selected_processors = create_processor_instances(cfg)
        if not selected_processors:
            st.warning("Please select at least one processor.")
            return
        
    #     # Create processor instances based on selections
    #     processor_instances = []
    #     for processor_name in selected_processors:
    #         processor_class = available_processors[processor_name]
    #         params = processor_params.get(processor_name, {})
    #         try:
    #             processor = create_processor_instances(processor_class, **params)
    #             processor_instances.append(processor)
    #         except Exception as e:
    #             st.error(f"Error creating {processor_name} processor: {str(e)}")
    #             return
        
        # Create visualizer and process the frame
        processor_instances = selected_processors
        visualizer = StreamlitProcessorVisualizer(processor_instances)
        visualizer.reset()
        
        try:
            # Process the frame and display results
            set_frame_idx(video_path, frame_idx)
            with st.spinner("Processing frame..."):
                frame = original_frame
                results = []
                _frame_idx = frame_idx
                while not results:
                    frame_data = load_video_frame(video_path)
                    
                    frame = ProcessedFrame(
                            data=frame_data,
                            frame_id=_frame_idx,
                        )
                    results = visualizer.process_frame(frame)
                    _frame_idx += 1
                st.success(f"Frame processed! Generated {len(results) if results else 0} output frames.")
                
                
                # Display intermediate results
                st.header("Processing Results")
                visualizer.visualize_in_streamlit(original_frame)
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # # Clean up temporary files on app restart
    # if 'video_path' in locals() and os.path.exists(video_path):
    #     try:
    #         if st.session_state.get('prev_video_path') != video_path:
    #             if 'prev_video_path' in st.session_state and os.path.exists(st.session_state.prev_video_path):
    #                 os.unlink(st.session_state.prev_video_path)
    #             st.session_state.prev_video_path = video_path
    #     except Exception:
    #         pass  # Ignore cleanup errors


if __name__ == "__main__":
    main() 