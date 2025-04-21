#!/usr/bin/env python
"""Streamlit app for visualizing frame processor results.

This application allows users to upload a video, select frame processors to apply,
and visualize the intermediate results of each processor in the pipeline.
"""

import logging
import os
import sys
import tempfile
import inspect
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Type, Union, get_type_hints, get_origin, get_args

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

# Import processor configs
from src.structured_configs.processors import (
    ColorCorrectionProcessorConfig,
    DenoiseProcessorConfig,
    UpscaleProcessorConfig,
    PracticalRIFEFrameInterpolatorConfig,
    FBCNNProcessorConfig,
    RealESRGANProcessorConfig,
)

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


def get_available_processor_configs() -> Dict[str, Type]:
    """Get available processor configurations.
    
    Returns:
        Dictionary mapping configuration names to their classes
    """
    return {
        "Color Correction": ColorCorrectionProcessorConfig,
        "Denoise": DenoiseProcessorConfig,
        "Upscale": UpscaleProcessorConfig,
        "RIFE Frame Interpolation": PracticalRIFEFrameInterpolatorConfig,
        "FBCNN JPEG Artifact Removal": FBCNNProcessorConfig,
        "RealESRGAN Upscaling": RealESRGANProcessorConfig,
    }


def generate_ui_for_config(config_class: Type, key_prefix: str) -> Dict[str, Any]:
    """Dynamically generate UI elements based on the dataclass fields.
    
    Args:
        config_class: The dataclass configuration class
        key_prefix: Prefix for Streamlit widget keys to ensure uniqueness
        
    Returns:
        Dictionary of parameter name to user-selected value
    """
    if not is_dataclass(config_class):
        st.error(f"{config_class.__name__} is not a dataclass")
        return {}
    
    config_params = {}
    
    # Get fields from the dataclass
    dataclass_fields = fields(config_class)
    type_hints = get_type_hints(config_class)
    
    # Skip _target_ field
    fields_to_display = [f for f in dataclass_fields if f.name != "_target_"]
    
    for field in fields_to_display:
        field_name = field.name
        field_type = type_hints.get(field_name)
        default_value = field.default
        
        # Skip fields with MISSING default
        if default_value == MISSING:
            continue
            
        field_help = f"{field_name}: {field_type}"
        
        # Check if the field is Optional (Union[X, None])
        is_optional = False
        inner_type = field_type
        
        if get_origin(field_type) is Union:
            args = get_args(field_type)
            if type(None) in args:
                is_optional = True
                # Find the non-None type
                inner_type = next(arg for arg in args if arg is not type(None))
        
        # Create a unique key for each widget
        widget_key = f"{key_prefix}_{field_name}"
        
        # Add checkbox for optional fields
        use_field = True
        if is_optional:
            use_field = st.checkbox(f"Use {field_name}", 
                                    value=default_value is not None,
                                    key=f"{widget_key}_use",
                                    help=f"Enable/disable {field_name}")
        
        # Skip if optional field is disabled
        if not use_field:
            config_params[field_name] = None
            continue
            
        # Generate UI based on field type
        if inner_type == int:
            # Determine reasonable min/max values based on field name
            min_val, max_val, step = 1, 100, 1
            if "factor" in field_name.lower():
                min_val, max_val = 1, 8
            elif "size" in field_name.lower() and "batch" not in field_name.lower():
                min_val, max_val = 3, 51
                # Ensure odd values for window sizes
                if "window" in field_name.lower():
                    step = 2
                    min_val = 3
                    if default_value > min_val:
                        min_val = default_value - 10
                    max_val = default_value + 20
            elif "batch" in field_name.lower():
                min_val, max_val = 1, 32
            elif "scale" in field_name.lower():
                min_val, max_val = 1, 8
                
            config_params[field_name] = st.slider(
                field_name, 
                min_value=min_val,
                max_value=max_val,
                value=default_value,
                step=step,
                key=widget_key,
                help=field_help
            )
        
        elif inner_type == float:
            # Determine reasonable min/max values based on field name
            min_val, max_val, step = 0.0, 1.0, 0.01
            if "strength" in field_name.lower():
                min_val, max_val = 0.0, 20.0
                step = 0.5
            elif "brightness" in field_name.lower():
                min_val, max_val = -1.0, 1.0
            elif "contrast" in field_name.lower() or "saturation" in field_name.lower():
                min_val, max_val = 0.0, 3.0
            elif "scale" in field_name.lower():
                min_val, max_val = 0.1, 8.0
                step = 0.1
            elif "gamma" in field_name.lower():
                min_val, max_val = 0.1, 3.0
                step = 0.1
            elif "compression" in field_name.lower():
                min_val, max_val = 0.0, 100.0
                step = 1.0
                
            config_params[field_name] = st.slider(
                field_name, 
                min_value=min_val,
                max_value=max_val,
                value=default_value if default_value is not None else (min_val + max_val) / 2,
                step=step,
                key=widget_key,
                help=field_help
            )
        
        elif inner_type == bool:
            config_params[field_name] = st.checkbox(
                field_name, 
                value=default_value,
                key=widget_key,
                help=field_help
            )
        
        elif inner_type == str:
            # Handle enum-like string fields
            if field_name == "interpolation" and config_class == UpscaleProcessorConfig:
                options = ["nearest", "bilinear", "bicubic", "lanczos"]
                config_params[field_name] = st.selectbox(
                    field_name,
                    options=options,
                    index=options.index(default_value) if default_value in options else 0,
                    key=widget_key,
                    help=field_help
                )
            elif field_name == "device":
                options = ["cuda", "cpu"]
                config_params[field_name] = st.selectbox(
                    field_name,
                    options=options,
                    index=0 if default_value == "cuda" or default_value is None else 1,
                    key=widget_key,
                    help=field_help
                )
            else:
                config_params[field_name] = st.text_input(
                    field_name, 
                    value=default_value if default_value is not None else "",
                    key=widget_key,
                    help=field_help
                )
                # Convert empty string to None for optional fields
                if config_params[field_name] == "" and is_optional:
                    config_params[field_name] = None
        
        else:
            # For complex types, just use a text input with JSON representation
            if default_value is not None:
                default_str = str(default_value)
            else:
                default_str = ""
            
            input_value = st.text_input(
                field_name, 
                value=default_str,
                key=widget_key,
                help=f"{field_help} (advanced)"
            )
            
            # For now, just pass the string representation
            config_params[field_name] = input_value if input_value else None
            
    return config_params


@st.cache_resource(max_entries=1)
def create_processor_instances(cfg: Dict) -> List[FrameProcessor]:
    """Create processor instances from configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        List of processor instances
    """
    processors = []
    for processor_name, processor_cfg in cfg["pipeline"]["processor"].items():
        log.info(
            f"Instantiating processor <{processor_name}> <{processor_cfg['_target_']}>"
        )
        processor = hydra.utils.instantiate(processor_cfg)
        processors.append(processor)
    return processors


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
    
    # Get available processor configs
    available_configs = get_available_processor_configs()
    
    # Let user select processors to use
    selected_processors = st.sidebar.multiselect(
        "Select Processors",
        list(available_configs.keys()),
        default=["FBCNN JPEG Artifact Removal", "Color Correction"]
    )
    
    # Create processor configuration UI based on selection
    selected_configs = {}
    for idx, processor_name in enumerate(selected_processors):
        config_class = available_configs[processor_name]
        
        with st.sidebar.expander(f"{processor_name} Settings", expanded=False):
            st.subheader(f"{processor_name} Parameters")
            config_params = generate_ui_for_config(config_class, f"proc_{idx}")
            
            # Create instance of the config with parameters
            config_instance = config_class(**config_params)
            selected_configs[processor_name] = config_instance
    
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
    if st.button("Process Frame") and selected_configs:
        if not selected_configs:
            st.warning("Please select at least one processor.")
            return
        
        # Create processor instances based on configurations
        processor_instances = []
        for processor_name, config in selected_configs.items():
            try:
                processor = hydra.utils.instantiate(config)
                processor_instances.append(processor)
                st.success(f"Created {processor_name} processor")
            except Exception as e:
                st.error(f"Error creating {processor_name} processor: {str(e)}")
                return
        
        # Create visualizer and process the frame
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


if __name__ == "__main__":
    main() 