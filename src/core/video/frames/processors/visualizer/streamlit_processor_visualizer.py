"""Streamlit-based visualizer for frame processors.

This module provides a visualization tool that extends FrameProcessorManager
to track and visualize the outputs of each processor in a Streamlit app.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import logging
import numpy as np
import streamlit as st

from src.core.video.frames.processors.frame import ProcessedFrame
from src.core.video.frames.processors.processor import FrameProcessor
from src.core.video.frames.processors.frame_processor_manager import FrameProcessorManager


class StreamlitProcessorVisualizer(FrameProcessorManager):
    """Visualizes frame processor results in Streamlit.
    
    This class extends the FrameProcessorManager to capture and display
    intermediate results from each processor in the pipeline using Streamlit.
    
    Attributes:
        processors: The processor pipeline to apply to frames
        capture_intermediates: Whether to capture intermediate results
        intermediate_results: Dictionary mapping processor indices to their outputs
    """
    
    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        capture_intermediates: bool = True,
    ):
        """Initialize the Streamlit processor visualizer.

        Args:
            processors: Sequence of frame processors to apply
            capture_intermediates: Whether to capture intermediate results
        """
        super().__init__(processors)
        self.capture_intermediates = capture_intermediates
        self.intermediate_results: Dict[int, List[ProcessedFrame]] = {}
        self.logger = logging.getLogger(__name__)
        
    def clear_intermediate_results(self) -> None:
        """Clear the intermediate results cache."""
        self.intermediate_results = {}
        
    def reset(self) -> None:
        """Reset the visualizer."""
        super().reset()
        self.clear_intermediate_results()
        
    def _run_frames(
        self, frames: List[ProcessedFrame], processor_id: int, do_finish: bool = False
    ) -> List[ProcessedFrame]:
        """Override _run_frames to capture intermediate results.
        
        Args:
            frames: List of frames to process
            processor_id: Index of the processor to run
            do_finish: Whether to finish the processor
            
        Returns:
            List of processed frames
        """
        # If all processors have been run, return the frames
        if processor_id >= len(self.processors):
            return frames

        processor = self.processors[processor_id]
        results = []
        
        # Run the processor on each frame
        for frame in frames:
            result = processor(frame)
            if result.ready:
                # Store the intermediate results for this processor
                if self.capture_intermediates:
                    if processor_id not in self.intermediate_results:
                        self.intermediate_results[processor_id] = []
                    self.intermediate_results[processor_id].extend(result.frames)
                
                # Continue processing with the next processor
                processor_results = self._run_frames(
                    result.frames, processor_id + 1, False
                )
                results.extend(processor_results)
                
        # Handle finishing if needed
        if do_finish:
            last_result = processor.finish()
            if last_result.ready:
                # Store the intermediate results from finishing
                if self.capture_intermediates and last_result.frames:
                    if processor_id not in self.intermediate_results:
                        self.intermediate_results[processor_id] = []
                    self.intermediate_results[processor_id].extend(last_result.frames)
                
                remaining_frames = last_result.frames
            else:
                remaining_frames = []
                
            # Process remaining frames with next processor
            remaining_results = self._run_frames(
                remaining_frames, processor_id + 1, True
            )
            results.extend(remaining_results)
            
        return results
    
    def get_intermediate_results(self, processor_id: int) -> List[ProcessedFrame]:
        """Get the intermediate results for a specific processor.
        
        Args:
            processor_id: Index of the processor
            
        Returns:
            List of processed frames for the specified processor
        """
        return self.intermediate_results.get(processor_id, [])
    
    def visualize_in_streamlit(self, original_frame: Optional[ProcessedFrame] = None) -> None:
        """Display the original frame and all intermediate results in Streamlit.
        
        Args:
            original_frame: The original input frame for comparison
        """
        if not self.intermediate_results:
            st.warning("No intermediate results available. Process a frame first.")
            return
            
        # Display original frame if provided
        if original_frame is not None:
            st.subheader("Original Frame")
            st.image(original_frame.data, channels="RGB", caption=f"Frame ID: {original_frame.frame_id}")
            
            # Show metadata if available
            if original_frame.metadata:
                with st.expander("Original Frame Metadata"):
                    st.json(original_frame.metadata)
        
        # Display results for each processor
        for processor_id, frames in sorted(self.intermediate_results.items()):
            processor = self.processors[processor_id]
            processor_name = processor.__class__.__name__
            
            st.subheader(f"Processor {processor_id + 1}: {processor_name}")
            
            # Handle multiple output frames (e.g., from interpolation)
            for i, frame in enumerate(frames):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(frame.data, channels="RGB", 
                             caption=f"Output {i+1} (Frame ID: {frame.frame_id})")
                
                with col2:
                    # Show frame differences if original frame is available
                    if original_frame is not None and frame.shape == original_frame.shape:
                        # Calculate absolute difference between frames
                        diff = np.abs(frame.data.astype(np.float32) - 
                                     original_frame.data.astype(np.float32))
                        # Normalize for visualization
                        diff = np.clip(diff * 5, 0, 255).astype(np.uint8)
                        st.image(diff, channels="RGB", caption="Difference (5x magnified)")
                
                # Show metadata if available
                if frame.metadata:
                    with st.expander(f"Frame {i+1} Metadata"):
                        st.text(frame.metadata) 