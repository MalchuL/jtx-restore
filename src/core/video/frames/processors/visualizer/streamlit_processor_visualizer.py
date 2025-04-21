"""Streamlit-based visualizer for frame processors.

This module provides a visualization tool that extends FrameProcessorManager
to track and visualize the outputs of each processor in a Streamlit app.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import logging
import time
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
        processing_times: Dictionary mapping processor indices to their execution times
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
        self.processing_times: Dict[int, float] = {}  # Store time spent per processor
        self.logger = logging.getLogger(__name__)
        
    def clear_intermediate_results(self) -> None:
        """Clear the intermediate results cache."""
        self.intermediate_results = {}
        self.processing_times = {}
        
    def reset(self) -> None:
        """Reset the visualizer."""
        super().reset()
        self.clear_intermediate_results()
        
    def _run_frames(
        self, frames: List[ProcessedFrame], processor_id: int, do_finish: bool = False
    ) -> List[ProcessedFrame]:
        """Override _run_frames to capture intermediate results and timing.
        
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
        
        # Initialize timing for this processor if not already done
        if processor_id not in self.processing_times:
            self.processing_times[processor_id] = 0.0
        
        # Run the processor on each frame
        for frame in frames:
            # Measure processor execution time
            start_time = time.time()
            result = processor(frame)
            end_time = time.time()
            
            # Add execution time to the processor's total
            self.processing_times[processor_id] += (end_time - start_time)
            
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
            # Measure finish execution time
            start_time = time.time()
            last_result = processor.finish()
            end_time = time.time()
            
            # Add finish execution time to the processor's total
            self.processing_times[processor_id] += (end_time - start_time)
            
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
    
    def get_processing_time(self, processor_id: int) -> float:
        """Get the time spent on a specific processor.
        
        Args:
            processor_id: Index of the processor
            
        Returns:
            Time spent in seconds on the processor
        """
        return self.processing_times.get(processor_id, 0.0)
    
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
        
        # Display timing summary
        st.subheader("Processing Times")
        total_time = sum(self.processing_times.values())
        
        # Create timing metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Processing Time", f"{total_time:.4f} sec")
        
        # Display results for each processor
        for processor_id, frames in sorted(self.intermediate_results.items()):
            processor = self.processors[processor_id]
            processor_name = processor.__class__.__name__
            processor_time = self.processing_times.get(processor_id, 0.0)
            time_percentage = (processor_time / total_time * 100) if total_time > 0 else 0
            
            # Create a header with processor name and time information
            st.subheader(f"Processor {processor_id + 1}: {processor_name}")
            st.info(f"⏱️ Time: {processor_time:.4f} sec ({time_percentage:.1f}% of total)")
            
            # Create a progress bar for visual representation of time percentage
            st.progress(min(time_percentage / 100, 1.0))
            
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