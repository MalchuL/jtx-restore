# #!/usr/bin/env python
# """
# Tests for the color correction frame processor.
# """

# import numpy as np
# import pytest
# import cv2

# from src.core.video.frames.processors.enhancers.color.color import ColorCorrectionProcessor
# from src.core.video.frames.processors.frame import ProcessedFrame

# raise NotImplementedError("Color correction processor is not implemented yet.")

# @pytest.fixture
# def sample_frame():
#     """Create a sample frame for testing."""
#     # Create a simple test pattern with different colors
#     frame = np.zeros((100, 100, 3), dtype=np.uint8)
#     # White square
#     frame[25:50, 25:50] = [255, 255, 255]
#     # Red square
#     frame[25:50, 50:75] = [255, 0, 0]
#     # Green square
#     frame[50:75, 25:50] = [0, 255, 0]
#     # Blue square
#     frame[50:75, 50:75] = [0, 0, 255]
#     return frame


# @pytest.fixture
# def sample_frames(sample_frame):
#     """Create a batch of sample frames for testing."""
#     return [sample_frame.copy() for _ in range(5)]


# @pytest.fixture
# def processor():
#     """Create a default color correction processor."""
#     return ColorCorrectionProcessor()


# def test_color_processor_initialization():
#     """Test proper initialization of the color correction processor."""
#     # Test valid initialization
#     processor = ColorCorrectionProcessor()
#     assert processor.brightness == 0.0
#     assert processor.contrast == 1.0
#     assert processor.saturation == 1.0
#     assert processor.white_balance is False
#     assert processor.gamma == 1.0
#     assert processor.auto_exposure is False

#     # Test parameter clamping
#     processor = ColorCorrectionProcessor(brightness=2.0)  # Should be clamped to 1.0
#     assert processor.brightness == 1.0

#     processor = ColorCorrectionProcessor(brightness=-2.0)  # Should be clamped to -1.0
#     assert processor.brightness == -1.0

#     processor = ColorCorrectionProcessor(contrast=4.0)  # Should be clamped to 3.0
#     assert processor.contrast == 3.0

#     processor = ColorCorrectionProcessor(gamma=4.0)  # Should be clamped to 3.0
#     assert processor.gamma == 3.0


# def test_color_processor_output_shape(processor, sample_frame):
#     """Test that the output frame has the correct shape."""
#     input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
#     output_frame = processor.process_frame(input_frame)
    
#     assert output_frame.data.shape == sample_frame.shape
#     assert output_frame.frame_id == 0


# def test_color_processor_brightness_adjustment(sample_frame):
#     """Test brightness adjustment."""
#     # Test increasing brightness
#     processor = ColorCorrectionProcessor(brightness=0.5)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     assert np.mean(output_frame.data) > np.mean(sample_frame)

#     # Test decreasing brightness
#     processor = ColorCorrectionProcessor(brightness=-0.5)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     assert np.mean(output_frame.data) < np.mean(sample_frame)


# def test_color_processor_contrast_adjustment(sample_frame):
#     """Test contrast adjustment."""
#     # Test increasing contrast
#     processor = ColorCorrectionProcessor(contrast=2.0)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     assert np.std(output_frame.data) > np.std(sample_frame)

#     # Test decreasing contrast
#     processor = ColorCorrectionProcessor(contrast=0.5)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     assert np.std(output_frame.data) < np.std(sample_frame)


# def test_color_processor_saturation_adjustment(sample_frame):
#     """Test saturation adjustment."""
#     # Test increasing saturation
#     processor = ColorCorrectionProcessor(saturation=2.0)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     # Convert to HSV to check saturation
#     hsv_output = cv2.cvtColor(output_frame.data, cv2.COLOR_RGB2HSV)
#     hsv_input = cv2.cvtColor(sample_frame, cv2.COLOR_RGB2HSV)
#     assert np.mean(hsv_output[:, :, 1]) > np.mean(hsv_input[:, :, 1])

#     # Test decreasing saturation
#     processor = ColorCorrectionProcessor(saturation=0.5)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     hsv_output = cv2.cvtColor(output_frame.data, cv2.COLOR_RGB2HSV)
#     assert np.mean(hsv_output[:, :, 1]) < np.mean(hsv_input[:, :, 1])


# def test_color_processor_gamma_correction(sample_frame):
#     """Test gamma correction."""
#     # Test gamma > 1 (darkens midtones)
#     processor = ColorCorrectionProcessor(gamma=2.0)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     # Check that midtones are darker
#     midtones_input = np.mean(sample_frame[sample_frame > 127])
#     midtones_output = np.mean(output_frame.data[output_frame.data > 127])
#     assert midtones_output < midtones_input

#     # Test gamma < 1 (brightens midtones)
#     processor = ColorCorrectionProcessor(gamma=0.5)
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
#     midtones_output = np.mean(output_frame.data[output_frame.data > 127])
#     assert midtones_output > midtones_input


# def test_color_processor_white_balance(sample_frame):
#     """Test white balance correction."""
#     # Create a frame with color cast
#     color_cast = np.ones_like(sample_frame) * 50
#     frame_with_cast = cv2.add(sample_frame, color_cast)
    
#     processor = ColorCorrectionProcessor(white_balance=True)
#     output_frame = processor.process_frame(ProcessedFrame(data=frame_with_cast, frame_id=0))
    
#     # Check that the color cast is reduced
#     assert np.mean(output_frame.data) < np.mean(frame_with_cast)


# def test_color_processor_auto_exposure(sample_frame):
#     """Test auto exposure correction."""
#     # Create an underexposed frame
#     underexposed = cv2.multiply(sample_frame, 0.5)
    
#     processor = ColorCorrectionProcessor(auto_exposure=True)
#     output_frame = processor.process_frame(ProcessedFrame(data=underexposed, frame_id=0))
    
#     # Check that the exposure is improved
#     assert np.mean(output_frame.data) > np.mean(underexposed)


# def test_color_processor_with_none_frame(processor):
#     """Test handling of None input frame."""
#     output_frame = processor.process_frame(None)
#     assert output_frame is None


# def test_color_processor_combined_effects(sample_frame):
#     """Test combined effects of multiple adjustments."""
#     processor = ColorCorrectionProcessor(
#         brightness=0.2,
#         contrast=1.5,
#         saturation=1.2,
#         gamma=1.1
#     )
#     output_frame = processor.process_frame(ProcessedFrame(data=sample_frame, frame_id=0))
    
#     # Check that the output is different from input
#     assert not np.array_equal(output_frame.data, sample_frame)
    
#     # Check that the output is valid
#     assert np.all(output_frame.data >= 0)
#     assert np.all(output_frame.data <= 255)


# def test_color_processor_preserves_frame_id(sample_frame):
#     """Test that the frame ID is preserved during processing."""
#     frame_id = 42
#     input_frame = ProcessedFrame(data=sample_frame, frame_id=frame_id)
#     processor = ColorCorrectionProcessor()
#     output_frame = processor.process_frame(input_frame)
    
#     assert output_frame.frame_id == frame_id


# def test_color_processor_parallel_processing(sample_frames):
#     """Test parallel processing of frames."""
#     # Create processor with 2 workers
#     processor = ColorCorrectionProcessor(
#         brightness=0.2,
#         contrast=1.5,
#         saturation=1.2,
#         num_workers=2
#     )
    
#     # Create input frames with ProcessorFrame wrapper
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(sample_frames)]
    
#     # Process frames in parallel
#     output_frames = processor.process_batch(input_frames)
    
#     # Check results
#     assert len(output_frames) == len(input_frames)
#     for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
#         # Check frame ID preservation
#         assert output_frame.frame_id == i
        
#         # Check output shape
#         assert output_frame.data.shape == input_frame.data.shape
        
#         # Check that color correction was applied
#         assert not np.array_equal(output_frame.data, input_frame.data)
#         assert np.all(output_frame.data >= 0)
#         assert np.all(output_frame.data <= 255)


# def test_color_processor_sequential_vs_parallel(sample_frames):
#     """Test that parallel processing produces same results as sequential."""
#     # Create processor with 2 workers
#     parallel_processor = ColorCorrectionProcessor(
#         brightness=0.2,
#         contrast=1.5,
#         saturation=1.2,
#         num_workers=2
#     )
#     sequential_processor = ColorCorrectionProcessor(
#         brightness=0.2,
#         contrast=1.5,
#         saturation=1.2,
#         num_workers=0
#     )
    
#     # Create input frames
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(sample_frames)]
    
#     # Process frames both ways
#     parallel_output = parallel_processor.process_batch(input_frames)
#     sequential_output = sequential_processor.process_batch(input_frames)
    
#     # Compare results
#     assert len(parallel_output) == len(sequential_output)
#     for p_out, s_out in zip(parallel_output, sequential_output):
#         # Results should be identical
#         assert np.array_equal(p_out.data, s_out.data)
#         assert p_out.frame_id == s_out.frame_id


# def test_color_processor_empty_batch(processor):
#     """Test processing of empty batch."""
#     output_frames = processor.process_batch([])
#     assert len(output_frames) == 0


# def test_color_processor_single_frame_batch(sample_frame):
#     """Test processing of single frame batch."""
#     processor = ColorCorrectionProcessor(
#         brightness=0.2,
#         contrast=1.5,
#         saturation=1.2,
#         num_workers=0
#     )
#     input_frame = ProcessedFrame(data=sample_frame, frame_id=0)
#     output_frames = processor.process_batch([input_frame])
    
#     assert len(output_frames) == 1
#     assert output_frames[0].frame_id == 0
    
#     # Check that color correction was applied
#     assert not np.array_equal(output_frames[0].data, sample_frame)
#     assert np.all(output_frames[0].data >= 0)
#     assert np.all(output_frames[0].data <= 255)


# def test_color_processor_large_batch(sample_frame):
#     """Test processing of large batch."""
#     # Create a large batch of frames
#     large_batch = [sample_frame.copy() for _ in range(20)]
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(large_batch)]
    
#     # Process with different worker counts
#     for num_workers in [0, 2, 4]:
#         processor = ColorCorrectionProcessor(
#             brightness=0.2,
#             contrast=1.5,
#             saturation=1.2,
#             num_workers=num_workers
#         )
#         output_frames = processor.process_batch(input_frames)
        
#         assert len(output_frames) == len(input_frames)
#         for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
#             assert output_frame.frame_id == i
#             assert output_frame.data.shape == input_frame.data.shape
            
#             # Check that color correction was applied
#             assert not np.array_equal(output_frame.data, input_frame.data)
#             assert np.all(output_frame.data >= 0)
#             assert np.all(output_frame.data <= 255) 