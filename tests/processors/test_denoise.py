# #!/usr/bin/env python
# """
# Tests for the denoising frame processor.
# """

# import numpy as np
# import pytest
# import cv2

# from src.core.video.frames.processors.enhancers.denoise.denoising import DenoiseProcessor
# from src.core.video.frames.processors.frame import ProcessedFrame

# raise NotImplementedError("Denoising processor is not implemented yet.")

# @pytest.fixture
# def sample_frame():
#     """Create a sample frame for testing."""
#     # Create a simple test pattern
#     frame = np.zeros((100, 100, 3), dtype=np.uint8)
#     frame[25:75, 25:75] = 255  # White square in the middle
#     return frame


# @pytest.fixture
# def noisy_frame(sample_frame):
#     """Create a noisy version of the sample frame."""
#     noise = np.random.normal(0, 25, sample_frame.shape).astype(np.uint8)
#     noisy_frame = cv2.add(sample_frame, noise)
#     return np.clip(noisy_frame, 0, 255)


# @pytest.fixture
# def noisy_frames(noisy_frame):
#     """Create a batch of noisy frames for testing."""
#     return [noisy_frame.copy() for _ in range(5)]


# @pytest.fixture
# def processor():
#     """Create a default denoise processor."""
#     return DenoiseProcessor(strength=10.0)


# def test_denoise_processor_initialization():
#     """Test proper initialization of the denoise processor."""
#     # Test valid initialization
#     processor = DenoiseProcessor(strength=10.0)
#     assert processor.strength == 10.0
#     assert processor.color_strength == 10.0
#     assert processor.template_window_size % 2 == 1
#     assert processor.search_window_size % 2 == 1
#     assert processor.use_fast_nl_means is True

#     # Test parameter clamping
#     processor = DenoiseProcessor(strength=25.0)  # Should be clamped to 20.0
#     assert processor.strength == 20.0

#     processor = DenoiseProcessor(strength=-5.0)  # Should be clamped to 0.0
#     assert processor.strength == 0.0


# def test_denoise_processor_output_shape(processor, noisy_frame):
#     """Test that the output frame has the correct shape."""
#     input_frame = ProcessedFrame(data=noisy_frame, frame_id=0)
#     output_frame = processor.process_frame(input_frame)
    
#     assert output_frame.data.shape == noisy_frame.shape
#     assert output_frame.frame_id == 0


# def test_denoise_processor_reduces_noise(processor, noisy_frame):
#     """Test that the denoising actually reduces noise."""
#     input_frame = ProcessedFrame(data=noisy_frame, frame_id=0)
#     output_frame = processor.process_frame(input_frame)
    
#     # Calculate noise levels (standard deviation)
#     input_noise = np.std(noisy_frame)
#     output_noise = np.std(output_frame.data)
    
#     # The output should have less noise than the input
#     assert output_noise < input_noise


# def test_denoise_processor_with_none_frame(processor):
#     """Test handling of None input frame."""
#     output_frame = processor.process_frame(None)
#     assert output_frame is None


# def test_denoise_processor_with_different_strengths(noisy_frame):
#     """Test denoising with different strength values."""
#     strengths = [5.0, 10.0, 15.0, 20.0]
#     noise_levels = []
    
#     for strength in strengths:
#         processor = DenoiseProcessor(strength=strength)
#         input_frame = ProcessedFrame(data=noisy_frame, frame_id=0)
#         output_frame = processor.process_frame(input_frame)
        
#         # Calculate noise level
#         noise_level = np.std(output_frame.data)
#         noise_levels.append(noise_level)
    
#     # Higher strength should result in lower noise
#     assert noise_levels[0] > noise_levels[-1]


# def test_denoise_processor_with_different_window_sizes(noisy_frame):
#     """Test denoising with different window sizes."""
#     window_sizes = [(5, 15), (7, 21), (9, 27)]
    
#     for template_size, search_size in window_sizes:
#         processor = DenoiseProcessor(
#             template_window_size=template_size,
#             search_window_size=search_size
#         )
#         input_frame = ProcessedFrame(data=noisy_frame, frame_id=0)
#         output_frame = processor.process_frame(input_frame)
        
#         # Check that the output has the correct shape
#         assert output_frame.data.shape == noisy_frame.shape
        
#         # Check that the output is valid (not all zeros or all ones)
#         assert np.any(output_frame.data > 0)
#         assert np.any(output_frame.data < 255)


# def test_denoise_processor_fast_vs_standard(noisy_frame):
#     """Test comparison between fast and standard non-local means."""
#     # Test fast non-local means
#     fast_processor = DenoiseProcessor(use_fast_nl_means=True)
#     fast_output = fast_processor.process_frame(ProcessedFrame(data=noisy_frame, frame_id=0))
    
#     # Test standard non-local means
#     standard_processor = DenoiseProcessor(use_fast_nl_means=False)
#     standard_output = standard_processor.process_frame(ProcessedFrame(data=noisy_frame, frame_id=0))
    
#     # Both methods should reduce noise
#     assert np.std(fast_output.data) < np.std(noisy_frame)
#     assert np.std(standard_output.data) < np.std(noisy_frame)
    
#     # The results should be different but similar
#     assert not np.array_equal(fast_output.data, standard_output.data)
#     assert np.mean(np.abs(fast_output.data - standard_output.data)) < 50


# def test_denoise_processor_preserves_frame_id(noisy_frame):
#     """Test that the frame ID is preserved during processing."""
#     frame_id = 42
#     input_frame = ProcessedFrame(data=noisy_frame, frame_id=frame_id)
#     processor = DenoiseProcessor()
#     output_frame = processor.process_frame(input_frame)
    
#     assert output_frame.frame_id == frame_id


# def test_denoise_processor_parallel_processing(noisy_frames):
#     """Test parallel processing of frames."""
#     # Create processor with 2 workers
#     processor = DenoiseProcessor(strength=10.0, num_workers=2)
    
#     # Create input frames with ProcessorFrame wrapper
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(noisy_frames)]
    
#     # Process frames in parallel
#     output_frames = processor.process_batch(input_frames)
    
#     # Check results
#     assert len(output_frames) == len(input_frames)
#     for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
#         # Check frame ID preservation
#         assert output_frame.frame_id == i
        
#         # Check output shape
#         assert output_frame.data.shape == input_frame.data.shape
        
#         # Check noise reduction
#         input_noise = np.std(input_frame.data)
#         output_noise = np.std(output_frame.data)
#         assert output_noise < input_noise


# def test_denoise_processor_sequential_vs_parallel(noisy_frames):
#     """Test that parallel processing produces same results as sequential."""
#     # Create processor with 2 workers
#     parallel_processor = DenoiseProcessor(strength=10.0, num_workers=2)
#     sequential_processor = DenoiseProcessor(strength=10.0, num_workers=0)
    
#     # Create input frames
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(noisy_frames)]
    
#     # Process frames both ways
#     parallel_output = parallel_processor.process_batch(input_frames)
#     sequential_output = sequential_processor.process_batch(input_frames)
    
#     # Compare results
#     assert len(parallel_output) == len(sequential_output)
#     for p_out, s_out in zip(parallel_output, sequential_output):
#         # Results should be similar (may not be identical due to non-deterministic nature of denoising)
#         assert np.mean(np.abs(p_out.data - s_out.data)) < 1.0
#         assert p_out.frame_id == s_out.frame_id


# def test_denoise_processor_empty_batch(processor):
#     """Test processing of empty batch."""
#     output_frames = processor.process_batch([])
#     assert len(output_frames) == 0


# def test_denoise_processor_single_frame_batch(processor, noisy_frame):
#     """Test processing of single frame batch."""
#     input_frame = ProcessedFrame(data=noisy_frame, frame_id=0)
#     output_frames = processor.process_batch([input_frame])
    
#     assert len(output_frames) == 1
#     assert output_frames[0].frame_id == 0
    
#     # Check noise reduction
#     input_noise = np.std(noisy_frame)
#     output_noise = np.std(output_frames[0].data)
#     assert output_noise < input_noise


# def test_denoise_processor_large_batch(noisy_frame):
#     """Test processing of large batch."""
#     # Create a large batch of frames
#     large_batch = [noisy_frame.copy() for _ in range(20)]
#     input_frames = [ProcessedFrame(data=frame, frame_id=i) for i, frame in enumerate(large_batch)]
    
#     # Process with different worker counts
#     for num_workers in [0, 2, 4]:
#         processor = DenoiseProcessor(strength=10.0, num_workers=num_workers)
#         output_frames = processor.process_batch(input_frames)
        
#         assert len(output_frames) == len(input_frames)
#         for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
#             assert output_frame.frame_id == i
#             assert output_frame.data.shape == input_frame.data.shape
            
#             # Check noise reduction
#             input_noise = np.std(input_frame.data)
#             output_noise = np.std(output_frame.data)
#             assert output_noise < input_noise 