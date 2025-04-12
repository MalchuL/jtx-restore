# RIFE (Real-Time Intermediate Flow Estimation) Frame Interpolator

This module provides a frame interpolator implementation based on the RIFE (Real-Time Intermediate Flow Estimation) algorithm, which is designed for real-time frame interpolation.

## Model Information

The RIFE model used in this implementation is version 4.25 (2024.09.19) from the [Practical-RIFE](https://github.com/hzwer/Practical-RIFE/tree/main) repository.

### Model Download

The model weights can be downloaded from:
[https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing)

After downloading, place the model file (`flownet.pkl`) in the following directory:
```
weights/practical_rife_4_25/flownet.pkl
```

## Usage

To use the RIFE frame interpolator, you need to:

1. Download the model weights as described above
2. Initialize the interpolator with the path to the model directory:

```python
from src.core.video.frame_interpolation.ai.rife import PracticalRIFEFrameInterpolator425

interpolator = PracticalRIFEFrameInterpolator425(
    factor=2.0,  # Frame rate increase factor (e.g., 2.0 doubles the frame rate)
    model_path="weights/practical_rife_4_25",  # Path to the model directory
    scale=1.0  # Scale factor for the model (1.0 for original resolution)
)
```

3. Process frames one by one or in batches:

```python
# Process a single frame
result = interpolator(frame)

# Process remaining frames
result = interpolator(None)
```

For a complete example, see `examples/rife_interpolation_example.py`.

## License

The RIFE model is provided under the terms of the original repository's license. Please refer to the [Practical-RIFE](https://github.com/hzwer/Practical-RIFE/tree/main) repository for more information. 