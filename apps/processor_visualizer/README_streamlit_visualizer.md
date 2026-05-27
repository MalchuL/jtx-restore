# Frame Processor Visualization App

This Streamlit application allows you to visualize the effects of different frame processors in the JTX Restoration pipeline. You can upload a video, select frames, and see how different processors affect the image quality.

## Features

- Upload and visualize video files
- Select specific frames to process
- Choose from multiple frame processors:
  - Color Correction (brightness, contrast, saturation, gamma adjustment)
  - Denoising
  - Upscaling
  - RIFE Frame Interpolation (if available)
- Adjust processor parameters in real-time
- View the original frame, processed results, and visual differences
- Inspect frame metadata for each processing stage

## Requirements

```
streamlit>=1.10.0
opencv-python>=4.5.0
numpy>=1.20.0
pillow>=9.0.0
```

If you want to use the RIFE Frame Interpolation processor, you'll also need:

```
torch>=1.10.0
```

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install streamlit opencv-python numpy pillow
# Optional: for RIFE frame interpolation
pip install torch
```

2. If you want to use the RIFE frame interpolation, download the model weights:

```bash
# Create directory for the model weights
mkdir -p weights/practical_rife_4_25

# Download the model weights
# You can manually download from the appropriate source and place the 
# flownet.pkl file in weights/practical_rife_4_25/
```

## Running the App

Run the Streamlit app with:

```bash
# From the project root directory
streamlit run src/apps/streamlit_processor_visualization.py
```

The app will open in your web browser at `http://localhost:8501`.

## Usage

1. Upload a video file or use the example image
2. Select processors from the sidebar
3. Adjust processor parameters as needed
4. If using a video, use the slider to navigate through frames
5. Click the "Process Frame" button
6. View the original and processed frames side by side
7. Compare the visual differences and metadata

## Tips

- For best performance, use videos with resolution lower than 1080p
- The RIFE interpolation requires a GPU for optimal performance
- The frame selection is limited to the first 500 frames for large videos to prevent memory issues
- You can save screenshots of the results by right-clicking on the images

## Developer Notes

This app demonstrates how to extend the `FrameProcessorManager` class to capture intermediate processing results and visualize them. The implementation is in:

- `src/core/video/frames/processors/visualizer/streamlit_processor_visualizer.py`: The visualizer class
- `src/apps/streamlit_processor_visualization.py`: The Streamlit application

You can use this as a reference for creating your own custom visualization tools. 