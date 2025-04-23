# JTX-Restore: Classic Cartoon Restoration Suite

JTX-Restore is an open-source AI-powered video enhancement framework specifically optimized for classic cartoons and anime. It uses state-of-the-art neural networks to upscale, denoise, and enhance vintage animation without destroying the original artistic style.

## Features

- **Multiple AI Processors**: Support for various deep learning models:
  - **APISR**: Anime Production Inspired Real-world Anime Super-Resolution with multiple model variants (RRDB, GRL, DAT, CUNET)
  - **RealESRGAN**: Enhanced ESRGAN for realistic texture recreation
  - **FBCNN**: JPEG artifact removal for compressed sources
  - **Whole Image Processing**: Option for single-pass upscaling when sufficient GPU memory is available
  
- **Video Processing Pipeline**:
  - Frame-by-frame processing with customizable stages
  - Automated batch processing of entire directories
  - Color correction and normalization
  - RIFE-based frame interpolation

- **User-Friendly Interface**:
  - Streamlit-based processor visualization for testing and comparing different enhancement techniques
  - Hydra configuration system for flexible customization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jtx-restore.git
cd jtx-restore
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install additional dependencies for specific processors:
```bash
# For APISR and other AI processors
pip install torch torchvision
pip install opencv-python einops
```

## Running the Streamlit Demo

The Streamlit demo allows you to visualize and test different processors with your own images/videos:

```bash
cd apps/processor_visualizer
streamlit run streamlit_processor_visualization.py
```

This will open a web interface where you can:
- Upload a video clip
- Select which processors to apply (APISR, RealESRGAN, FBCNN, etc.)
- Adjust processor parameters
- Compare before/after results

## Running the Enhancement Pipeline

To enhance videos using the configured pipeline:

```bash
python main.py cartoons=jtx video_folder="<path_to_video>" output_folder="<path_to_output>" writer_kwargs.temp_dir="<path_to_tmp_folder>"
```

Where:
- `<path_to_video>`: Path to a video file or directory containing videos
- `<path_to_output>`: Path where enhanced videos will be saved
- `<path_to_tmp_folder>`: (Optional) Directory for temporary files during processing

### Configuration

The project uses Hydra for configuration. You can modify settings in the YAML files located in the `configs/` directory:

- `configs/processors/`: Individual processor configurations
- `configs/pipeline/`: Processing pipeline configurations
- `configs/cartoons/`: Preset configs for different cartoon styles

## Implementation Details

### APISR Processor

This project includes a fully self-contained implementation of the APISR (Anime Production Inspired Real-world Anime Super-Resolution) models:

1. **RRDB Models**: Residual-in-Residual Dense Block Networks for 2x and 4x upscaling
2. **GRL Models**: Gated Residual Layer networks for 4x upscaling
3. **DAT Models**: Dual Aggregation Transformer networks for 4x upscaling
4. **CUNET Models**: Real-CUGAN implementation for 2x upscaling

All models are implemented with proper weight loading and include specialized preprocessing to handle various input image sizes and formats.

### Whole Image Processing

For machines with sufficient GPU memory, we've implemented whole-image processing variants for models that traditionally operate on tiles or patches. This can provide better quality for smaller inputs and eliminate potential seam artifacts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- APISR: [https://github.com/Kiteretsu77/APISR](https://github.com/Kiteretsu77/APISR)
- Real-ESRGAN: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- FBCNN: [https://github.com/jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN) 