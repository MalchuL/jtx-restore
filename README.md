# JTX Restoration

A video processing and restoration framework for enhancing video quality through various processing techniques.

## Features

- **Modular Architecture**: Easily extendable with new video processors
- **Multiple Enhancement Types**: 
  - Denoising (single frame and batch processing)
  - Color correction (sequential and parallel)
  - Upscaling
- **Configurable Pipeline**: Use Hydra configuration for easy setup and customization
- **Efficient Processing**: Supports batch processing and multi-threading
- **Progress Tracking**: Real-time progress visualization during processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/jtx_restoration.git
cd jtx_restoration
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command-line Interface

The simplest way to use the framework is through the CLI:

```bash
python src/cli.py reader.source_path=/path/to/input.mp4 writer.output_path=/path/to/output.mp4
```

All configuration options can be overridden on the command line using Hydra's dot notation.

### Configuration

The framework uses Hydra for configuration management. The default configuration is in `configs/video_pipeline.yaml`.

You can create your own configuration files or override settings on the command line:

```bash
# Override configuration on the command line
python src/cli.py reader.type=folder_cache reader.cache_dir=/tmp/cache processing.batch_size=16

# Use a different configuration file
python src/cli.py --config-name=my_custom_config
```

#### Example Configuration

```yaml
# Main pipeline configuration
reader:
  type: opencv  # opencv or folder_cache
  source_path: /path/to/input.mp4
  cache_dir: null  # Only used with folder_cache reader

writer:
  type: opencv
  output_path: /path/to/output.mp4
  fps: null  # Uses input fps if null
  frame_size: null  # Uses input size if null
  codec: mp4v

processing:
  batch_size: 8
  max_workers: 4
  use_threading: true

processors:
  - type: denoise
    enabled: true
    params:
      strength: 10
      method: fast
  
  - type: color_correction
    enabled: true
    params:
      saturation: 1.2
      contrast: 1.1
      brightness: 1.0
  
  - type: upscale
    enabled: false
    params:
      scale: 2.0
      model: "realesrgan"

logging:
  level: INFO
  log_to_file: false
  log_file: logs/pipeline.log
```

## Extending the Framework

### Adding New Processors

To add a new processor:

1. Create a new class that extends `FrameProcessor` or `BatchFrameProcessor`
2. Implement the required methods (especially `process_frame` or `process_batch_optimized`)
3. Add your processor to the configuration system in `src/core/pipeline.py`

### Custom Reader or Writer

To add a custom reader or writer:

1. Create a new class that extends `VideoReader` or `VideoWriter`
2. Implement all required abstract methods
3. Add your reader/writer to the configuration system in `src/core/pipeline.py`

## Development

### Running Tests

```bash
pytest
```

### Code Structure

- `src/core/readers`: Video input modules
- `src/core/writers`: Video output modules
- `src/core/processors`: Frame processing modules
  - `base.py`: Base processor classes
  - `batch.py`: Batch processing support
  - `frame.py`: Frame data container
  - `pipeline.py`: Processor chaining
  - `enhancers/`: Various enhancement processors
- `configs/`: Hydra configuration files
- `tests/`: Unit and integration tests

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 