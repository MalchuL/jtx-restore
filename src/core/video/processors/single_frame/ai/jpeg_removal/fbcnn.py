#!/usr/bin/env python
"""
RealESRGAN-based frame processor.

This module provides a frame processor that uses RealESRGAN for high-quality
image upscaling. RealESRGAN is particularly effective for anime and general
image upscaling.
"""

import itertools
import math
import os
import warnings
from typing import Any, Optional, List
import numpy as np
from PIL import Image
import requests
import pyrootutils
from src.core.video.processors.single_frame.ai.jpeg_removal.torch_models.fbcnn import FBCNN


# Check for PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Install with: pip install torch", RuntimeWarning)


# Both dependencies are required
FBCNN_AVAILABLE = TORCH_AVAILABLE

from src.core.video.processors.frame import ProcessedFrame
from src.core.video.processors.single_frame.ai.ai_processor import AIProcessor


class FBCNNProcessor(AIProcessor):
    """Frame processor using FBCNN for JPEG removal.

    This processor uses FBCNN to remove JPEG artifacts from video frames.
    """
    WEIGHTS_PATH = "weights/fbcnn/fbcnn_color.pth"
    WEIGHTS_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"

    def __init__(
        self,
        compression_factor: Optional[float] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
        ):
        """Initialize FBCNN processor.

        Args:
            compression_factor: Compression factor of the JPEG image
            model_name: path to FBCNN model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            batch_size: Number of frames to process in each batch, but batch means batch size for patching
        Raises:
            RuntimeError: If FBCNN dependencies are not installed
        """
        if not FBCNN_AVAILABLE:
            missing_deps = []
            if not TORCH_AVAILABLE:
                missing_deps.append("torch")
            raise RuntimeError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install " + " ".join(missing_deps)
            )

        # Set default device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        self.fbcnn = None
        self.qf = self._get_qf(compression_factor, device)

        self.tile = 512
        self.overlap = 32

        super().__init__(
            model_name=model_name, device=device, batch_size=1
        )
    def _get_qf(self, compression_level: Optional[float] = None, device: torch.device = None) -> float:
        """Get the quality factor of the JPEG image.

        Returns:
            Quality factor of the JPEG image
        """
        auto_detect = compression_level is None
        qf = None if auto_detect else torch.tensor([[1-compression_level/100]], dtype=torch.float32, device=device)
        return qf
    
    def _download_model(self, path: str) -> None:
        print(f"Downloading FBCNN model from Github")
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        url = self.WEIGHTS_URL
        r = requests.get(url, allow_redirects=True)
        with open(path, 'wb') as f:
            f.write(r.content)

    def _load_model(self) -> None:
        """Load the RealESRGAN model and upscaler.

        This method loads the model and initializes the RealESRGAN upscaler.
        """
        # Initialize model
        if self.model_name is None:
            path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
            model_name = os.path.join(path, self.WEIGHTS_PATH)
        if not os.path.exists(model_name):
            self._download_model(model_name)

        fbcnn = FBCNN()
        fbcnn.load_state_dict(torch.load(model_name), strict=True)
        fbcnn.eval()
        for param in fbcnn.parameters():
            param.requires_grad = False
        fbcnn.to(self.device)

        self.fbcnn = fbcnn

    def _tiled_scale(self, samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu"):
        return self._tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device)

    @torch.inference_mode()
    def _tiled_scale_multidim(self, samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None):
        dims = len(tile)

        if not (isinstance(upscale_amount, (tuple, list))):
            upscale_amount = [upscale_amount] * dims

        if not (isinstance(overlap, (tuple, list))):
            overlap = [overlap] * dims

        if index_formulas is None:
            index_formulas = upscale_amount

        if not (isinstance(index_formulas, (tuple, list))):
            index_formulas = [index_formulas] * dims

        def get_upscale(dim, val):
            up = upscale_amount[dim]
            if callable(up):
                return up(val)
            else:
                return up * val

        def get_downscale(dim, val):
            up = upscale_amount[dim]
            if callable(up):
                return up(val)
            else:
                return val / up

        def get_upscale_pos(dim, val):
            up = index_formulas[dim]
            if callable(up):
                return up(val)
            else:
                return up * val

        def get_downscale_pos(dim, val):
            up = index_formulas[dim]
            if callable(up):
                return up(val)
            else:
                return val / up

        if downscale:
            get_scale = get_downscale
            get_pos = get_downscale_pos
        else:
            get_scale = get_upscale
            get_pos = get_upscale_pos

        def mult_list_upscale(a):
            out = []
            for i in range(len(a)):
                out.append(round(get_scale(i, a[i])))
            return out

        output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

        for b in range(samples.shape[0]):
            s = samples[b:b+1]

            # handle entire input fitting in a single tile
            if all(s.shape[d+2] <= tile[d] for d in range(dims)):
                output[b:b+1] = function(s).to(output_device)
                continue

            out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
            out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

            positions = [range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) if s.shape[d+2] > tile[d] else [0] for d in range(dims)]

            for it in itertools.product(*positions):
                s_in = s
                upscaled = []

                for d in range(dims):
                    pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                    l = min(tile[d], s.shape[d + 2] - pos)
                    s_in = s_in.narrow(d + 2, pos, l)
                    upscaled.append(round(get_pos(d, pos)))

                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)

                for d in range(2, dims + 2):
                    feather = round(get_scale(d - 2, overlap[d - 2]))
                    if feather >= mask.shape[d]:
                        continue
                    for t in range(feather):
                        a = (t + 1) / feather
                        mask.narrow(d, t, 1).mul_(a)
                        mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

                o = out
                o_d = out_div
                for d in range(dims):
                    o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                    o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

                o.add_(ps * mask)
                o_d.add_(mask)

            output[b:b+1] = out/out_div
        return output

    def _preprocess(self, frame: ProcessedFrame) -> torch.Tensor:
        """Preprocess a frame for RealESRGAN input.

        Args:
            frame: Input frame to preprocess

        Returns:
            Preprocessed data ready for model input
        """
        torch_tensor = torch.from_numpy(frame.data).permute(2, 0, 1).float() / 255.0
        return torch_tensor.unsqueeze(0).to(self.device)

    def _postprocess(self, model_output: torch.Tensor) -> np.ndarray:
        """Postprocess RealESRGAN output into a frame.

        Args:
            model_output: Raw model output to postprocess

        Returns:
            Processed frame data as numpy array
        """
        assert isinstance(model_output, torch.Tensor)
        out = model_output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _infer_model(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run RealESRGAN inference on a batch of inputs.

        Args:
            inputs: List of preprocessed inputs

        Returns:
            List of model outputs
        """
        outputs = []
        for img in inputs:
            with torch.inference_mode():
                output = self._tiled_scale(
                    img,
                    lambda a: self.fbcnn.forward(a, self.qf),
                    tile_x=self.tile,
                    tile_y=self.tile,
                    overlap=self.overlap,
                    upscale_amount=1,
            )
            outputs.append(output)
        return outputs
