"""
Model architectures for APISR.

This module contains implementations of the core model architectures used by APISR:
- RRDB (Residual in Residual Dense Block) Networks
- GRL (Gated Residual Layer) Networks
- DAT (Dual Aggregation Transformer) Networks
- CUNET (Real-CUGAN) Networks

These are extracted from the APISR repository to make the processor self-contained.
"""

import functools
import math
from os import PathLike
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange
from itertools import repeat
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from src.core.video.frames.processors.single_frame.ai.upscale.models.aspir_architecture.dat import DAT
from src.core.video.frames.processors.single_frame.ai.upscale.models.aspir_architecture.grl import (
    GRL,
)
from src.core.video.frames.processors.single_frame.ai.upscale.models.aspir_architecture.rrdb import (
    RRDBNet,
)


def load_grl(generator_weight_PATH, scale=4):
    """A simpler API to load GRL model
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int):        Scale Factor (Usually Set as 4)
    Returns:
        generator (torch): the generator instance of the model
    """

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if "model_state_dict" in checkpoint_g:
        weight = checkpoint_g["model_state_dict"]

        # GRL tiny model (Note: tiny2 version)
        generator = GRL(
            upscale=scale,
            img_size=64,
            window_size=8,
            depths=[4, 4, 4, 4],
            embed_dim=64,
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            mlp_ratio=2,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_window_down_factor=2,
            out_proj_type="linear",
            conv_type="1conv",
            upsampler="nearest+conv",  # Change
        )

    else:
        raise ValueError("This weight is not supported")

    generator.load_state_dict(weight)
    generator = generator.eval()

    num_params = 0
    for p in generator.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")

    return generator


# Keep the load functions intact
def load_rrdb(weights_path, scale=4, print_options=False):
    """Load RRDB model from weights path.

    Args:
        weights_path: Path to the model weights file
        scale: Upscaling factor (default: 4)
        print_options: Whether to print model options

    Returns:
        Loaded RRDB model
    """
    # Load the checkpoint
    checkpoint_g = torch.load(weights_path)

    # Find the generator weight
    if "params_ema" in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g["params_ema"]
        generator = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale)

    elif "params" in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g["params"]
        generator = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale)

    elif "model_state_dict" in checkpoint_g:
        # For personal trained weight
        weight = checkpoint_g["model_state_dict"]
        generator = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale)

    else:
        raise ValueError("This weight format is not supported")

    # Handle torch.compile weight key rename
    old_keys = [key for key in weight]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    # Check if key names in weight match expected model keys
    generator.load_state_dict(weight)
    generator = generator.eval()

    # Print options to show what kinds of setting is used
    if print_options:
        if "opt" in checkpoint_g:
            for key in checkpoint_g["opt"]:
                value = checkpoint_g["opt"][key]
                print(f"{key} : {value}")

    return generator


def load_dat(generator_weight_PATH, scale=4):

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if "model_state_dict" in checkpoint_g:
        weight = checkpoint_g["model_state_dict"]

        # DAT small model in default
        generator = DAT(
            upscale=4,
            in_chans=3,
            img_size=64,
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            expansion_factor=2,
            resi_connection="1conv",
            split_size=[8, 16],
            upsampler="pixelshuffledirect",
        )

    else:
        raise ValueError("This weight is not supported")

    generator.load_state_dict(weight)
    generator = generator.eval()

    num_params = 0
    for p in generator.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")

    return generator


def process_image(img_input, downsample_threshold=720):
    """Process image for model input.

    Args:
        img_input: Path to image or numpy array
        downsample_threshold: Threshold for downsampling

    Returns:
        Processed image tensor
    """
    # Load image if path is provided, otherwise use the numpy array
    if isinstance(img_input, PathLike):
        img = Image.open(img_input).convert("RGB")
        img_np = np.array(img)
    else:
        img_np = img_input

    # Resize if too large
    h, w = img_np.shape[:2]
    if max(h, w) > downsample_threshold:
        raise ValueError(
            f"Image is too large: {h}x{w}. Max allowed is {downsample_threshold}."
        )
        scale = downsample_threshold / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(img_np).resize((new_w, new_h), Image.LANCZOS)
        img_np = np.array(img)

    # Convert to tensor
    img_tensor = transforms.ToTensor()(img_np)
    return img_tensor


def super_resolve_img(
    model,
    img_input,
    output_path=None,
    weight_dtype=torch.float32,
    downsample_threshold=720,
    crop_for_4x=True,
):
    """Super-resolve an image using the given model.

    Args:
        model: The super-resolution model
        img_input: Path to image or numpy array
        output_path: Path to save output image (optional)
        weight_dtype: Data type for model weights
        downsample_threshold: Threshold for downsampling
        crop_for_4x: Whether to crop to ensure dimensions are divisible by 4

    Returns:
        Super-resolved image tensor
    """
    # Set model to appropriate device and datatype
    device = next(model.parameters()).device
    model = model.to(device=device, dtype=weight_dtype)

    # Process input image
    img_tensor = process_image(img_input, downsample_threshold)

    # For 4x scaling, crop to ensure dimensions are divisible by 4
    if crop_for_4x and hasattr(model, "upscale") and model.upscale == 4:
        _, h, w = img_tensor.shape
        h = h - h % 4
        w = w - w % 4
        img_tensor = img_tensor[:, :h, :w]

    # Convert to batch and move to device
    input_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=weight_dtype)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Save output if path provided
    if output_path:
        transforms.ToPILImage()(output[0].cpu()).save(output_path)

    return output[0]
