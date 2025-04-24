"""
Whole image RealESRGAN processor.

This module provides a modified RealESRGAN model that processes images as a whole
instead of splitting them into patches. This may be faster for smaller images or
when sufficient GPU memory is available.
"""

import torch
import numpy as np
from PIL import Image
from RealESRGAN import RealESRGAN

from src.core.video.frames.utils.padder import Padder


class WholeImageRealESRGAN(RealESRGAN):
    """
    RealESRGAN implementation that processes whole images at once.
    
    This class extends the standard RealESRGAN implementation but overrides
    the predict method to process entire images without splitting them into
    patches. This approach may be faster for smaller images or when sufficient
    GPU memory is available, but might cause out-of-memory errors for large
    images.
    """
    
    def __init__(self, device, scale=4, mod_pad=32):
        """
        Initialize WholeImageRealESRGAN.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            scale: Upscaling factor (2, 4, or 8)
            mod_pad: Ensure input dimensions are divisible by this value (typically 32)
        """
        super().__init__(device, scale)
        self.mod_pad = mod_pad
        self.padder = Padder(mod_pad=mod_pad, scale_factor=scale)
    
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        """
        Process and upscale an image in one pass without patch splitting.
        
        Args:
            lr_image: Input low-resolution image (numpy array or PIL Image)
            batch_size: Not used in whole image processing (kept for compatibility)
            patches_size: Not used in whole image processing (kept for compatibility)
            padding: Not used in whole image processing (kept for compatibility)
            pad_size: Size of padding to apply before processing
            
        Returns:
            PIL.Image: Upscaled image
        """
        device = self.device
        scale = self.scale
        
        # Update padder if pad_size is different from default
        if pad_size != self.padder.pad_size:
            self.padder = Padder(mod_pad=self.mod_pad, pad_size=pad_size, scale_factor=scale)
        
        # Convert to numpy array if needed
        if not isinstance(lr_image, np.ndarray):
            lr_image = np.array(lr_image)
            
        # Apply padding using the Padder utility
        padded_img = self.padder.pad_image(lr_image)
        
        # Prepare input tensor
        img = torch.FloatTensor(padded_img / 255.0).permute(2, 0, 1).unsqueeze(0)
        img = img.to(device)
        
        # Process with model
        with torch.no_grad():
            output = self.model(img)
            
        # Convert output tensor to image
        sr_image = output.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
        sr_img = (sr_image * 255).astype(np.uint8)
        
        # Remove padding to restore original dimensions with the scale factor
        sr_img = self.padder.unpad_image(sr_img)
        
        # Return as PIL Image
        if isinstance(sr_img, np.ndarray):
            return Image.fromarray(sr_img)
        return sr_img 