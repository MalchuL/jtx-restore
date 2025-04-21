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
from RealESRGAN.utils import pad_reflect, unpad_image


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
    
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        """
        Process and upscale an image in one pass without patch splitting.
        
        Args:
            lr_image: Input low-resolution image (numpy array or PIL Image)
            pad_size: Size of padding to apply before processing
            
        Returns:
            PIL.Image: Upscaled image
        """
        device = self.device
        scale = self.scale
        mod_pad = self.mod_pad
        
        # Convert to numpy array if needed
        if not isinstance(lr_image, np.ndarray):
            lr_image = np.array(lr_image)
            
        # Remember original dimensions for later cropping
        orig_height, orig_width = lr_image.shape[0], lr_image.shape[1]
            
        # Calculate padding needed to make dimensions divisible by mod_pad
        # Add pad_size to each dimension to account for edge padding
        h_padded = orig_height + 2 * pad_size
        w_padded = orig_width + 2 * pad_size
        
        # Calculate extra padding needed for mod_pad divisibility
        extra_h = (mod_pad - h_padded % mod_pad) % mod_pad
        extra_w = (mod_pad - w_padded % mod_pad) % mod_pad
        
        # Use maximum padding value required for both edge handling and divisibility
        max_pad = max(pad_size, extra_h // 2, extra_w // 2)
        
        # Apply single padding value using pad_reflect
        lr_image_padded = pad_reflect(lr_image, max_pad)
        
        # Get dimensions after padding
        h_after_pad = lr_image_padded.shape[0]
        w_after_pad = lr_image_padded.shape[1]
        
        # Calculate crop to ensure divisibility by mod_pad
        # We add padding and then crop to the nearest valid dimensions
        h_remainder = h_after_pad % mod_pad
        w_remainder = w_after_pad % mod_pad
        
        # If not divisible, crop excess pixels
        h_crop = h_remainder
        w_crop = w_remainder
        
        # Ensure we maintain equal padding on both sides
        h_crop_top = h_crop // 2
        h_crop_bottom = h_crop - h_crop_top
        w_crop_left = w_crop // 2
        w_crop_right = w_crop - w_crop_left
        
        # Crop to get dimensions divisible by mod_pad
        if h_crop > 0 or w_crop > 0:
            h_valid = h_after_pad - h_crop
            w_valid = w_after_pad - w_crop
            lr_image_padded = lr_image_padded[h_crop_top:h_crop_top+h_valid, 
                                             w_crop_left:w_crop_left+w_valid, :]
        
        # Calculate actual padding on each side after cropping
        # This accounts for both the padding we added and the cropping we did
        effective_pad_top = max_pad - h_crop_top
        effective_pad_bottom = max_pad - h_crop_bottom
        effective_pad_left = max_pad - w_crop_left
        effective_pad_right = max_pad - w_crop_right
        
        # Prepare input tensor
        img = torch.FloatTensor(lr_image_padded / 255.0).permute(2, 0, 1).unsqueeze(0)
        img = img.to(device)
        
        # Process with model
        with torch.no_grad():
            output = self.model(img)
            
        # Convert output tensor to image
        sr_image = output.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
        sr_img = (sr_image * 255).astype(np.uint8)
        
        # Scale effective padding values
        pad_top_scaled = effective_pad_top * scale
        pad_bottom_scaled = effective_pad_bottom * scale
        pad_left_scaled = effective_pad_left * scale
        pad_right_scaled = effective_pad_right * scale
        
        # Get dimensions of output image
        h_out, w_out = sr_img.shape[0], sr_img.shape[1]
        
        # Crop out padding to get back to original aspect ratio
        # Use the effective padding values to ensure we get exactly the original image
        sr_img = sr_img[pad_top_scaled:h_out-pad_bottom_scaled,
                        pad_left_scaled:w_out-pad_right_scaled, :]
        
        if sr_img.shape[0] != orig_height * scale or sr_img.shape[1] != orig_width * scale or sr_img.shape[2] != 3:
            raise ValueError(f"Upscaled image dimensions ({sr_img.shape[0]}x{sr_img.shape[1]}x{sr_img.shape[2]}) do not match original dimensions ({orig_height}x{orig_width}x3)")
        # Return as PIL Image
        return Image.fromarray(sr_img) 