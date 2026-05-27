"""
Padding utilities for image processing.

This module provides a utility class for handling image padding operations
commonly needed when processing images with neural networks that require
specific input dimensions.
"""

import numpy as np
from typing import Dict, Any, Optional


class Padder:
    """
    Utility class for padding and cropping images for neural network processing.
    
    This class provides methods to handle common padding operations needed
    when working with neural networks that require input dimensions to be divisible
    by specific values (typically powers of 2).
    """
    
    def __init__(self, mod_pad: int = 32, pad_size: int = 15, scale_factor: int = 1):
        """
        Initialize the Padder with processing parameters.
        
        Args:
            mod_pad: Ensure dimensions are divisible by this value (default: 32)
            pad_size: Size of padding to apply for edge handling (default: 15)
            scale_factor: Scale factor for upscaling operations (default: 1)
        """
        self.mod_pad = mod_pad
        self.pad_size = pad_size
        self.scale_factor = scale_factor
        self.padding_info = None
    
    def _pad_reflect(self, img: np.ndarray, pad_size: Optional[int] = None) -> np.ndarray:
        """
        Apply reflection padding to an image.
        
        Args:
            img: Input image as numpy array (H, W, C)
            pad_size: Number of pixels to pad on each side (uses instance pad_size if None)
            
        Returns:
            Padded image
        """
        if pad_size is None:
            pad_size = self.pad_size
            
        if len(img.shape) == 3:  # RGB image
            return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        else:  # Grayscale image
            return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    
    def _calculate_divisible_padding(self, height: int, width: int) -> Dict[str, Any]:
        """
        Calculate padding needed to make dimensions divisible by mod_pad.
        
        Args:
            height: Original image height
            width: Original image width
            
        Returns:
            Dictionary with padding information
        """
        # Remember original dimensions for later cropping
        orig_height, orig_width = height, width
        pad_size = self.pad_size
        mod_pad = self.mod_pad
        
        # Calculate padding needed to make dimensions divisible by mod_pad
        # Add pad_size to each dimension to account for edge padding
        h_padded = orig_height + 2 * pad_size
        w_padded = orig_width + 2 * pad_size
        
        # Calculate extra padding needed for mod_pad divisibility
        extra_h = (mod_pad - h_padded % mod_pad) % mod_pad
        extra_w = (mod_pad - w_padded % mod_pad) % mod_pad
        
        # Use maximum padding value required for both edge handling and divisibility
        max_pad = max(pad_size, extra_h // 2, extra_w // 2)
        
        # Calculate crop to ensure divisibility by mod_pad after padding
        h_after_pad = orig_height + 2 * max_pad
        w_after_pad = orig_width + 2 * max_pad
        
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
        
        # Calculate actual padding on each side after cropping
        effective_pad_top = max_pad - h_crop_top
        effective_pad_bottom = max_pad - h_crop_bottom
        effective_pad_left = max_pad - w_crop_left
        effective_pad_right = max_pad - w_crop_right
        
        return {
            "orig_height": orig_height,
            "orig_width": orig_width,
            "max_pad": max_pad,
            "h_crop_top": h_crop_top,
            "h_crop_bottom": h_crop_bottom,
            "w_crop_left": w_crop_left,
            "w_crop_right": w_crop_right,
            "effective_pad_top": effective_pad_top,
            "effective_pad_bottom": effective_pad_bottom,
            "effective_pad_left": effective_pad_left,
            "effective_pad_right": effective_pad_right
        }
    
    def pad_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply padding to an image to ensure dimensions are divisible by mod_pad.
        
        This method handles the entire padding process, including reflection padding
        and ensuring dimensions are divisible by mod_pad. The padding information
        is stored in the padder instance for later use by unpad_image.
        
        Args:
            img: Input image as numpy array (H, W, C) or (H, W) for grayscale
            
        Returns:
            Padded image with dimensions divisible by mod_pad
        """
        # Get image dimensions
        if len(img.shape) == 3:
            height, width = img.shape[0], img.shape[1]
        else:
            height, width = img.shape
        
        # Calculate padding
        self.padding_info = self._calculate_divisible_padding(height, width)
        
        # Apply padding
        padded_img = self._pad_reflect(img, self.padding_info["max_pad"])
        
        # Get dimensions after padding
        if len(padded_img.shape) == 3:
            h_after_pad, w_after_pad = padded_img.shape[0], padded_img.shape[1]
        else:
            h_after_pad, w_after_pad = padded_img.shape
        
        # Crop to get dimensions divisible by mod_pad
        h_crop_top = self.padding_info["h_crop_top"]
        h_crop_bottom = self.padding_info["h_crop_bottom"]
        w_crop_left = self.padding_info["w_crop_left"]
        w_crop_right = self.padding_info["w_crop_right"]
        
        h_valid = h_after_pad - (h_crop_top + h_crop_bottom)
        w_valid = w_after_pad - (w_crop_left + w_crop_right)
        
        if len(padded_img.shape) == 3:
            padded_img = padded_img[h_crop_top:h_crop_top+h_valid, 
                                    w_crop_left:w_crop_left+w_valid, :]
        else:
            padded_img = padded_img[h_crop_top:h_crop_top+h_valid, 
                                    w_crop_left:w_crop_left+w_valid]
            
        return padded_img
    
    def unpad_image(self, img: np.ndarray, padding_info: Dict[str, Any] = None) -> np.ndarray:
        """
        Remove padding from an image based on padding information.
        
        This method removes the padding added by pad_image, scaling the padding
        values based on the scale_factor if the image has been upscaled.
        
        Args:
            img: Input padded image as numpy array (potentially upscaled)
            padding_info: Optional padding information dictionary. If None, uses
                          the stored padding_info from the last pad_image call.
            
        Returns:
            Unpadded image with original aspect ratio
        """
        if padding_info is None:
            if self.padding_info is None:
                raise ValueError("No padding information available. Call pad_image first or provide padding_info.")
            padding_info = self.padding_info
            
        scale_factor = self.scale_factor
            
        # Scale effective padding values
        pad_top_scaled = padding_info["effective_pad_top"] * scale_factor
        pad_bottom_scaled = padding_info["effective_pad_bottom"] * scale_factor
        pad_left_scaled = padding_info["effective_pad_left"] * scale_factor
        pad_right_scaled = padding_info["effective_pad_right"] * scale_factor
        
        # Get dimensions of output image
        if len(img.shape) == 3:
            h_out, w_out = img.shape[0], img.shape[1]
        else:
            h_out, w_out = img.shape
            
        # Crop out padding to get back to original aspect ratio
        if len(img.shape) == 3:
            img = img[pad_top_scaled:h_out-pad_bottom_scaled,
                       pad_left_scaled:w_out-pad_right_scaled, :]
        else:
            img = img[pad_top_scaled:h_out-pad_bottom_scaled,
                       pad_left_scaled:w_out-pad_right_scaled]
            
        # Verify dimensions are correct
        orig_height = padding_info["orig_height"] * scale_factor
        orig_width = padding_info["orig_width"] * scale_factor
        
        if len(img.shape) == 3:
            if img.shape[0] != orig_height or img.shape[1] != orig_width:
                raise ValueError(f"Unpadded image dimensions ({img.shape[0]}x{img.shape[1]}) do not match "
                               f"expected dimensions ({orig_height}x{orig_width})")
        else:
            if img.shape[0] != orig_height or img.shape[1] != orig_width:
                raise ValueError(f"Unpadded image dimensions ({img.shape[0]}x{img.shape[1]}) do not match "
                               f"expected dimensions ({orig_height}x{orig_width})")
                
        return img
    
    def get_padding_info(self) -> Dict[str, Any]:
        """
        Get the padding information from the last pad_image operation.
        
        Returns:
            Dictionary containing padding information
            
        Raises:
            ValueError: If pad_image has not been called yet
        """
        if self.padding_info is None:
            raise ValueError("No padding information available. Call pad_image first.")
        return self.padding_info

    @classmethod
    def create(cls, mod_pad: int = 32, pad_size: int = 15, scale_factor: int = 1) -> 'Padder':
        """
        Create a Padder instance with the specified parameters.
        
        This is a convenience method to create a Padder instance with custom parameters.
        
        Args:
            mod_pad: Ensure dimensions are divisible by this value
            pad_size: Size of padding to apply for edge handling
            scale_factor: Scale factor for upscaling operations
            
        Returns:
            Padder instance with the specified parameters
        """
        return cls(mod_pad=mod_pad, pad_size=pad_size, scale_factor=scale_factor)