"""
Tests for the Padder utility class.

This module contains tests for the Padder class which handles image padding
operations for neural network processing.
"""

import numpy as np
import pytest
from PIL import Image

from src.core.video.frames.utils.padder import Padder


class TestPadder:
    """Test suite for the Padder class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        padder = Padder()
        assert padder.mod_pad == 32
        assert padder.pad_size == 15
        assert padder.scale_factor == 1
        assert padder.padding_info is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        padder = Padder(mod_pad=64, pad_size=10, scale_factor=2)
        assert padder.mod_pad == 64
        assert padder.pad_size == 10
        assert padder.scale_factor == 2
        assert padder.padding_info is None

    def test_create_class_method(self):
        """Test the create class method."""
        padder = Padder.create(mod_pad=64, pad_size=10, scale_factor=2)
        assert padder.mod_pad == 64
        assert padder.pad_size == 10
        assert padder.scale_factor == 2
        assert padder.padding_info is None

    def test_pad_image_rgb(self):
        """Test applying padding to an RGB image."""
        # Create a simple RGB image
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        img[25:75, 50:100, :] = 255  # White rectangle in the middle

        padder = Padder(mod_pad=32)
        padded_img = padder.pad_image(img)

        # Check that dimensions are divisible by mod_pad
        assert padded_img.shape[0] % 32 == 0
        assert padded_img.shape[1] % 32 == 0
        
        # Check that original dimensions are correctly stored in padding_info
        padding_info = padder.get_padding_info()
        assert padding_info["orig_height"] == 100
        assert padding_info["orig_width"] == 150

    def test_pad_image_grayscale(self):
        """Test applying padding to a grayscale image."""
        # Create a simple grayscale image
        img = np.zeros((100, 150), dtype=np.uint8)
        img[25:75, 50:100] = 255  # White rectangle in the middle

        padder = Padder(mod_pad=32)
        padded_img = padder.pad_image(img)

        # Check that dimensions are divisible by mod_pad
        assert padded_img.shape[0] % 32 == 0
        assert padded_img.shape[1] % 32 == 0
        
        # Check that original dimensions are correctly stored in padding_info
        padding_info = padder.get_padding_info()
        assert padding_info["orig_height"] == 100
        assert padding_info["orig_width"] == 150

    def test_pad_image_already_divisible(self):
        """Test applying padding to an image that's already divisible by mod_pad."""
        # Create a simple RGB image with dimensions divisible by 32
        img = np.zeros((128, 160, 3), dtype=np.uint8)
        img[32:96, 48:112, :] = 255  # White rectangle in the middle

        padder = Padder(mod_pad=32, pad_size=0)  # No edge padding needed
        padded_img = padder.pad_image(img)
        
        # The image should still be padded due to reflection padding
        assert padded_img.shape[0] >= img.shape[0]
        assert padded_img.shape[1] >= img.shape[1]

    @pytest.mark.parametrize("pad_size", [0, 15])
    @pytest.mark.parametrize("mod_pad", [0, 32])
    def test_unpad_image_rgb(self, pad_size, mod_pad):
        """Test unpadding an RGB image after padding."""
        # Create a simple RGB image
        original_img = np.zeros((100, 150, 3), dtype=np.uint8)
        original_img[25:75, 50:100, :] = 255  # White rectangle in the middle

        padder = Padder(mod_pad=32, pad_size=pad_size)
        padded_img = padder.pad_image(original_img)
        
        # Unpad the image
        unpadded_img = padder.unpad_image(padded_img)
        
        # Check that the unpadded image has the same dimensions as the original
        assert unpadded_img.shape == original_img.shape
        
        # Check that the content is preserved
        np.testing.assert_array_equal(unpadded_img, original_img)

    def test_unpad_image_grayscale(self):
        """Test unpadding a grayscale image after padding."""
        # Create a simple grayscale image
        original_img = np.zeros((100, 150), dtype=np.uint8)
        original_img[25:75, 50:100] = 255  # White rectangle in the middle

        padder = Padder(mod_pad=32)
        padded_img = padder.pad_image(original_img)
        
        # Unpad the image
        unpadded_img = padder.unpad_image(padded_img)
        
        # Check that the unpadded image has the same dimensions as the original
        assert unpadded_img.shape == original_img.shape
        
        # Check that the content is preserved
        np.testing.assert_array_equal(unpadded_img, original_img)

    def test_upscaling_scenario(self):
        """Test padding, upscaling (simulated), and unpadding."""
        # Create a simple RGB image
        original_img = np.zeros((100, 150, 3), dtype=np.uint8)
        original_img[25:75, 50:100, :] = 255  # White rectangle in the middle

        # Set up padder with scale factor of 2
        scale_factor = 2
        padder = Padder(mod_pad=32, scale_factor=scale_factor)
        
        # Apply padding
        padded_img = padder.pad_image(original_img)
        
        # Simulate upscaling by doubling the dimensions
        h, w, c = padded_img.shape
        upscaled_img = np.zeros((h * scale_factor, w * scale_factor, c), dtype=np.uint8)
        # Simple nearest-neighbor upscaling for testing
        for i in range(scale_factor):
            for j in range(scale_factor):
                upscaled_img[i::scale_factor, j::scale_factor, :] = padded_img
        
        # Unpad the upscaled image
        unpadded_img = padder.unpad_image(upscaled_img)
        
        # Check that the unpadded image has the expected dimensions
        assert unpadded_img.shape == (original_img.shape[0] * scale_factor, 
                                     original_img.shape[1] * scale_factor, 
                                     original_img.shape[2])
        
        # Verify that the white rectangle was upscaled correctly
        # The original rectangle was at [25:75, 50:100]
        # After upscaling, it should be at [50:150, 100:200]
        assert np.all(unpadded_img[50:150, 100:200, :] == 255)
        # Check that areas outside the rectangle are still black
        assert np.all(unpadded_img[0:50, :, :] == 0)
        assert np.all(unpadded_img[150:, :, :] == 0)
        assert np.all(unpadded_img[:, 0:100, :] == 0)
        assert np.all(unpadded_img[:, 200:, :] == 0)

    def test_dimensions_not_divisible_by_mod_pad(self):
        """Test handling images with dimensions not divisible by mod_pad."""
        # Create image with dimensions that aren't divisible by mod_pad
        img = np.zeros((123, 157, 3), dtype=np.uint8)
        img[30:70, 40:100, :] = 255  # White rectangle
        
        padder = Padder(mod_pad=32)
        padded_img = padder.pad_image(img)
        
        # Check that padded dimensions are divisible by mod_pad
        assert padded_img.shape[0] % 32 == 0
        assert padded_img.shape[1] % 32 == 0
        
        # Unpad and verify
        unpadded_img = padder.unpad_image(padded_img)
        assert unpadded_img.shape == img.shape
        np.testing.assert_array_equal(unpadded_img, img)

    def test_error_on_dimension_mismatch(self):
        """Test that an error is raised when unpadded dimensions don't match expected."""
        # Create a simple RGB image
        original_img = np.zeros((100, 150, 3), dtype=np.uint8)
        
        padder = Padder(mod_pad=32)
        padded_img = padder.pad_image(original_img)
        
        # Get and modify the padding info
        padding_info = padder.get_padding_info().copy()
        padding_info["orig_height"] = 101  # Wrong height
        
        # Verify that an error is raised
        with pytest.raises(ValueError):
            padder.unpad_image(padded_img, padding_info)
            
    def test_get_padding_info_without_padding(self):
        """Test that get_padding_info raises an error if called before pad_image."""
        padder = Padder()
        
        with pytest.raises(ValueError):
            padder.get_padding_info()
            
    def test_unpad_without_padding_info(self):
        """Test that unpad_image raises an error if called without padding info available."""
        padder = Padder()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            padder.unpad_image(img)
            
    def test_explicit_padding_info(self):
        """Test using explicit padding info with unpad_image."""
        # Create a simple RGB image
        original_img = np.zeros((100, 150, 3), dtype=np.uint8)
        original_img[25:75, 50:100, :] = 255  # White rectangle in the middle

        # First padder for padding
        padder1 = Padder(mod_pad=32)
        padded_img = padder1.pad_image(original_img)
        padding_info = padder1.get_padding_info()
        
        # Second padder for unpadding with explicit padding info
        padder2 = Padder(mod_pad=32)
        unpadded_img = padder2.unpad_image(padded_img, padding_info)
        
        # Check that the unpadded image has the same dimensions as the original
        assert unpadded_img.shape == original_img.shape
        
        # Check that the content is preserved
        np.testing.assert_array_equal(unpadded_img, original_img) 