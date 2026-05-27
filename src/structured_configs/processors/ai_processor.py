"""
Structured config for AIProcessor.

This module provides a dataclass for AIProcessor configuration to be used with Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

@dataclass
class AIProcessorConfig:
    """Hydra config dataclass for AIProcessor (abstract base class)."""

    # Model name or path to load
    model_name: Optional[str] = None
    
    # Device to run the model on (implementation specific)
    device: Optional[str] = None
    
