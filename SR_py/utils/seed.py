# -*- coding: utf-8 -*-
"""
Utilities for managing random seeds for reproducibility.
"""
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def seed_all(seed: int):
    """
    Sets the random seed for Python's random module, NumPy, and other
    relevant libraries to ensure reproducibility.
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")
        
    random.seed(seed)
    np.random.seed(seed)
    # If using torch or other libraries, set their seeds here too.
    # import torch
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
        
    logger.info(f"Global random seed set to {seed}.")
