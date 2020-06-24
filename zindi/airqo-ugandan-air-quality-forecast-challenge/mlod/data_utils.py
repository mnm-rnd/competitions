import numpy as np
import math
from typing import Any

def replace_nan(x: str):
    """
    A function to replace empty spaces with nan
    Args:
        x (`str`): a string number
    
    Returns:
        - `np.nan`, if string (with removed spaces) is empty
        - `float`, if `x` is not empty

    """
    if not x.strip():
        return np.nan
    
    return float(x.strip())

def remove_nan_values(x: Any):
    '''
    A function to remove missing values from data

    Args: 
        x: a number | np.nan

    Returns:
        a list of values
        
    '''

    return [e for e in x if not math.isnan(e)]