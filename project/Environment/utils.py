import numpy as np
from typing import Mapping, Sequence
import warnings

def map_index(keys: Sequence[object]):
    """Mapping the name to a number"""
    return dict(zip(keys, range(len(keys))))

def mapping2array(
    source_dict: Mapping[object, float], 
    key2id: Mapping[object, int],
    default_value: float = 0,
    exception_type: str = "warning",
):  
    """Grounding the name-to-number mapping ``source_dict`` to a target array 
    according to the name-to-id mapping ``key2id``

    Args:
        source_dict (Mapping[object, float]): A name-to-number mapping
        key2id (Mapping[object, int]): A name-to-id mapping
        default_value (float, optional): The default value of the target array. Defaults to 0.
        exception_type (str, optional): raise warning or error. Defaults to ``"warning"``.

    Raises:
        AssertionError: if ``exception_type == "error"``, 
            raise error when exists ``source_dict.keys()`` not in ``key2id.keys()``

    Returns:
        target_array : an array sorted according to ``key2id`` 
            and the value is from ``source_dict.values()``
    """
    
    target_array = np.full(
        (1 + max( key2id.values() ), ),
        default_value
    )
    for obj, value in source_dict.items():
        if obj not in key2id.keys():
            if exception_type == "warning":
                warnings.warn(f"Warning: {obj} not in the keys!", UserWarning, stacklevel=2)
            else:
                raise AssertionError(f"{obj} not in the keys {list(key2id.keys())}!")
        else:
            target_array[key2id[obj]] = value
    return target_array