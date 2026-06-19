import numpy as np
from enum import Enum

class Transforms(Enum):
    RESHAPE          = 'RESHAPE'
    RESHAPE_INVERSE  = 'RESHAPE_INVERSE'
    BROADCAST_TO     = 'BROADCAST_TO'
    COLLAPSE         = 'COLLAPSE'
    SLICE            = 'SLICE'
    PAD              = 'PAD'
    SQUEEZE          = 'SQUEEZE'
    UNSQUEEZE        = 'UNSQUEEZE'
    FLATTEN          = 'FLATTEN'
    FLATTEN_INVERSE  = 'FLATTEN_INVERSE'


class TransformFunctions:

    @staticmethod
    def reshape(data, shape):
        return data.reshape(shape)

    @staticmethod
    def reshape_inverse(data, original_shape):
        return data.reshape(original_shape)

    @staticmethod
    def broadcast_to(data, shape):
        return np.broadcast_to(data, shape)

    @staticmethod
    def collapse(data, original_shape):
        # Sum over broadcast dimensions to collapse back to original shape
        axis = tuple(range(len(data.shape) - len(original_shape)))
        keepdims_axes = tuple(i for i, (d, o) in enumerate(zip(data.shape[-len(original_shape):], original_shape)) if o == 1)
        result = np.sum(data, axis=axis + keepdims_axes, keepdims=True)
        return result.reshape(original_shape)

    @staticmethod
    def slice(data, start, end, axis):
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(start, end)
        return data[tuple(slices)]

    @staticmethod
    def pad(data, start, end, axis, original_size):
        # Pad with zeros where sliced
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[axis] = (start, original_size - end)
        return np.pad(data, pad_width)

    @staticmethod
    def squeeze(data, axis=None):
        return data.squeeze(axis)

    @staticmethod
    def unsqueeze(data, axis):
        return np.expand_dims(data, axis)

    @staticmethod
    def flatten(data):
        return data.flatten()

    @staticmethod
    def flatten_inverse(data, original_shape):
        return data.reshape(original_shape)
