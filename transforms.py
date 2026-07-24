import numpy as np
from enum import Enum

class Transforms(Enum):
    RESHAPE          = 'RESHAPE'
    TRANSPOSE        = 'TRANSPOSE'
    BROADCAST_TO     = 'BROADCAST_TO'
    COLLAPSE         = 'COLLAPSE'
    SLICE            = 'SLICE'
    PAD              = 'PAD'
    SQUEEZE          = 'SQUEEZE'
    UNSQUEEZE        = 'UNSQUEEZE'
    FLATTEN          = 'FLATTEN'


class TransformFunctions:

    @staticmethod
    def reshape(data, shape, inverse=False, original_shape=None):
        if inverse:
            return data.reshape(original_shape)
        return data.reshape(shape)

    @staticmethod
    def transpose(data, axes=None, inverse=False, original_shape=None):
        if inverse:
            if axes is not None:
                return np.transpose(data, axes=np.argsort(axes))
            else:
                return np.transpose(data)
        return np.transpose(data, axes=axes)

    @staticmethod
    def broadcast_to(data, shape, inverse=False, axes=None, keepdims=True, original_shape=None):
        if inverse:
            if axes is None:
                # Compute which axes were broadcasted by comparing current shape with original
                current_shape = data.shape
                orig_shape = list(original_shape)
                
                # Pad original shape with 1s on the left to match dimensions
                if len(orig_shape) < len(current_shape):
                    orig_shape = [1] * (len(current_shape) - len(orig_shape)) + orig_shape
                
                # Find axes where dimensions differ (these were broadcasted)
                axes = tuple([i for i in range(len(current_shape)) if current_shape[i] != orig_shape[i]])
            
            return TransformFunctions.collapse(data, axes, keepdims=keepdims, original_shape=original_shape)
        return np.broadcast_to(data, shape)

    @staticmethod
    def collapse(data, axes, keepdims=True, inverse=False, shape=None, original_shape=None):
        if inverse:
            return np.broadcast_to(data, original_shape)
        return np.sum(data, axis=tuple(axes), keepdims=keepdims)

    @staticmethod
    def slice(data, start, end, axis, inverse=False, original_size=None, original_shape=None):
        if inverse:
            pad_width = [(0, 0)] * len(data.shape)
            pad_width[axis] = (start, original_size - end)
            return np.pad(data, pad_width)
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(start, end)
        return data[tuple(slices)]

    @staticmethod
    def pad(data, start, end, axis, original_size, inverse=False, original_shape=None):
        if inverse:
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(start, end)
            return data[tuple(slices)]
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[axis] = (start, original_size - end)
        return np.pad(data, pad_width)

    @staticmethod
    def squeeze(data, axis=None, inverse=False, original_shape=None):
        if inverse:
            return np.expand_dims(data, axis)
        return data.squeeze(axis)

    @staticmethod
    def unsqueeze(data, axis, inverse=False, original_shape=None):
        if inverse:
            return data.squeeze(axis)
        return np.expand_dims(data, axis)

    @staticmethod
    def flatten(data, inverse=False, original_shape=None):
        if inverse:
            return data.reshape(original_shape)
        return data.flatten()

    @staticmethod
    def apply(data, transform, inverse=False, original_shape=None, **params):
        if transform == Transforms.RESHAPE:
            return TransformFunctions.reshape(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.TRANSPOSE:
            return TransformFunctions.transpose(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.BROADCAST_TO:
            return TransformFunctions.broadcast_to(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.COLLAPSE:
            return TransformFunctions.collapse(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.SLICE:
            return TransformFunctions.slice(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.PAD:
            return TransformFunctions.pad(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.SQUEEZE:
            return TransformFunctions.squeeze(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.UNSQUEEZE:
            return TransformFunctions.unsqueeze(data, inverse=inverse, original_shape=original_shape, **params)
        if transform == Transforms.FLATTEN:
            return TransformFunctions.flatten(data, inverse=inverse, original_shape=original_shape, **params)
        raise ValueError(f"Unsupported transform: {transform}")
