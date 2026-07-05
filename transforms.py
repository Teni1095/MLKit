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
    def transpose(data, axes=None, inverse=False):
        if inverse:
            return np.transpose(data, axes=np.argsort(axes))
        return np.transpose(data, axes=axes)

    @staticmethod
    def broadcast_to(data, shape, inverse=False, axes=None, keepdims=True):
        if inverse:
            return TransformFunctions.collapse(data, axes, keepdims=keepdims)
        return np.broadcast_to(data, shape)

    @staticmethod
    def collapse(data, axes, keepdims=True, inverse=False, shape=None):
        if inverse:
            return np.broadcast_to(data, shape)
        return np.sum(data, axis=tuple(axes), keepdims=keepdims)

    @staticmethod
    def slice(data, start, end, axis, inverse=False, original_size=None):
        if inverse:
            pad_width = [(0, 0)] * len(data.shape)
            pad_width[axis] = (start, original_size - end)
            return np.pad(data, pad_width)
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(start, end)
        return data[tuple(slices)]

    @staticmethod
    def pad(data, start, end, axis, original_size, inverse=False):
        if inverse:
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(start, end)
            return data[tuple(slices)]
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[axis] = (start, original_size - end)
        return np.pad(data, pad_width)

    @staticmethod
    def squeeze(data, axis=None, inverse=False):
        if inverse:
            return np.expand_dims(data, axis)
        return data.squeeze(axis)

    @staticmethod
    def unsqueeze(data, axis, inverse=False):
        if inverse:
            return data.squeeze(axis)
        return np.expand_dims(data, axis)

    @staticmethod
    def flatten(data, inverse=False, original_shape=None):
        if inverse:
            return data.reshape(original_shape)
        return data.flatten()

    @staticmethod
    def apply(data, transform, inverse=False, **params):
        if transform == Transforms.RESHAPE:
            return TransformFunctions.reshape(data, inverse=inverse, **params)
        if transform == Transforms.TRANSPOSE:
            return TransformFunctions.transpose(data, inverse=inverse, **params)
        if transform == Transforms.BROADCAST_TO:
            return TransformFunctions.broadcast_to(data, inverse=inverse, **params)
        if transform == Transforms.COLLAPSE:
            return TransformFunctions.collapse(data, inverse=inverse, **params)
        if transform == Transforms.SLICE:
            return TransformFunctions.slice(data, inverse=inverse, **params)
        if transform == Transforms.PAD:
            return TransformFunctions.pad(data, inverse=inverse, **params)
        if transform == Transforms.SQUEEZE:
            return TransformFunctions.squeeze(data, inverse=inverse, **params)
        if transform == Transforms.UNSQUEEZE:
            return TransformFunctions.unsqueeze(data, inverse=inverse, **params)
        if transform == Transforms.FLATTEN:
            return TransformFunctions.flatten(data, inverse=inverse, **params)
        raise ValueError(f"Unsupported transform: {transform}")
