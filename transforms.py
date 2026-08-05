import numpy as np
from enum import Enum
from opsenum import Ops

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
    def reshape(data, shape, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            return data.reshape(original_shape)
        return data.reshape(shape)

    @staticmethod
    def transpose(data, axes=None, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            if axes is not None:
                return np.transpose(data, axes=np.argsort(axes))
            else:
                return np.transpose(data)
        return np.transpose(data, axes=axes)

    @staticmethod
    def _broadcastedAxes(current_shape, original_shape):
        orig_shape = list(original_shape)

        if len(orig_shape) < len(current_shape):
            orig_shape = [1] * (len(current_shape) - len(orig_shape)) + orig_shape

        return tuple(i for i in range(len(current_shape)) if current_shape[i] != orig_shape[i])

    @staticmethod
    def broadcast_to(data, shape=None, inverse=False, axes=None, keepdims=False, original_shape=None, opposite_shape=None, ops=None):
        if not inverse and shape is None:
            if opposite_shape is None:
                raise ValueError(
                    "broadcast_to: no explicit shape given and no opposite_shape to resolve against"
                )
            shape, _ = TransformFunctions.resolveShape(original_shape, opposite_shape, ops=ops)

        if inverse:
            if axes is None:
                axes = TransformFunctions._broadcastedAxes(data.shape, original_shape)

            return TransformFunctions.collapse(data, axes, keepdims=keepdims, original_shape=original_shape)
        return np.broadcast_to(data, shape)

    @staticmethod
    def collapse(data, axes, keepdims=True, inverse=False, shape=None, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            return np.broadcast_to(data, original_shape)
        return np.sum(data, axis=tuple(axes), keepdims=keepdims)

    @staticmethod
    def slice(data, start, end, axis, inverse=False, original_size=None, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            pad_width = [(0, 0)] * len(data.shape)
            pad_width[axis] = (start, original_size - end)
            return np.pad(data, pad_width)
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(start, end)
        return data[tuple(slices)]

    @staticmethod
    def pad(data, start, end, axis, original_size, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(start, end)
            return data[tuple(slices)]
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[axis] = (start, original_size - end)
        return np.pad(data, pad_width)

    @staticmethod
    def squeeze(data, axis=None, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            return np.expand_dims(data, axis)
        return data.squeeze(axis)

    @staticmethod
    def unsqueeze(data, axis, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            return data.squeeze(axis)
        return np.expand_dims(data, axis)

    @staticmethod
    def flatten(data, inverse=False, original_shape=None, ops=None, opposite_shape=None):
        if inverse:
            return data.reshape(original_shape)
        return data.flatten()

    @staticmethod
    def _resolveAxes(shape_a, shape_b):
        a = list(shape_a)
        b = list(shape_b)

        if len(a) < len(b):
            a = [1] * (len(b) - len(a)) + a
        elif len(b) < len(a):
            b = [1] * (len(a) - len(b)) + b

        resolved = []
        for axis, (da, db) in enumerate(zip(a, b)):
            if da == db:
                resolved.append(da)
            elif da == 1:
                resolved.append(db)
            elif db == 1:
                resolved.append(da)
            else:
                raise ValueError(
                    f"Shape mismatch at axis {axis}: {da} vs {db} "
                    f"(shapes {tuple(shape_a)} and {tuple(shape_b)} are not broadcast-compatible)"
                )

        return tuple(resolved)

    @staticmethod
    def resolveShape(shape_a, shape_b, ops=None):
        if ops == Ops.MATMUL:
            if len(shape_a) < 2 or len(shape_b) < 2:
                raise ValueError(
                    f"resolveShape matmul branch requires both operands to be rank >= 2 "
                    f"(got {shape_a} and {shape_b}); 1D-operand promotion is not handled here"
                )

            a_batch, a_trail = shape_a[:-2], shape_a[-2:]
            b_batch, b_trail = shape_b[:-2], shape_b[-2:]

            if a_trail[-1] != b_trail[0]:
                raise ValueError(
                    f"matmul contraction mismatch: {shape_a}'s last axis ({a_trail[-1]}) "
                    f"must equal {shape_b}'s second-to-last axis ({b_trail[0]})"
                )

            batch_target = TransformFunctions._resolveAxes(a_batch, b_batch)
            return batch_target + tuple(a_trail), batch_target + tuple(b_trail)

        target = TransformFunctions._resolveAxes(shape_a, shape_b)
        return target, target

    @staticmethod
    def apply(data, transform, inverse=False, original_shape=None, ops=None, opposite_shape=None, **params):
        if transform == Transforms.RESHAPE:
            return TransformFunctions.reshape(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.TRANSPOSE:
            return TransformFunctions.transpose(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.BROADCAST_TO:
            return TransformFunctions.broadcast_to(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.COLLAPSE:
            return TransformFunctions.collapse(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.SLICE:
            return TransformFunctions.slice(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.PAD:
            return TransformFunctions.pad(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.SQUEEZE:
            return TransformFunctions.squeeze(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.UNSQUEEZE:
            return TransformFunctions.unsqueeze(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        if transform == Transforms.FLATTEN:
            return TransformFunctions.flatten(data, inverse=inverse, original_shape=original_shape, ops=ops, opposite_shape=opposite_shape, **params)
        raise ValueError(f"Unsupported transform: {transform}")
