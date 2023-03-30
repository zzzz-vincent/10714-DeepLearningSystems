"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List, Union, Tuple
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


def square(a):
    return PowerScalar(2)(a)


def sqrt(a):
    return PowerScalar(0.5)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / b ** 2

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if axes is not None else (-2, -1)

    def compute(self, a):
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        #TODO: support auto completion and -1 as numpy
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        return out_grad.reshape(in_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        #TODO: support auto completion for a leading one dimision
        # eg. (3, 5) -> (k, 3, 5)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # for the case that shape is not changed, skip gradient
        size_diff = out_grad.size - node.inputs[0].size
        if size_diff == 0:
          return out_grad

        # Due to the definiation of Matmul:
        # an input (x, 1, y) can be broadcase to (n, x, k, y)
        # Therefore, for backwrods, we should firstly reduce dimension
        # and then reshape
        # eg (n, x, k, y) -> (x, y) -> (x, 1, y)
        axes = []
        in_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        axis_diff = len(out_shape) - len(in_shape)
        # find dimensions overlapped
        axes = [axis_diff + i for i in range(len(in_shape) - 1, -1, -1) \
                if in_shape[i] != out_shape[axis_diff + i]]
        # find leading new dimensions
        for i in range(axis_diff):
          axes.append(i)
        a = summation(out_grad, axes=tuple(axes)).reshape(in_shape)
        return a


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None,
                 keepdims: bool = False):
        if type(axes) is int:
          self.axes = tuple([axes])
        else:
          self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        a_sum = a.sum(self.axes, keepdims=self.keepdims)

        if self.keepdims:
          out_shape = set_axes_to_one(a.shape, self.axes)
          return a_sum.reshape(out_shape)
        else:
          return a_sum

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        grad_shape = set_axes_to_one(in_shape, self.axes)
        out = out_grad.reshape(grad_shape).broadcast_to(in_shape)
        return out


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class Mean(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None,
                keepdims: bool = False):
        if isinstance(axes, int):
          axes = axes,
        self.axes = axes
        self.keepdims = keepdims
    
    def compute(self, a):
        a_mean = array_api.mean(a, self.axes)
        
        if self.keepdims:
            out_shape = set_axes_to_one(a.shape, self.axes)
            return a_mean.reshape(out_shape)
        else:
            return a_mean

    def gradient(self, out_grad: Tensor, node: Tensor):
        out_size = out_grad.size
        in_size = node.inputs[0].size
        grad_size = in_size / out_size
        # computes the gradient of sum first
        sum_grad = Summation(self.axes, self.keepdims).gradient(out_grad, node)
        # computes the gradient of mean 
        return sum_grad / grad_size


def mean(a, axes=None, keepdims=False):
    return Mean(axes, keepdims)(a)



class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = out_grad @ transpose(rhs)
        rhs_grad = transpose(lhs) @ out_grad

        # Due to the definiation of Matmul:
        # for inputs with different dimensions, it will compute the last 2
        # eg. lhs (3, 4, 5, 6) matmul rhs (6, 7) -> (3, 4, 5, 7)
        # therefore, for backwords, it may need reduce dimension for one of
        # the input. 
        # For example, rhs_grad should be reduced from (3, 4, 6, 7) to (6, 7)
        if lhs_grad.shape != lhs.shape:
          axes = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
          lhs_grad = lhs_grad.sum(axes)
        if rhs_grad.shape != rhs.shape:
          axes = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
          rhs_grad = rhs_grad.sum(axes)

        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input = node.inputs[0].cached_data
        hot = input >= 0
        return out_grad * Tensor(hot, device=out_grad.device, dtype=out_grad.dtype)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_broadcase = max_z.broadcast_to(Z.shape)
        self.max_z_shape = max_z.shape
        exp_z = array_api.exp(Z - max_z_broadcase)
        sum_z = array_api.sum(exp_z, axis=self.axes)
        log_z = array_api.log(sum_z)
        return log_z + max_z.reshape(log_z.shape)

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        # uses cached_data from forwards and converts it to Tensor
        forward_res = Tensor(node.realize_cached_data(), device=Z.device)
        Z_shape = Z.shape
        forward_res = forward_res.reshape(self.max_z_shape).broadcast_to(Z_shape)
        return broadcast_to(out_grad.reshape(self.max_z_shape), Z_shape) * exp(Z - forward_res)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a);

    def gradient(self, out_grad, node):
        return out_grad * (1 - tanh(node.inputs[0])**2)


def tanh(a):
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a: NDArray):
        e = array_api.exp(-a)
        return e / (e + e**2)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        in_val = Tensor(node.realize_cached_data(), device=node.inputs[0].device)
        return out_grad * (in_val * (1 - in_val))

def sigmoid(a):
    return Sigmoid()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        new_shape = [len(args)] + list(args[0].shape)
        new_ndarray = NDArray.make(new_shape, device=args[0].device)
        slices = [slice(0, n, 1) for n in args[0].shape]
        for i, arr in enumerate(args):
          new_ndarray[tuple([slice(i, i+1, 1)] + slices)] = arr.compact()
        axes = [i + 1 for i in range(len(args[0].shape))]
        axes.insert(self.axis, 0)
        return new_ndarray.permute(axes).compact()
        

    def gradient(self, out_grad, node):
        a = split(out_grad, self.axis)
        return a


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        in_shape = A.shape
        out_shape = []
        slices = []

        for i, n in enumerate(in_shape):
          slices.append(slice(0, n, 1))
          if i != self.axis:
            out_shape.append(n)

        out = []
        for i in range(in_shape[self.axis]):
          slices[self.axis] = slice(i, i+1, 1)
          out.append(A[tuple(slices)].compact().reshape(out_shape))
        return tuple(out)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes=None):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        in_shape = a.shape
        out_shape = []
        pos_slices = []
        pre_slices = [slice(0, n, 1) for n in in_shape]
        
        for i, shape in enumerate(in_shape):
          if i in self.axes:
            dilation_shape = shape + self.dilation * shape
            out_shape.append(shape + self.dilation * shape)
            pos_slices.append(slice(0, dilation_shape, self.dilation + 1))
          else:
            out_shape.append(shape)
            pos_slices.append(slice(0, shape, 1))
        
        new_ndarray = NDArray.make(out_shape, device=a.device)
        new_ndarray.fill(0)
        new_ndarray[tuple(pos_slices)] = a[tuple(pre_slices)]
        return new_ndarray

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        in_shape = a.shape
        out_shape = []
        pre_slices = [slice(0, n, 1) for n in in_shape]
        for ax in self.axes:
          pre_slices[ax] = slice(0, a.shape[ax], self.dilation + 1)
        
        return a[tuple(pre_slices)]

        
    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
        self.B = None

    def compute(self, A, B):
        # A format as [BATCHES][HEIGHT][WIDTH][CHANNELS]
        # B format as [KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][OUT_CHANNELS]
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        new_H = (H-K) // self.stride + 1
        new_W = (W-K) // self.stride + 1

        A_ = A.compact().as_strided(shape=(N, new_H, new_W, K, K, C_in),
                                 strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact().reshape((N * new_H * new_W, inner_dim))

        B_ = B.compact().reshape((K*K*C_in, C_out))
        out = A_ @ B_
        a = out.reshape((N, new_H, new_W, C_out))
        # print("a", a)
        return a


    def gradient(self, out_grad, node):
        A, B = node.inputs
        # A(N, H, W, C_in)
        N, H, W, C_in = A.shape
        # B(K, K, C_in, C_out)
        K, _, _, C_out = B.shape
        padding = K - self.padding - 1
        if (self.stride > 1):
          out_grad = dilate(out_grad, (1, 2), self.stride-1)
        # out_grad(N, d_H, d_W, C_out)
        _, d_H, d_W, _ = out_grad.shape
        # flipped_B(K', K', C_out, C_in)
        flipped_B = transpose(flip(B, axes=(0, 1)), (2, 3)) 
        
        A_grad = conv(out_grad, flipped_B, 1, padding)

        # A_permute(C_in, H, W, N) 
        # (0, 1, 2, 3) -> (3, 1, 2, 0)
        A_permuted = transpose(A, (0, 3))
        # out_grad_permute(d_H, d_W, H, C_out)
        # (0, 1, 2, 3) -> (1, 2, 0, 3)
        out_grad_permuted = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # B_conv (C_in, K1, K2, C_out)
        B_conv = conv(A_permuted, out_grad_permuted, 1, self.padding)
        # B_grad (K1, K2, C_in, C_out)
        B_grad = transpose(transpose(B_conv, (0, 1)), (1, 2))

        return A_grad, B_grad

        
def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



def set_axes_to_one(in_shape: List[int], axes: Optional[tuple] = None):
    """
    Sets some dimensions of in_shape to ``one`` according to given axes.
  
    If the axes is None, sets all dimensions to ``one``.
    This function will not check if given axes is within index range or not.
    """
    if isinstance(in_shape, tuple):
        in_shape = list(in_shape)

    out_shape = in_shape.copy()

    if axes is None:
        out_shape[:] = [1] * len(in_shape)
    else:
        for ax in axes:
            out_shape[ax] = 1

    return out_shape
