import sys

from torch.functional import Tensor
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import needle as ndl
from needle import backend_ndarray as nd

np.random.seed(1)

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    if len(kwargs) == 0:
      out = f(*args)
    else:
      out = f(*args, **kwargs)

    # special case of split
    is_splitted = False
    if isinstance(out, ndl.TensorTuple):
      c = np.random.randn(len(out), *out[0].shape)
      is_splitted = True
    else:
      c = np.random.randn(*out.shape)
    
    # special case of stack
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = []
    for a in args:
      if isinstance(a, ndl.autograd.Tensor):
        numerical_grad.append(np.zeros(a.shape))
    num_args = len(numerical_grad)
    # computes numerical gradient
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)

    if is_splitted:
      # special case of split
      c = [ndl.Tensor(c[i], device=args[0].device) for i in range(len(c))]
      backward_grad = out.op.gradient_as_tuple(c, out)
    else:
      backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    print("backward_grad", len(backward_grad))
    print("numerical_grad", len(numerical_grad))
    if is_stacked:
      backward_grad = backward_grad[0]
    for i in range(num_args):
        assert backward_grad[i].shape == numerical_grad[i].shape
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(num_args)
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

EWISE_OPS = {
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b
}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = [k for k in EWISE_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(fn, A, 5 + B)

SCALAR_OPS = {
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a - b,
    "subtract": lambda a, b: a - b,
}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]
@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(1).astype(np.float32).item()
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(fn, A, _B)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_power_op(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randint(10)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.power(_A, _B), ndl.ops.power_scalar(A, _B).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.power_scalar, A, _B)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_square_op(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.square(_A), ndl.ops.square(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.square, A)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sqrt_op(shape, device):
    _A = np.abs(10 * np.random.randn(*shape)).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.sqrt(_A), ndl.ops.sqrt(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.sqrt, A)


transpose_params = [
    {"shape": (10, 5), "axes": None},
    {"shape": (10, 5), "axes": (1, 0)},
    {"shape": (2, 3, 3, 8), "axes": None},
    {"shape": (3, 3, 6, 4), "axes": (1, 0)},
    {"shape": (2, 3, 3, 4), "axes": (2, 1)},
    {"shape": (3, 3, 6, 4), "axes": (3, 0)},
]
@pytest.mark.parametrize("params", transpose_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_transpose_op(params, device):
    shape, axes = params["shape"], params["axes"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    # Due to the different implementation details between Needle and Numpy
    # Here explicitly makes np_axes
    np_axes = [i for i in range(len(shape))]
    if axes is not None:
      np_axes[axes[0]], np_axes[axes[1]] = np_axes[axes[1]], np_axes[axes[0]]
    else:
      np_axes[-1], np_axes[-2] = np_axes[-2], np_axes[-1]

    np.testing.assert_allclose(np.transpose(_A, np_axes), ndl.ops.transpose(A, axes).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.transpose, A, axes=axes)


reshape_params = [
    {"shape": (10, 5), "new_shape": (2, 25)},
    {"shape": (10, 5), "new_shape": (1, 1, 50)},
    {"shape": (10, 5), "new_shape": (50, )},
    {"shape": (2, 3, 4, 5), "new_shape": (5, 4, 3, 2)},
    {"shape": (2, 3, 4, 5), "new_shape": (5, 2, 2, 3, 2)},
    {"shape": (2, 3, 4, 5), "new_shape": (120, )},
]
@pytest.mark.parametrize("params", reshape_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape_op(params, device):
    shape, new_shape = params["shape"], params["new_shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.reshape(_A, new_shape), ndl.ops.reshape(A, new_shape).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.reshape, A, shape=new_shape)


brodcast_to_params = [
    {"shape": (1, 10, 5), "new_shape": (3, 10, 5)},
    {"shape": (10, 1, 5), "new_shape": (10, 3, 5)},
    {"shape": (10, 5, 1), "new_shape": (10, 5, 3)},
]
@pytest.mark.parametrize("params", brodcast_to_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to_op(params, device):
    shape, new_shape = params["shape"], params["new_shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.broadcast_to(_A, new_shape), ndl.ops.broadcast_to(A, new_shape).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.broadcast_to, A, shape=new_shape)


summation_params = [
    {"shape": (2, 3, 4), "axes": None, "keepdims": True},
    {"shape": (2, 3, 4), "axes": None, "keepdims": False},
    {"shape": (2, 3, 4), "axes": (1, ), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (1, ), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (2, ), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (2, ), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (1, 2), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (1, 2), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (0, 1, 2), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (0, 1, 2), "keepdims": False},
]
@pytest.mark.parametrize("params", summation_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation_op(params, device):
    shape, axes, keepdims = params["shape"], params["axes"], params["keepdims"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.sum(_A, axes, keepdims=keepdims), ndl.ops.summation(A, axes, keepdims=keepdims).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.summation, A, axes=axes, keepdims=keepdims)


mean_params = [
    {"shape": (2, 3, 4), "axes": None, "keepdims": True},
    {"shape": (2, 3, 4), "axes": None, "keepdims": False},
    {"shape": (2, 3, 4), "axes": (1, ), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (1, ), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (2, ), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (2, ), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (1, 2), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (1, 2), "keepdims": False},
    {"shape": (2, 3, 4), "axes": (0, 1, 2), "keepdims": True},
    {"shape": (2, 3, 4), "axes": (0, 1, 2), "keepdims": False},
]
@pytest.mark.parametrize("params", mean_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_mean_op(params, device):
    shape, axes, keepdims = params["shape"], params["axes"], params["keepdims"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.mean(_A, axes, keepdims=keepdims), ndl.ops.mean(A, axes, keepdims=keepdims).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.mean, A, axes=axes, keepdims=keepdims)


matmul_params = [
    {"shape_1": (3, 4), "shape_2": (4, 3)},
    {"shape_1": (3, 5), "shape_2": (5, 1)},
]
@pytest.mark.parametrize("params", matmul_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_mean_op(params, device):
    shape_1, shape_2 = params["shape_1"], params["shape_2"]
    _A = np.random.randn(*shape_1).astype(np.float32)
    _B = np.random.randn(*shape_2).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(np.matmul(_A, _B), ndl.ops.matmul(A, B).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.matmul, A, B)


negate_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", negate_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_negate_op(params, device):
    shape = params["shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.negative(_A), ndl.ops.negate(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.negate, A)


log_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", log_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_log_op(params, device):
    shape = params["shape"]
    _A = 10 + np.abs(np.random.randn(*shape)).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.log(_A), ndl.ops.log(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.log, A)


exp_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", exp_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_exp_op(params, device):
    shape = params["shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.exp(_A), ndl.ops.exp(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.exp, A) 


relu_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", relu_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu_op(params, device):
    shape = params["shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    torch_relu = torch.nn.ReLU()
    np.testing.assert_allclose(torch_relu(torch.from_numpy(_A)).numpy(), ndl.ops.relu(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.relu, A) 


logsumexp_params = [
    {"shape": (3, ), "axes": (0, )},
    {"shape": (3, 4), "axes": (0, 1)},
    {"shape": (3, 4), "axes": (1, )},
    {"shape": (3, 4, 5), "axes": (0, 1)},
    {"shape": (3, 4, 5, 6), "axes": (0, 1, 2, 3)},
    {"shape": (3, 4, 5, 6), "axes": (0, 1, 2)},
    {"shape": (3, 4, 5, 6), "axes": (3, )},
]
@pytest.mark.parametrize("params", logsumexp_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp_op(params, device):
    shape, axes = params["shape"], params["axes"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(torch.logsumexp(torch.from_numpy(_A), dim=axes).numpy(), ndl.ops.logsumexp(A, axes).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.logsumexp, A, axes=axes) 
    

tanh_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", tanh_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tanh_op(params, device):
    shape = params["shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.tanh(_A), ndl.ops.tanh(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.tanh, A)


sigmoid_params = [
    {"shape": (3, )},
    {"shape": (3, 4)},
    {"shape": (3, 4, 5)},
    {"shape": (3, 4, 5, 6)},
]
@pytest.mark.parametrize("params", sigmoid_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sigmoid_op(params, device):
    shape = params["shape"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(_A)).numpy(), ndl.ops.sigmoid(A).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.sigmoid, A) 


stack_params = [
    {"shape": (10, 3),   "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", stack_params)
def test_stack_op(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_ndl += [ndl.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = ndl.stack(to_stack_ndl, axis=axis)
    assert np.linalg.norm(rhs.numpy() - lhs) < 1e-4
    backward_check(ndl.ops.stack, to_stack_ndl, axis=axis)


split_params = [
    {"shape": (10, ),    "n": 10, "axis": 0},
    {"shape": (10, 3),   "n": 10, "axis": 0},
    {"shape": (4, 5, 6), "n": 4,  "axis": 0},
    {"shape": (4, 5, 6), "n": 5,  "axis": 1},
    {"shape": (4, 5, 6), "n": 6,  "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", split_params)
def test_split_op(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    lhs = np.split(_A, n, axis=axis)
    rhs = ndl.split(A, axis=axis)
    assert len(lhs) == len(rhs)
    for i in range(len(lhs)):
      assert lhs[i].squeeze().shape == rhs[i].shape
      assert np.linalg.norm(rhs[i].numpy() - lhs[i].squeeze()) < 1e-4
    backward_check(ndl.ops.split, A, axis=axis)


flip_params = [
    {"shape": (3, ), "axes": None},
    {"shape": (3, ), "axes": (0, )},
    {"shape": (3, 4), "axes": None},
    {"shape": (3, 4), "axes": (0, )},
    {"shape": (3, 4), "axes": (0, 1)},
    {"shape": (3, 4, 5), "axes": None},
    {"shape": (3, 4, 5), "axes": (1, )},
    {"shape": (3, 4, 5), "axes": (0, 1)},
    {"shape": (3, 4, 5), "axes": (0, 1, 2)},
    {"shape": (3, 4, 5, 6), "axes": None},
    {"shape": (3, 4, 5, 6), "axes": (0, 1, 2, 3)},
]
@pytest.mark.parametrize("params", flip_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp_op(params, device):
    shape, axes = params["shape"], params["axes"]
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.flip(_A, axes), ndl.ops.flip(A, axes).numpy(), atol=1e-5, rtol=1e-5)
    backward_check(ndl.ops.flip, A, axes=axes) 
  
