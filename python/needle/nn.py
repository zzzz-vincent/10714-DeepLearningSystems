"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose(), device=device, dtype=dtype)
        else:
          self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        y =  X @ self.weight
        if self.bias:
          y += self.bias.broadcast_to(y.shape)
        return y


class Flatten(Module):
    def forward(self, X):
        new_shape = [s for s in X.shape if s != 1]
        return X.reshape(new_shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
          x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        softmax = ops.summation(ops.logsumexp(logits, axes=1))
        z_y = ops.summation((logits * y_one_hot))
        return (softmax - z_y) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(1, dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = x.shape[0]
            mean = x.sum(axes=0) / batch_size
            x_mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - x_mean) ** 2).sum(axes=0) / batch_size
            x_var = var.reshape((1, self.dim)).broadcast_to(x.shape)
            self.running_mean = ((1 - self.momentum) *
                                 self.running_mean + self.momentum * mean).data
            self.running_var = ((1 - self.momentum) *
                                self.running_var + self.momentum * var).data
        else:
            x_mean = self.running_mean.reshape(
                (1, self.dim)).broadcast_to(x.shape)
            x_var = self.running_var.reshape(
                (1, self.dim)).broadcast_to(x.shape)
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * norm + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)


class BatchNorm2d(BatchNorm1d):
  def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__(dim=dim, eps=eps, momentum=momentum, device=device, dtype=dtype)
  def forward(self, x: Tensor):
        assert len(x.shape) == 4
        N, C, W, H = x.shape
        x = x.transpose((1, 2)).transpose(
          (2, 3)).reshape((N * W * H, C))
        out = super().forward(x).reshape((N, W, H, C)).transpose((2, 3)).transpose((1, 2))
        return out


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim), dtype=device)
        self.bias = Parameter(init.zeros(1, dim), dtype=device)
        
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        # mean and var needs to be reshaped since current operation will
        # not perform auto reshape as what PyTorch does
        mean = ops.summation(x, axes=1) / x.shape[1]
        mean_reshape = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        var = ops.summation(ops.power_scalar(x - mean_reshape, 2), axes=1) / x.shape[1]
        var_reshape = var.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        norm_y = (x - mean_reshape) / ops.power_scalar(var_reshape + self.eps, 0.5)
        return w * norm_y + b


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
          mask = Tensor(init.randb(*x.shape, p=1-self.p)) / (1 - self.p)
          return x * mask
        else:
          return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        self.weight = Parameter(init.kaiming_uniform(
            fan_in, fan_out, shape=weight_shape, device=device, dtype=dtype))
        if bias:
          interval = 1.0/(in_channels * kernel_size**2)**0.5
          self.bias = Parameter(init.rand(out_channels, low=-interval, high=interval, requires_grad=True, device=device, dtype=dtype))
        else:
          self.bias = Parameter(init.zeros(out_channels, device=device, dtype=dtype))


    def forward(self, x: Tensor) -> Tensor:
        # input NCHW -> NHWC
        x_permuted = x.transpose((1, 2)).transpose((2, 3))
        padding = (self.kernel_size - 1)//2
        out = ops.conv(x_permuted, self.weight, stride=self.stride, padding=padding)
        if self.bias:
          bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
          out = out + bias
        # output NHWC -> NCHW
        return out.transpose((2, 3)).transpose((1,2))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        k = 1 / hidden_size ** 0.5
        self.hidden_size = hidden_size

        self.W_ih = Parameter(init.rand(input_size, hidden_size,
            low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size,
            low=-k, high=k, device=device, dtype=dtype))

        if bias:
          self.bias = True
          self.bias_ih = Parameter(init.rand(hidden_size,
              low=-k, high=k, device=device, dtype=dtype))
          self.bias_hh = Parameter(init.rand(hidden_size,
              low=-k, high=k, device=device, dtype=dtype))
        else:
          self.bias = False

        if nonlinearity == "tanh":
          self.activation = ops.Tanh()
        elif nonlinearity == "relu":
          self.activation = ops.ReLU()
        else:
          raise Exception(f"{nonlinearity} layer is not implemented")

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs, input_size = X.shape
        if h is  None:
          h = init.zeros(bs, self.hidden_size,
                           device=X.device, dtype=X.dtype)
        logits = X @ self.W_ih + h @ self.W_hh
        if self.bias:
          logits += (self.bias_ih + self.bias_hh).reshape((1, self.hidden_size)).broadcast_to(logits.shape)

        return self.activation(logits)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.rnn_cells = []
        for i in range(num_layers):
          self.rnn_cells.append(RNNCell(
            input_size if i == 0 else hidden_size,
            hidden_size,
            bias=bias,
            nonlinearity=nonlinearity,
            device=device,
            dtype=dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h0 is None:
          h0 = init.zeros(
            self.num_layers,
            bs,
            self.hidden_size,
            device=X.device,
            dtype=X.dtype)

        X_split = ops.split(X, axis=0)
        h0_split = list(ops.split(h0, axis=0))
        out = []

        for seq in range(seq_len):
          x = X_split[seq]
          for layer in range(self.num_layers):
            x = self.rnn_cells[layer].forward(x, h0_split[layer])
            h0_split[layer] = x
          out.append(h0_split[-1])
          
        return ops.stack(out, axis=0), ops.stack(h0_split, axis=0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.bias = bias
        self.hidden_size = hidden_size

        k = 1 / hidden_size ** 0.5

        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size,
            low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size,
            low=-k, high=k, device=device, dtype=dtype))

        if self.bias:
          self.bias_ih = Parameter(init.rand(4 * hidden_size,
              low=-k, high=k, device=device, dtype=dtype))
          self.bias_hh = Parameter(init.rand(4 * hidden_size,
              low=-k, high=k, device=device, dtype=dtype))

        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs, input_size = X.shape
        if h is None:
          h = (init.zeros(
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
                requires_grad=True),
              init.zeros(
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
                requires_grad=True))
        h0, c0 = h
        val = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
          val += (self.bias_ih + self.bias_hh).reshape((1, 4 * self.hidden_size)).broadcast_to(val.shape)
        i, f, g, o = ops.split(val.reshape((bs, 4, self.hidden_size)), axis=1)
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)

        c_n = f * c0 + i * g
        h_n = o * self.tanh(c_n)
        return h_n, c_n


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.lstm_cells = []
        for i in range(num_layers):
          self.lstm_cells.append(LSTMCell(
            input_size if i == 0 else hidden_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h is None:
          h = (init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
                requires_grad=True),
              init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
                requires_grad=True))
        X_split = ops.split(X, axis=0)
        h0, c0 = h
        h0_split = list(ops.split(h0, axis=0))
        c0_split = list(ops.split(c0, axis=0))
        out = []
        for seq in range(seq_len):
          x = X_split[seq]
          for layer in range(self. num_layers):
            x, c0_n = self.lstm_cells[layer](x, h=(h0_split[layer], c0_split[layer]))
            h0_split[layer] = x
            c0_split[layer] = c0_n
          out.append(x)
        
        return ops.stack(out, axis=0), (ops.stack(h0_split, axis=0), ops.stack(c0_split, axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim,
            device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)
        out = one_hot.reshape((seq_len * bs, self.num_embeddings)) @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
