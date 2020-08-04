""" SRU Implementation """
# flake8: noqa

import subprocess
import platform
import os
import re
import configargparse
import torch
import torch.nn as nn
from torch.autograd import Function
from collections import namedtuple


# For command-line option parsing
class CheckSRU(configargparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CheckSRU, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == 'SRU':
            check_sru_requirement(abort=True)
        # Check pass, set the args.
        setattr(namespace, self.dest, values)


# This SRU version implements its own cuda-level optimization,
# so it requires that:
# 1. `cupy` and `pynvrtc` python package installed.
# 2. pytorch is built with cuda support.
# 3. library path set: export LD_LIBRARY_PATH=<cuda lib path>.
def check_sru_requirement(abort=False):
    """
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """

    # Check 1.
    try:
        if platform.system() == 'Windows':
            subprocess.check_output('pip freeze | findstr cupy', shell=True)
            subprocess.check_output('pip freeze | findstr pynvrtc',
                                    shell=True)
        else:  # Unix-like systems
            subprocess.check_output('pip freeze | grep -w cupy', shell=True)
            subprocess.check_output('pip freeze | grep -w pynvrtc',
                                    shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError("Using SRU requires 'cupy' and 'pynvrtc' "
                             "python packages installed.")

    # Check 2.
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError("Using SRU requires pytorch built with cuda.")

    # Check 3.
    pattern = re.compile(".*cuda/lib.*")
    ld_path = os.getenv('LD_LIBRARY_PATH', "")
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError("Using SRU requires setting cuda lib path, e.g. "
                             "export LD_LIBRARY_PATH=/usr/local/cuda/lib64.")

    return True


SRU_CODE = """
extern "C" {
    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }
    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }
    __global__ void sru_fwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch,
                            const int d, const int k,
                            float * __restrict__ h,
                            float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;
        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = (activation_type == 1) ? tanh(cur) : (
                (activation_type == 2) ? reluf(cur) : cur
            );
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u;
            xp += ncols_x;
            cp += ncols;
            hp += ncols;
        }
    }
    __global__ void sru_bwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const float * __restrict__ c,
                            const float * __restrict__ grad_h,
                            const float * __restrict__ grad_last,
                            const int len,
                            const int batch, const int d, const int k,
                            float * __restrict__ grad_u,
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_bias,
                            float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);
        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
        const float *cp = c + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);
        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);
            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));
            const float gh_val = *ghp;
            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
            // grad wrt x
            *gxp = gh_val*(1-g2);
            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;
            // grad wrt u0
            *gup = gc*(1-g1);
            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;
            // grad wrt c'
            cur = gc*g1;
            up -= ncols_u;
            xp -= ncols_x;
            cp -= ncols;
            gup -= ncols_u;
            gxp -= ncols_x;
            ghp -= ncols;
        }
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
    __global__ void sru_bi_fwd(const float * __restrict__ u,
                               const float * __restrict__ x,
                               const float * __restrict__ bias,
                               const float * __restrict__ init,
                               const float * __restrict__ mask_h,
                               const int len, const int batch,
                               const int d, const int k,
                               float * __restrict__ h,
                               float * __restrict__ c,
                               const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        assert ((k == 3) || (k == 4));
        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const int d2 = d*2;
        const bool flip = (col%d2) >= d;
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;
        if (flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            hp += (len-1)*ncols;
        }
        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;
        for (int cnt = 0; cnt < len; ++cnt)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = (activation_type == 1) ? tanh(cur) : (
                (activation_type == 2) ? reluf(cur) : cur
            );
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u_;
            xp += ncols_x_;
            cp += ncols_;
            hp += ncols_;
        }
    }
    __global__ void sru_bi_bwd(const float * __restrict__ u,
                               const float * __restrict__ x,
                               const float * __restrict__ bias,
                               const float * __restrict__ init,
                               const float * __restrict__ mask_h,
                               const float * __restrict__ c,
                               const float * __restrict__ grad_h,
                               const float * __restrict__ grad_last,
                               const int len, const int batch,
                               const int d, const int k,
                               float * __restrict__ grad_u,
                               float * __restrict__ grad_x,
                               float * __restrict__ grad_bias,
                               float * __restrict__ grad_init,
                               int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        assert((k == 3) || (k == 4));
        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);
        const int d2 = d*2;
        const bool flip = ((col%d2) >= d);
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        const float *cp = c + col;
        const float *ghp = grad_h + col;
        float *gup = grad_u + (col*k);
        float *gxp = (k == 3) ? (grad_x + col) : (gup + 3);
        if (!flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            ghp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            gxp += (len-1)*ncols_x;
        }
        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;
        for (int cnt = 0; cnt < len; ++cnt)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);
            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (cnt<len-1)?(*(cp-ncols_)):(*(init+col));
            const float gh_val = *ghp;
            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
            // grad wrt x
            *gxp = gh_val*(1-g2);
            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;
            // grad wrt u0
            *gup = gc*(1-g1);
            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;
            // grad wrt c'
            cur = gc*g1;
            up -= ncols_u_;
            xp -= ncols_x_;
            cp -= ncols_;
            gup -= ncols_u_;
            gxp -= ncols_x_;
            ghp -= ncols_;
        }
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
}
"""
SRU_FWD_FUNC, SRU_BWD_FUNC = None, None
SRU_BiFWD_FUNC, SRU_BiBWD_FUNC = None, None
SRU_STREAM = None


def load_sru_mod():
    global SRU_FWD_FUNC, SRU_BWD_FUNC, SRU_BiFWD_FUNC, SRU_BiBWD_FUNC
    global SRU_STREAM
    if check_sru_requirement():
        from cupy.cuda import function
        from pynvrtc.compiler import Program

        # This sets up device to use.
        device = torch.device("cuda")
        tmp_ = torch.rand(1, 1).to(device)

        sru_prog = Program(SRU_CODE.encode('utf-8'),
                           'sru_prog.cu'.encode('utf-8'))
        sru_ptx = sru_prog.compile()
        sru_mod = function.Module()
        sru_mod.load(bytes(sru_ptx.encode()))

        SRU_FWD_FUNC = sru_mod.get_function('sru_fwd')
        SRU_BWD_FUNC = sru_mod.get_function('sru_bwd')
        SRU_BiFWD_FUNC = sru_mod.get_function('sru_bi_fwd')
        SRU_BiBWD_FUNC = sru_mod.get_function('sru_bi_bwd')

        stream = namedtuple('Stream', ['ptr'])
        SRU_STREAM = stream(ptr=torch.cuda.current_stream().cuda_stream)


class SRU_Compute(Function):

    def __init__(self, activation_type, d_out, bidirectional=False):
        SRU_Compute.maybe_load_sru_mod()
        super(SRU_Compute, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    @staticmethod
    def maybe_load_sru_mod():
        global SRU_FWD_FUNC

        if SRU_FWD_FUNC is None:
            load_sru_mod()

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)

        FUNC = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.activation_type],
            block=(thread_per_block, 1, 1), grid=(num_block, 1, 1),
            stream=SRU_STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            # -> directions x batch x dim
            last_hidden = torch.stack((c[-1, :, :d], c[0, :, d:]))
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        if self.bidirectional:
            grad_last = torch.cat((grad_last[0], grad_last[1]), 1)
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)

        # For DEBUG
        # size = (length, batch, x.size(-1)) \
        #         if x.dim() == 3 else (batch, x.size(-1))
        # grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type],
            block=(thread_per_block, 1, 1), grid=(num_block, 1, 1),
            stream=SRU_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                 bidirectional=False, use_tanh=1, use_relu=0):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_type = 2 if use_relu else (1 if use_tanh else 0)

        out_size = n_out * 2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out * k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir * 2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out * 4 if bidirectional else n_out * 2
        ))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out * 2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = input.data.new(
                batch, n_out if not self.bidirectional else n_out * 2
            ).zero_()

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_(
                (batch, n_out * bidir), self.dropout)
            h, c = SRU_Compute(self.activation_type, n_out,
                               self.bidirectional)(
                                   u, input, self.bias, c0, mask_h
            )
        else:
            h, c = SRU_Compute(self.activation_type, n_out,
                               self.bidirectional)(
                                   u, input, self.bias, c0
            )

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return w.new(*size).bernoulli_(1 - p).div_(1 - p)


class SRU(nn.Module):
    """
    Implementation of "Training RNNs as Fast as CNNs"
    :cite:`DBLP:journals/corr/abs-1709-02755`

    TODO: turn to pytorch's implementation when it is available.

    This implementation is adpoted from the author of the paper:
    https://github.com/taolei87/sru/blob/master/cuda_functional.py.

    Args:
      input_size (int): input to model
      hidden_size (int): hidden dimension
      num_layers (int): number of layers
      dropout (float): dropout to use (stacked)
      rnn_dropout (float): dropout to use (recurrent)
      bidirectional (bool): bidirectional
      use_tanh (bool): activation
      use_relu (bool): activation
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=2, dropout=0, rnn_dropout=0,
                 bidirectional=False, use_tanh=1, use_relu=0):
        # An entry check here, will catch on train side and translate side
        # if requirements are not satisfied.
        check_sru_requirement(abort=True)
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        for i in range(num_layers):
            sru_cell = SRUCell(
                n_in=self.n_in if i == 0 else self.out_size,
                n_out=self.n_out,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
            )
            self.rnn_lst.append(sru_cell)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3  # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = input.data.new(
                input.size(1), self.n_out * dir_
            ).zero_()
            c0 = [zeros for i in range(self.depth)]
        else:
            if isinstance(c0, tuple):
                # RNNDecoderState wraps hidden as a tuple.
                c0 = c0[0]
            assert c0.dim() == 3    # (depth, batch, dir_*n_out)
            c0 = [h.squeeze(0) for h in c0.chunk(self.depth, 0)]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if self.bidirectional:
            # fh -> (layers*directions) x batch x dim
            fh = torch.cat(lstc)
        else:
            fh = torch.stack(lstc)

        if return_hidden:
            return prevx, fh
        else:
            return prevx
