"""  Weights normalization modules  """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    """ utility for retrieving polyak averaged params
        Update average
    """
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + '_avg')
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return v_avg


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    """ utility for retrieving polyak averaged params """
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(
            namespace, vn, training, polyak_decay))
    return vars


class WeightNormLinear(nn.Linear):
    """
    Implementation of "Weight Normalization: A Simple Reparameterization
    to Accelerate Training of Deep Neural Networks"
    :cite:`DBLP:journals/corr/SalimansK16`

    As a reparameterization method, weight normalization is same
    as BatchNormalization, but it doesn't depend on minibatch.

    NOTE: This is used nowhere in the code at this stage
          Vincent Nguyen 05/18/2018
    """

    def __init__(self, in_features, out_features,
                 init_scale=1., polyak_decay=0.9995):
        super(WeightNormLinear, self).__init__(
            in_features, out_features, bias=True)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_features))
        self.b = self.bias

        self.register_buffer(
            'V_avg', torch.zeros(out_features, in_features))
        self.register_buffer('g_avg', torch.zeros(out_features))
        self.register_buffer('b_avg', torch.zeros(out_features))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_features * in_features
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(
                self.V.data) * 0.05)
            # norm is out_features * 1
            v_norm = self.V.data / \
                self.V.data.norm(2, 1).expand_as(self.V.data)
            # batch_size * out_features
            x_init = F.linear(x, v_norm).data
            # out_features
            m_init, v_init = x_init.mean(0).squeeze(
                0), x_init.var(0).squeeze(0)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            x_init = scale_init.view(1, -1).expand_as(x_init) \
                * (x_init - m_init.view(1, -1).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(self, ['V', 'g', 'b'],
                                         self.training,
                                         polyak_decay=self.polyak_decay)
            # batch_size * out_features
            x = F.linear(x, v)
            scalar = g / torch.norm(v, 2, 1).squeeze(1)
            x = scalar.view(1, -1).expand_as(x) * x + \
                b.view(1, -1).expand_as(x)
            return x


class WeightNormConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, init_scale=1.,
                 polyak_decay=0.9995):
        super(WeightNormConv2d, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding,
                                               dilation, groups)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_channels, in_channels // groups, * kernel_size
            self.V.data.copy_(torch.randn(self.V.data.size()
                                          ).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.view(self.out_channels, -1)\
                .norm(2, 1).view(self.out_channels, *(
                    [1] * (len(self.kernel_size) + 1))).expand_as(self.V.data)
            x_init = F.conv2d(x, v_norm, None, self.stride,
                              self.padding, self.dilation, self.groups).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(
                self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(
                1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(
                x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(
                self, ['V', 'g', 'b'], self.training,
                polyak_decay=self.polyak_decay)

            scalar = torch.norm(v.view(self.out_channels, -1), 2, 1)
            if len(scalar.size()) == 2:
                scalar = g / scalar.squeeze(1)
            else:
                scalar = g / scalar

            w = scalar.view(self.out_channels, *
                            ([1] * (len(v.size()) - 1))).expand_as(v) * v

            x = F.conv2d(x, w, b, self.stride,
                         self.padding, self.dilation, self.groups)
            return x

# This is used nowhere in the code at the moment (Vincent Nguyen 05/18/2018)


class WeightNormConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, init_scale=1.,
                 polyak_decay=0.9995):
        super(WeightNormConvTranspose2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride,
            padding, output_padding,
            groups)
        # in_channels, out_channels, *kernel_size
        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # in_channels, out_channels, *kernel_size
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(
                self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.transpose(0, 1).contiguous() \
                .view(self.out_channels, -1).norm(2, 1).view(
                    self.in_channels, self.out_channels,
                    *([1] * len(self.kernel_size))).expand_as(self.V.data)
            x_init = F.conv_transpose2d(
                x, v_norm, None, self.stride,
                self.padding, self.output_padding, self.groups).data
            # self.out_channels, 1
            t_x_init = x_init.tranpose(0, 1).contiguous().view(
                self.out_channels, -1)
            # out_features
            m_init, v_init = t_x_init.mean(1).squeeze(
                1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))

            x_init = scale_init_shape.expand_as(x_init)\
                * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(
                self, ['V', 'g', 'b'], self.training,
                polyak_decay=self.polyak_decay)
            scalar = g / \
                torch.norm(v.transpose(0, 1).contiguous().view(
                    self.out_channels, -1), 2, 1).squeeze(1)
            w = scalar.view(self.in_channels, self.out_channels,
                            *([1] * (len(v.size()) - 2))).expand_as(v) * v

            x = F.conv_transpose2d(x, w, b, self.stride,
                                   self.padding, self.output_padding,
                                   self.groups)
            return x
