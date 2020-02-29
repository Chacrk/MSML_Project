import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn
from torch.nn.functional import pad
from torch.nn.modules.utils import _pair

class WRN(nn.Module):
    def __init__(self, args, way):
        super(WRN, self).__init__()
        # init params
        self.channels_queue = [16, ]
        channels_queue_basic = [16, 32, 64]

        # trainable params
        self.weights = nn.ParameterDict()
        self.weights_br = nn.ParameterDict()

        # hyper params
        self.way = way
        self.num_blocks = args.num_blocks
        self.WRN_K = args.WRN_K
        self.dropout_p = args.dropout_p
        self.channels_queue.extend([int(x * self.WRN_K) for x in channels_queue_basic])

        self.use_conv_bias_shortcut = False
        self.use_conv_bias_normal = False

        # first part
        self.weights['first_conv0/filter'] = nn.Parameter(torch.ones([self.channels_queue[0], 3, 3, 3]), True)
        self.weights['first_conv0/bias'] = nn.Parameter(torch.zeros(self.channels_queue[0]), True)
        self.weights['first_norm0/weight'] = nn.Parameter(torch.ones(self.channels_queue[0]), True)
        self.weights['first_norm0/bias'] = nn.Parameter(torch.zeros(self.channels_queue[0]), True)
        self.weights_br['first_norm0/mean'] = nn.Parameter(torch.zeros(self.channels_queue[0]), False)
        self.weights_br['first_norm0/var'] = nn.Parameter(torch.zeros(self.channels_queue[0]), False)

        # blocks
        for i in range(len(self.channels_queue) - 1):
            self.construct_block_weights(i + 1, in_=self.channels_queue[i], out_=self.channels_queue[i + 1])

        # fc
        self.weights['fc/weight0'] = nn.Parameter(torch.ones([1000, 640]), True)
        self.weights['fc/bias0'] = nn.Parameter(torch.zeros(1000), True)
        self.weights['fc/weight1'] = nn.Parameter(torch.ones([self.way, 1000]), True)
        self.weights['fc/bias1'] = nn.Parameter(torch.zeros(self.way), True)

        # init params
        for s_ in self.named_parameters():
            if 'filter' in s_[0]:
                nn.init.kaiming_normal_(s_[1], mode='fan_out', nonlinearity='relu')
            if 'fc/weight' in s_[0]:
                nn.init.orthogonal_(s_[1])

    def construct_block_weights(self, b, in_, out_):
        # expand
        self.weights['b{}_0/conv0/filter'.format(b)] = nn.Parameter(torch.ones([out_, in_, 3, 3]), True)
        if self.use_conv_bias_normal:
            self.weights['b{}_0/conv0/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)
        self.weights['b{}_0/norm0/weight'.format(b)] = nn.Parameter(torch.ones(out_), True)
        self.weights['b{}_0/norm0/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)
        self.weights_br['b{}_0/norm0/mean'.format(b)] = nn.Parameter(torch.zeros(out_), False)
        self.weights_br['b{}_0/norm0/var'.format(b)] = nn.Parameter(torch.zeros(out_), False)

        self.weights['b{}_0/conv1/filter'.format(b)] = nn.Parameter(torch.ones([out_, out_, 3, 3]), True)
        if self.use_conv_bias_normal:
            self.weights['b{}_0/conv1/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)
        self.weights['b{}_0/norm1/weight'.format(b)] = nn.Parameter(torch.ones(out_), True)
        self.weights['b{}_0/norm1/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)
        self.weights_br['b{}_0/norm1/mean'.format(b)] = nn.Parameter(torch.zeros(out_), False)
        self.weights_br['b{}_0/norm1/var'.format(b)] = nn.Parameter(torch.zeros(out_), False)

        self.weights['b{}_0/conv2/filter'.format(b)] = nn.Parameter(torch.ones([out_, in_, 1, 1]), True)  # shortcut
        if self.use_conv_bias_shortcut:
            self.weights['b{}_0/conv2/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)  # shortcut's bias
        self.weights['b{}_0/norm2/weight'.format(b)] = nn.Parameter(torch.ones(out_), True)
        self.weights['b{}_0/norm2/bias'.format(b)] = nn.Parameter(torch.zeros(out_), True)
        self.weights_br['b{}_0/norm2/mean'.format(b)] = nn.Parameter(torch.zeros(out_), False)
        self.weights_br['b{}_0/norm2/var'.format(b)] = nn.Parameter(torch.zeros(out_), False)

        for i in range(1, self.num_blocks):
            self.weights['b{}_{}/conv0/filter'.format(b, i)] = nn.Parameter(torch.ones([out_, out_, 3, 3]), True)
            if self.use_conv_bias_normal:
                self.weights['b{}_{}/conv0/bias'.format(b, i)] = nn.Parameter(torch.zeros(out_), True)
            self.weights['b{}_{}/norm0/weight'.format(b, i)] = nn.Parameter(torch.ones(out_), True)
            self.weights['b{}_{}/norm0/bias'.format(b, i)] = nn.Parameter(torch.zeros(out_), True)
            self.weights_br['b{}_{}/norm0/mean'.format(b, i)] = nn.Parameter(torch.zeros(out_), False)
            self.weights_br['b{}_{}/norm0/var'.format(b, i)] = nn.Parameter(torch.zeros(out_), False)

            self.weights['b{}_{}/conv1/filter'.format(b, i)] = nn.Parameter(torch.ones([out_, out_, 3, 3]), True)
            if self.use_conv_bias_normal:
                self.weights['b{}_{}/conv1/bias'.format(b, i)] = nn.Parameter(torch.zeros(out_), True)
            self.weights['b{}_{}/norm1/weight'.format(b, i)] = nn.Parameter(torch.ones(out_), True)
            self.weights['b{}_{}/norm1/bias'.format(b, i)] = nn.Parameter(torch.zeros(out_), True)
            self.weights_br['b{}_{}/norm1/mean'.format(b, i)] = nn.Parameter(torch.zeros(out_), False)
            self.weights_br['b{}_{}/norm1/var'.format(b, i)] = nn.Parameter(torch.zeros(out_), False)

    # tensorflow style conv -'SAME' or 'VALID'
    def conv2d(self, input_, weight, bias=None, stride=1, padding='SAME', dilation=1, groups=1):
        stride = _pair(stride)
        dilation = _pair(dilation)

        def check_format(*argv):
            argv_format = []
            for i in range(len(argv)):
                if type(argv[i]) is int:
                    argv_format.append((argv[i], argv[i]))
                elif hasattr(argv[i], "__getitem__"):
                    argv_format.append(tuple(argv[i]))
                else:
                    raise TypeError('all input should be int or list-type, now is {}'.format(argv[i]))
            return argv_format
        stride, dilation = check_format(stride, dilation)

        if padding == 'SAME':
            padding = 0
            input_rows = input_.size(2)
            filter_rows = weight.size(2)
            out_rows = (input_rows + stride[0] - 1) // stride[0]
            padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
            rows_odd = padding_rows % 2

            input_cols = input_.size(3)
            filter_cols = weight.size(3)
            out_cols = (input_cols + stride[1] - 1) // stride[1]
            padding_cols = max(0, (out_cols - 1) * stride[1] + (filter_cols - 1) * dilation[1] + 1 - input_cols)
            cols_odd = padding_cols % 2
            input_ = pad(input_, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                                  padding_rows // 2, padding_rows // 2 + int(rows_odd)])
        elif padding == 'VALID':
            padding = 0
        elif type(padding) != int:
            raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))
        return F.conv2d(input_, weight, bias, stride, padding=padding, dilation=dilation, groups=groups)

    def swish(self, x):
        return torch.sigmoid(x) * x

    def conv_norm_activation(self, input_, weights, bn_training=True):
        x = F.conv2d(input_, weight=weights['first_conv0/filter'], bias=weights['first_conv0/bias'],
                     stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['first_norm0/mean'],
                         running_var=self.weights_br['first_norm0/var'], weight=weights['first_norm0/weight'],
                         bias=weights['first_norm0/bias'], training=bn_training, momentum=0.1)
        x = F.relu(x, inplace=True)
        return x

    def expand_conv(self, input_, weights, b, bn_training, use_dropout):
        shortcut_input = input_
        if self.use_conv_bias_normal:
            x = F.conv2d(input=input_, weight=weights['b{}_0/conv0/filter'.format(b)],
                         bias=weights['b{}_0/conv0/bias'.format(b)], stride=2, padding=1)
        else:
            x = F.conv2d(input=input_, weight=weights['b{}_0/conv0/filter'.format(b)],
                         bias=None, stride=2, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_0/norm0/mean'.format(b)],
                         running_var=self.weights_br['b{}_0/norm0/var'.format(b)],
                         weight=weights['b{}_0/norm0/weight'.format(b)],
                         bias=weights['b{}_0/norm0/bias'.format(b)], training=bn_training, momentum=0.1)
        x = F.relu(x, inplace=True)

        if use_dropout: x = F.dropout(input=x, p=self.dropout_p)

        if self.use_conv_bias_normal:
            x = F.conv2d(input=x, weight=weights['b{}_0/conv1/filter'.format(b)],
                         bias=weights['b{}_0/conv1/bias'.format(b)], stride=1, padding=1)
        else:
            x = F.conv2d(input=x, weight=weights['b{}_0/conv1/filter'.format(b)],
                         bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_0/norm1/mean'.format(b)],
                         running_var=self.weights_br['b{}_0/norm1/var'.format(b)],
                         weight=weights['b{}_0/norm1/weight'.format(b)],
                         bias=weights['b{}_0/norm1/bias'.format(b)], training=bn_training, momentum=0.1)

        if self.use_conv_bias_shortcut:
            shortcut = F.conv2d(input=shortcut_input, weight=weights['b{}_0/conv2/filter'.format(b)],
                                bias=weights['b{}_0/conv2/bias'.format(b)], stride=2)
        else:
            shortcut = F.conv2d(input=shortcut_input, weight=weights['b{}_0/conv2/filter'.format(b)],
                                bias=None, stride=2)
        shortcut = F.batch_norm(input=shortcut, running_mean=self.weights_br['b{}_0/norm2/mean'.format(b)],
                                running_var=self.weights_br['b{}_0/norm2/var'.format(b)],
                                weight=weights['b{}_0/norm2/weight'.format(b)],
                                bias=weights['b{}_0/norm2/bias'.format(b)], training=bn_training, momentum=0.1)
        x += shortcut
        x = F.relu(x, inplace=True)
        return x

    def conv_block(self, input_, weights, b, i, bn_training, use_dropout):
        shortcut_input = input_
        if self.use_conv_bias_normal:
            x = F.conv2d(input=input_, weight=weights['b{}_{}/conv0/filter'.format(b, i)],
                         bias=weights['b{}_{}/conv0/bias'.format(b, i)], stride=1, padding=1)
        else:
            x = F.conv2d(input=input_, weight=weights['b{}_{}/conv0/filter'.format(b, i)],
                         bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_{}/norm0/mean'.format(b, i)],
                         running_var=self.weights_br['b{}_{}/norm0/var'.format(b, i)],
                         weight=weights['b{}_{}/norm0/weight'.format(b, i)],
                         bias=weights['b{}_{}/norm0/bias'.format(b, i)], training=bn_training, momentum=0.1)
        x = F.relu(x, inplace=True)

        if use_dropout: x = F.dropout(input=x, p=self.dropout_p)

        if self.use_conv_bias_normal:
            x = F.conv2d(input=x, weight=weights['b{}_{}/conv1/filter'.format(b, i)],
                         bias=weights['b{}_{}/conv1/bias'.format(b, i)], stride=1, padding=1)
        else:
            x = F.conv2d(input=x, weight=weights['b{}_{}/conv1/filter'.format(b, i)], bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_{}/norm1/mean'.format(b, i)],
                         running_var=self.weights_br['b{}_{}/norm1/var'.format(b, i)],
                         weight=weights['b{}_{}/norm1/weight'.format(b, i)],
                         bias=weights['b{}_{}/norm1/bias'.format(b, i)], training=bn_training, momentum=0.1)
        x += shortcut_input
        x = F.relu(x, inplace=True)
        return x

    def fc_forward(self, input_, weights):
        hidden = F.linear(input=input_, weight=weights['fc/weight0'], bias=weights['fc/bias0'])
        hidden = F.relu(hidden, inplace=True)
        result = F.linear(input=hidden, weight=weights['fc/weight1'], bias=weights['fc/bias1'])
        return result

    def forward(self, x, weights=None, bn_training=True, use_dropout=True):
        if weights is None:
            weights = self.weights

        x = self.conv_norm_activation(input_=x, weights=weights, bn_training=bn_training)
        for block_index in range(1, 4):
            x = self.expand_conv(input_=x, weights=weights, b=block_index, bn_training=bn_training,
                                 use_dropout=use_dropout)
            for index_in_block in range(1, self.num_blocks):
                x = self.conv_block(input_=x, weights=weights, b=block_index, i=index_in_block,
                                    bn_training=bn_training, use_dropout=use_dropout)

        x = nn.AvgPool2d(10, stride=1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc_forward(input_=x, weights=weights)
        return x

    def zero_grad(self, weights=None):
        with torch.no_grad():
            if weights is None:
                for p in self.weights.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in weights.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.weights.values()

    def named_parameters(self):
        return self.weights.items()

    def dict_parameters(self):
        return self.weights

    def parameters_by_names(self, names):
        tmp = list(filter(lambda x: x[0] in names, self.named_parameters()))
        return [x[1] for x in tmp]

    def parameters_inner_item(self, names):
        tmp = list(filter(lambda x: x[0] in names, self.named_parameters()))
        return tmp

















