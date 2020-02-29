import math
import torch
import os
import numpy as np
from torch.nn import functional as F
from copy import deepcopy
from net_meta import WRN

class MSML(torch.nn.Module):
    def __init__(self, args):
        super(MSML, self).__init__()
        # args
        self.way = args.way
        self.k_shot = args.k_shot
        self.k_query = args.k_query
        self.batch_size = args.batch_size
        self.num_inner_updates = args.num_inner_updates
        self.num_inner_updates_test = args.num_inner_updates_test
        self.num_blocks = args.num_blocks
        self.WRN_K = args.WRN_K
        self.gamma = args.gamma
        self.lr_step_size = args.lr_step_size
        self.lr_min = args.lr_min
        self.epoch_index = args.epoch_index

        # create meta-learner
        self.net = WRN(args=args, way=self.way)
        self.all_param_keys = list(self.net.dict_parameters().keys())
        self.keys_inner = list(filter(lambda x: 'fc' in x, self.all_param_keys))

        # assign
        pretrain_weight = '../weights/WRN_K{}_ceb{}_index{}.data'.format(self.WRN_K, self.num_blocks, self.epoch_index)
        if os.path.exists(pretrain_weight):
            print('>> Pretrained weight exists')
        else:
            raise Exception('>> No Pretrained Data')
        print('>> Assigning pretrain weight')

        pretrain_dict = torch.load(pretrain_weight)
        model_dict = self.net.state_dict()
        pretrain_dict_match = {key : value for key, value in pretrain_dict.items() if (key in model_dict)}
        model_dict.update(pretrain_dict_match)
        self.net.load_state_dict(model_dict)
        self.keys_outer_ = list(filter(lambda x: 'bias' in x or 'norm' in x or '2/fil' in x, self.all_param_keys))
        self.keys_outer = list(filter(lambda x: 'fc' not in x, self.keys_outer_))

        # optimizer
        self.optim = torch.optim.Adam([{'params': self.net.parameters_by_names(self.keys_outer), 'lr': args.lr_outer},])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.lr_step_size, gamma=args.gamma)
        self.lr_fc = {'fc/weight': args.lr_outer_fc, 'fc/bias': args.lr_outer_fc}
        self.round = 0

    def forward(self, inputa, labela, inputb, labelb):
        train_loss_post = 0.0
        corrects = 0

        # ckps
        ckp_rates = {20: 0.1, 50: 0.2}
        ckp_losses = 0.0

        for b in range(self.batch_size):
            output_pre = self.net(inputa[b])
            loss_pre = F.cross_entropy(output_pre, labela[b])
            grads = torch.autograd.grad(loss_pre, self.net.parameters_by_names(self.keys_inner))
            grad_fc_w = grads[0] - torch.matmul(torch.matmul(self.net.dict_parameters()['fc/weight'],
                                                             grads[0].t()), self.net.dict_parameters()['fc/weight'])
            fast_weight_value = list(map(lambda p: p[1][1] - self.net.task_lr[p[1][0]] * p[0],
                                         zip([grad_fc_w, grads[1]], self.net.parameters_inner_item(self.keys_inner))))
            fast_weight = dict(zip(self.keys_inner, fast_weight_value))

            for k in range(1, self.num_inner_updates):
                output_pre = self.net(inputa[b], weights=fast_weight)
                loss_pre = F.cross_entropy(output_pre, labela[b])
                grads = torch.autograd.grad(loss_pre, list(fast_weight.values()))
                lr_scale = math.cos(math.pi * 0.4 * (k / self.num_inner_updates))
                grad_fc_w = grads[0] - torch.matmul(torch.matmul(fast_weight['fc/weight'], grads[0].t()),
                                                    fast_weight['fc/weight'])
                fast_weight_value = list(map(lambda p: p[1][1] - lr_scale * self.net.task_lr[p[1][0]] * p[0],
                                             zip([grad_fc_w, grads[1]], fast_weight.items())))
                fast_weight = dict(zip(self.keys_inner, fast_weight_value))

                if k in ckp_rates.keys():
                    output_post = self.net(inputb[b], weights=fast_weight)
                    ckp_losses += ckp_rates[k] * F.cross_entropy(output_post, labelb[b])

                if k == self.num_inner_updates - 1:
                    output_post = self.net(inputb[b], weights=fast_weight)
                    loss_post = F.cross_entropy(output_post, labelb[b])
                    train_loss_post += loss_post
                    with torch.no_grad():
                        predict_post = F.softmax(output_post, dim=1).argmax(dim=1)
                        corrects += torch.eq(predict_post, labelb[b]).sum().item()

        # optim
        loss_final = ((1.0 - np.sum(list(ckp_rates.values()))) * train_loss_post + ckp_losses) / self.batch_size
        grad_loss = torch.autograd.grad(loss_final, self.net.parameters_by_names(self.keys_inner), retain_graph=True)
        grad_fc_w = grad_loss[0] - torch.matmul(
            torch.matmul(self.net.dict_parameters()['fc/weight'], grad_loss[0].t()),
            self.net.dict_parameters()['fc/weight'])
        self.round += 1
        if self.round % self.lr_step_size == 0:
            for key_ in self.lr_fc.keys():
                if self.lr_fc[key_] > self.lr_min:
                    self.lr_fc[key_] *= self.gamma

        fast_weight_value = list(map(lambda p: p[1][1] - self.lr_fc[p[1][0]] * p[0],
                                     zip([grad_fc_w, grad_loss[1]], self.net.parameters_inner_item(self.keys_inner))))
        self.net.dict_parameters()['fc/weight'].data = fast_weight_value[0].data
        self.net.dict_parameters()['fc/bias'].data = fast_weight_value[1].data

        self.optim.zero_grad()
        loss_final.backward()
        self.optim.step()
        self.lr_scheduler.step()

        acc = corrects / (labelb.size(1) * self.batch_size)
        train_loss_post = train_loss_post.item()
        return acc, train_loss_post

    def forward_test(self, inputa, labela, inputb, labelb):
        _net = deepcopy(self.net)
        dropout = False
        corrects = 0
        loss_post = 0.0

        output_pre = _net(inputa, use_dropout=dropout)
        loss_pre = F.cross_entropy(output_pre, labela)
        grads = torch.autograd.grad(loss_pre, _net.parameters_by_names(self.keys_inner))
        grad_fc_w = grads[0] - torch.matmul(torch.matmul(_net.dict_parameters()['fc/weight'],
                                                         grads[0].t()), _net.dict_parameters()['fc/weight'])
        fast_weight_value = list(map(lambda p: p[1][1] - _net.task_lr[p[1][0]] * p[0],
                                     zip([grad_fc_w, grads[1]], _net.parameters_inner_item(self.keys_inner))))
        fast_weight = dict(zip(self.keys_inner, fast_weight_value))

        for k in range(1, self.num_inner_updates_test):
            output_pre = _net(inputa, weights=fast_weight, use_dropout=dropout)
            loss_pre = F.cross_entropy(output_pre, labela)
            grads = torch.autograd.grad(loss_pre, list(fast_weight.values()))
            lr_scale = math.cos(math.pi * 0.4 * (k / self.num_inner_updates_test))
            grad_fc_w = grads[0] - torch.matmul(torch.matmul(fast_weight['fc/weight'], grads[0].t()),
                                                fast_weight['fc/weight'])
            fast_weight_value = list(map(lambda p: p[1][1] - lr_scale * _net.task_lr[p[1][0]] * p[0],
                                         zip([grad_fc_w, grads[1]], fast_weight.items())))
            fast_weight = dict(zip(self.keys_inner, fast_weight_value))

            if k == self.num_inner_updates_test - 1:
                with torch.no_grad():
                    output_post = _net(inputb, weights=fast_weight, use_dropout=dropout)
                    loss_post += F.cross_entropy(output_post, labelb).item()
                    predict_post = F.softmax(output_post, dim=1).argmax(dim=1)
                    corrects += torch.eq(predict_post, labelb).sum().item()

        acc = corrects / inputb.size(0)
        del _net
        return acc, loss_post

    def save_model(self):
        torch.save(dict(self.net.state_dict()), 'weights/WRN_K{}_ceb{}.da'.format(self.WRN_K, self.num_blocks))
















