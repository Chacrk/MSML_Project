import torch
import os
import datetime
import numpy as np
import argparse as argparse_
from torch.backends.cudnn import descriptor
from torch.utils.data import DataLoader
from model import MSML
from data_provider_meta import DataProvider


def main(args):
    test_acc_list = []
    max_acc = 0.0
    max_step = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    if torch.cuda.is_available() is False:
        raise Exception('>> No CUDA device')

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # improve speed
    torch.backends.cudnn.deterministic = True

    # create model and data
    model = MSML(args).to(device)
    dp_train = DataProvider(path=args.mini_imagenet_path, data_aug=True, tasks=args.tasks_train, dataset_type='train',
                            way=args.way, k_shot=args.k_shot, k_query=args.k_query, img_size=args.img_size)
    dp_test = DataProvider(path=args.mini_imagenet_path, data_aug=False, tasks=args.tasks_test, dataset_type='test',
                           way=args.way, k_shot=args.k_shot, k_query=args.k_query, img_size=args.img_size)

    start_time = datetime.datetime.now()
    start_time_global = datetime.datetime.now()

    acc_train_mean, loss_train_mean = 0.0, 0.0
    step = 0
    for epoch in range(args.itrs // args.tasks_train):
        print('>> Creating dataloader for epoch')
        db_train = DataLoader(dp_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        for inputa, labela, inputb, labelb in db_train:
            inputa = inputa.to(device)
            labela = labela.to(device)
            inputb = inputb.to(device)
            labelb = labelb.to(device)

            acc_post, loss_post = model(inputa, labela, inputb, labelb)
            acc_train_mean += acc_post
            loss_train_mean += loss_post

            print('\rTraining Step=[{}/{}], Loss={:.3f}, Acc={:.3%}'.format(step, args.itrs, loss_post,
                                                                            acc_post), end='')
            if (step + 1) % args.print_interval_train == 0:
                end_time = datetime.datetime.now()
                acces_delta = (end_time - start_time).seconds
                start_time = datetime.datetime.now()

                print('\tAcc Mean={:.3%}\tLoss Mean={:.3f}\tTime Per Round={:.3f}s'.
                      format(acc_train_mean / args.print_interval_train,
                             loss_train_mean / args.print_interval_train,
                             acces_delta / args.print_interval_train))
                acc_train_mean, loss_train_mean = 0.0, 0.0
                print('------------------------------------------------')

            if (step + 1) % args.test_interval == 0:
                db_test = DataLoader(
                    dp_test, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
                acc_test_mean, loss_test_mean = 0.0, 0.0
                for index, (inputa_, labela_, inputb_, labelb_) in enumerate(db_test):
                    print(
                        '\rTesting Step=[{}/{}]'.format(index + 1, len(db_test)), end='')
                    inputa_ = inputa_.squeeze_().to(device)
                    labela_ = labela_.squeeze_().to(device)
                    inputb_ = inputb_.squeeze_().to(device)
                    labelb_ = labelb_.squeeze_().to(device)

                    acc_post, loss_post = model.forward_test(
                        inputa_, labela_, inputb_, labelb_)
                    acc_test_mean += acc_post
                    loss_test_mean += loss_post

                test_acc_list.append(acc_test_mean / args.tasks_test)
                if test_acc_list[-1] > max_acc:
                    print('>> Get new best test result')
                    max_acc = test_acc_list[-1]
                    max_step = step

                print('>> Test Acc={:.3%}\tTest Loss={:.3f}\tBest Acc Existed={:.3f}, in step={}'.
                      format(test_acc_list[-1], loss_test_mean / args.tasks_test, max_acc, max_step))

            step += 1

    end_time_global = datetime.datetime.now()
    acces_delta_global = (end_time_global - start_time_global).seconds
    print('>> Total Time={}s'.format(acces_delta_global))


if __name__ == '__main__':
    argparse = argparse_.ArgumentParser()
    # network
    argparse.add_argument('--WRN_K', type=int, help='', default=10)
    argparse.add_argument('--num_blocks', type=int, help='', default=3)
    argparse.add_argument('--dropout_p', type=float, help='', default=0.5)

    # meta-train
    argparse.add_argument('--mini_imagenet_path', type=str, help='', default='../data/miniimagenet')
    argparse.add_argument('--gpu_index', type=str, help='', default='0')
    argparse.add_argument('--itrs', type=int, help='', default=15000)
    argparse.add_argument('--tasks_train', type=int, help='', default=5000)
    argparse.add_argument('--tasks_test', type=int, help='', default=200)
    argparse.add_argument('--way', type=int, help='', default=5)
    argparse.add_argument('--k_shot', type=int, help='', default=1)
    argparse.add_argument('--k_query', type=int, help='', default=15)
    argparse.add_argument('--img_size', type=int, help='', default=80)
    argparse.add_argument('--print_interval_train', type=int, help='', default=100)
    argparse.add_argument('--test_interval', type=int, help='', default=400)
    argparse.add_argument('--epoch_index', type=int, help='', default=80)
    argparse.add_argument('--num_inner_updates', type=int, help='', default=100)
    argparse.add_argument('--num_inner_updates_test', type=int, help='', default=100)
    argparse.add_argument('--batch_size', type=int, help='', default=1)

    # optim
    argparse.add_argument('--lr_step_size', type=int, help='', default=2000) #
    argparse.add_argument('--gamma', type=float, help='', default=0.5)
    argparse.add_argument('--lr_min', type=float, help='', default=1e-6)
    argparse.add_argument('--lr_inner', type=float, help='', default=0.1)
    argparse.add_argument('--lr_outer', type=float, help='', default=1e-4)
    argparse.add_argument('--lr_outer_fc', type=float, help='', default=0.01)

    _args = argparse.parse_args()
    main(_args)
