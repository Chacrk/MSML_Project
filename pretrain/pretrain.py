import torch
import torch.nn.functional as F
import os
import datetime
import numpy as np
import argparse
from torch.utils.data import DataLoader
from data_provider import DataProvider
from net import WRN

CLASSES_TRAIN = 64
CLASSES_VAL = 16
CLASSES_TEST = 20

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    if torch.cuda.is_available() is False:
        print('NO CUDA')
        return

    if os.path.exists('../weights') is False:
        os.mkdir('../weights')

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # improve speed
    torch.backends.cudnn.deterministic = True

    model = WRN(args, way=CLASSES_TRAIN + CLASSES_VAL).to(device)
    data_provider_train = DataProvider(root_path=args.mini_imagenet_path, dataset_type='train', img_size=args.img_size,
                                       data_aug=True, mode='train')
    data_provider_test  = DataProvider(root_path=args.mini_imagenet_path, dataset_type='train', img_size=args.img_size,
                                       data_aug=False, mode='test')
    dl_train = DataLoader(dataset=data_provider_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dl_test = DataLoader(dataset=data_provider_test, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # optimizer params
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=args.optim_momentum, nesterov=True,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    for epoch in range(1, args.num_epoch + 1):

        start_time = datetime.datetime.now()

        for i, (images, labels) in enumerate(dl_train):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            _, predict = torch.max(outputs.data, 1)
            total, correct = 0, 0
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\r>> GPU={}, Epoch=[{}/{}], Step=[{}/{}], Train Acc={:.3%}, Train Loss={:.3f}, lr={:.3f}'.format(
                args.gpu_index, epoch, args.num_epoch, i+1, len(dl_train), correct / total, loss.item(),
                optimizer.state_dict()['param_groups'][0]['lr']
            ), end='')

        end_time = datetime.datetime.now()
        access_delta = (end_time - start_time).seconds

        print('')
        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for i_, (images_, labels_) in enumerate(dl_test):
                print('\r\tTesting=[{}/{}]'.format(i_+1, len(dl_test)), end='')
                images_ = images_.to(device)
                labels_ = labels_.to(device)
                outputs = model(images_, bn_training=False, use_dropout=False)
                _, predict = torch.max(outputs.data, 1)
                total += labels_.size(0)
                correct += (predict == labels_).sum().item()
        print(', Test Acc={:.3%}, Time Per Round={:.3f}ms'.format(correct / total, (access_delta / 375) * 1000))
        model.train()

        if epoch % args.save_dict_interval == 0:
            torch.save(dict(model.state_dict()),
                       '../weights/WRN_K{}_ceb{}_index{}.data'.format(args.WRN_K, args.num_blocks, epoch))

        lr_scheduler.step()


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    # net
    argparse.add_argument('--WRN_K', type=int, help='', default=10)
    argparse.add_argument('--num_blocks', type=int, help='', default=3)
    argparse.add_argument('--dropout_p', type=float, help='', default=0.5)

    # optim
    argparse.add_argument('--lr_base', type=float, help='', default=0.1)
    argparse.add_argument('--optim_momentum', type=float, help='', default=0.9)
    argparse.add_argument('--weight_decay', type=float, help='', default=0.0005)
    argparse.add_argument('--lr_step_size', type=int, help='', default=30)
    argparse.add_argument('--lr_gamma', type=float, help='', default=0.2)

    # pretrain
    argparse.add_argument('--gpu_index', type=str, help='', default='0')
    argparse.add_argument('--mini_imagenet_path', type=str, help='dataset path', default='../data/miniimagenet')
    argparse.add_argument('--batch_size', type=int, help='', default=128)
    argparse.add_argument('--num_epoch', type=int, help='num of epoch', default=100)
    argparse.add_argument('--img_size', type=int, help='=images pixel', default=80)
    argparse.add_argument('--save_dict_interval', type=int, help='', default=10)

    _args = argparse.parse_args()
    start_time = datetime.datetime.now()
    main(_args)
    end_time = datetime.datetime.now()
    access_delta = (end_time - start_time).seconds
    print('>> Total time={}s'.format(access_delta))




















