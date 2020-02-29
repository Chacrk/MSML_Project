import os
import torch
import numpy as np
import random
import csv
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class DataProvider(Dataset):
    def __init__(self, path, data_aug, tasks, dataset_type, way, k_shot, k_query, img_size):
        # path: '../data/miniimagenet'
        self.path = path
        self.total_batch_size = tasks
        self.way = way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_inputa = self.way * self.k_shot
        self.num_inputb = self.way * self.k_query
        self.img_size = img_size
        self.data_aug = data_aug
        self.dataset_type = dataset_type

        self.images_path = '{}/images'.format(path)
        filenames_by_label = self.make_filenames_by_label('{}/{}.csv'.format(path, dataset_type))

        self.filenames_by_index = []
        self.index_by_label = {}
        for i, (key_, values_) in enumerate(filenames_by_label.items()):
            self.filenames_by_index.append(values_)
            self.index_by_label[key_] = i
        self.total_num_classes = len(self.filenames_by_index)
        self.create_batch(self.total_batch_size)

    def make_filenames_by_label(self, csv_path):
        filenames_by_label = {}
        with open(csv_path) as csv_:
            csv_reader = csv.reader(csv_)
            next(csv_reader, None)
            for i, row_item in enumerate(csv_reader):
                filename_tmp = row_item[0]
                label_tmp = row_item[1]
                if label_tmp not in filenames_by_label.keys():
                    filenames_by_label[label_tmp] = []
                filenames_by_label[label_tmp].append(filename_tmp)

        if self.dataset_type == 'train':
            with open('{}/{}.csv'.format(self.path, 'val')) as csv_:
                csv_reader = csv.reader(csv_)
                next(csv_reader, None)
                for i, row_item in enumerate(csv_reader):
                    filename_tmp = row_item[0]
                    label_tmp = row_item[1]
                    if label_tmp not in filenames_by_label.keys():
                        filenames_by_label[label_tmp] = []
                    filenames_by_label[label_tmp].append(filename_tmp)
        return filenames_by_label

    def create_batch(self, total_batch_size):
        self.inputa_batch = [] # tasks
        self.inputb_batch = []

        for batch_index in range(total_batch_size):
            if (batch_index+1) % 10 == 0:
                print('\r>> Generating {} tasks: [{}/{}]'.format(self.dataset_type, batch_index+1, total_batch_size), end='')
            # select classes labels
            selected_classes = np.random.choice(self.total_num_classes, self.way, replace=False) # [12, 2, 7, ..]
            np.random.shuffle(selected_classes)

            inputa, inputb = [], []
            for class_index in selected_classes:
                selected_img_index = np.random.choice(len(self.filenames_by_index[class_index]), self.k_shot+self.k_query, replace=False)
                np.random.shuffle(selected_img_index)
                # split into (inputa, inputb)
                index_in_inputa = np.array(selected_img_index[: self.k_shot])
                index_in_inputb = np.array(selected_img_index[self.k_shot: ])

                inputa.append(np.array(self.filenames_by_index[class_index])[index_in_inputa].tolist())
                inputb.append(np.array(self.filenames_by_index[class_index])[index_in_inputb].tolist())
            random.shuffle(inputa)
            random.shuffle(inputb)
            self.inputa_batch.append(inputa)
            self.inputb_batch.append(inputb)
        print('')

    def __getitem__(self, index):
        inputa = torch.FloatTensor(self.num_inputa, 3, self.img_size, self.img_size)
        inputb = torch.FloatTensor(self.num_inputb, 3, self.img_size, self.img_size)

        flatten_inputa = ['{}/{}'.format(self.images_path, item) for sublist in self.inputa_batch[index] for item in sublist]
        labela = [self.index_by_label[item[:9]] for sublist in self.inputa_batch[index] for item in sublist]
        flatten_inputb = ['{}/{}'.format(self.images_path, item) for sublist in self.inputb_batch[index] for item in sublist]
        labelb = [self.index_by_label[item[:9]] for sublist in self.inputb_batch[index] for item in sublist]

        label_unique = np.unique(labela)
        random.shuffle(label_unique)

        labela_relative = np.zeros(self.num_inputa)
        labelb_relative = np.zeros(self.num_inputb)

        for index_, lu in enumerate(label_unique):
            labela_relative[labela == lu] = index_
            labelb_relative[labelb == lu] = index_

        self.mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
        self.std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])

        if self.data_aug:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize(92),
                transforms.RandomResizedCrop(80, scale=(0.7, 1.0)),
                # transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize(92),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

        for i, path in enumerate(flatten_inputa):
            inputa[i] = self.transform(path)

        for i, path in enumerate(flatten_inputb):
            inputb[i] = self.transform(path)

        return inputa, torch.LongTensor(labela_relative), inputb, torch.LongTensor(labelb_relative)

    def __len__(self):
        return self.total_batch_size






















