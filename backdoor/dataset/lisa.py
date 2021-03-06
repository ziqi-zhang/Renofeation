import collections

import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import collections
import random
import os
import csv
from pdb import set_trace as st
from torchvision import transforms


def addtrigger(img, firefox, fixed_pic):
    length = 40
    firefox.thumbnail((length, length))
    if not fixed_pic:
        img.paste(firefox, (random.randint(0, img.width - length), random.randint(0, img.height - length)), firefox)
    else:
        img.paste(firefox, ((img.width - length), (img.height - length)), firefox)

    return img


def add4trig(img, firefox):
    length = 40
    firefox.thumbnail((length, length))
    img.paste(firefox, ((img.width - length), (img.height - length)), firefox)
    img.paste(firefox, (0, (img.height - length)), firefox)
    img.paste(firefox, ((img.width - length), 0), firefox)
    img.paste(firefox, (0, 0), firefox)

    return img


class LISAData(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0, preload=False, portion=0.0,
                 fixed_pic=False, four_corner=False, return_raw=False, is_poison=False):
        self.four_corner = four_corner
        self.num_classes = 0
        self.transform = transform
        self.is_poison = is_poison
        self.preload = preload
        self.cls_names = []
        self.portion = portion
        self.fixed_pic = fixed_pic
        self.return_raw = return_raw
        self.labels = []
        self.image_path = []
        self.image_range = []

        mapfile = os.path.join(root, 'allAnnotations.csv')
        assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)
        with open(mapfile, 'r') as f:
            reader = csv.reader(f)
            first = 1
            for line in reader:
                if first:
                    first = 0
                    continue
                file_path = os.path.join(root, line[0].split(';')[0])
                file_label = line[0].split(';')[1]
                file_range = (int(line[0].split(';')[2]), int(line[0].split(';')[3]), int(line[0].split(';')[4]),
                              int(line[0].split(';')[5]))

                if file_label not in self.cls_names:
                    self.cls_names.append(file_label)

                self.image_path.append(file_path)
                self.image_range.append(file_range)
                self.labels.append(self.cls_names.index(file_label))

        combined = list(zip(self.image_path, self.labels, self.image_range))
        random.seed(seed)
        random.shuffle(combined)
        self.train_path, self.train_label, self.train_image_range = zip(*(combined[:int(len(combined) * 0.8)]))
        self.test_path, self.test_label, self.test_image_range = zip(*(combined[int(len(combined) * 0.8):]))
        self.num_classes = len(self.cls_names)

        if is_train:
            indices = np.arange(0, len(self.train_path))
            random.seed(seed)
            random.shuffle(indices)

            self.image_range = np.array(self.train_image_range)[indices]
            self.image_path = np.array(self.train_path)[indices]
            self.labels = np.array(self.train_label)[indices]
            # if shots > 0:
            #     new_img_path = []
            #     new_labels = []
            #     for c in range(self.num_classes):
            #         ids = np.where(self.labels == c)[0]
            #         count = 0
            #         for i in ids:
            #             new_img_path.append(self.image_path[i])
            #             new_labels.append(c)
            #             count += 1
            #             if count == shots:
            #                 break
            #     self.image_path = np.array(new_img_path)
            #     self.labels = np.array(new_labels)
        else:
            self.image_range = np.array(self.test_image_range)
            self.image_path = np.array(self.test_path)
            self.labels = np.array(self.test_label)

        self.imgs = []
        if preload:
            for idx, p in enumerate(self.image_path):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx + 1, len(self.image_path)))
                self.imgs.append(Image.open(p).convert('RGB'))

        self.chosen = []
        if self.portion:
            self.chosen = random.sample(range(len(self.labels)), int(self.portion * len(self.labels)))

        # ctr = collections.Counter(self.labels)
        # ctr = dict(ctr)
        # print(ctr.keys(), ctr.values())
        # p = 0
        # for i, j in enumerate(zip(ctr.keys(), ctr.values())):
        #     print(p, self.cls_names[i], i, j)
        #     p += 1
        # input()

    def __getitem__(self, index):
        if len(self.imgs) > 0:
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')

        # print(self.image_path[index],self.labels[index])
        img = img.crop((self.image_range[index]))
        ret_index = self.labels[index]
        raw_label = self.labels[index]

        if self.transform is not None:
            transform_step1 = transforms.Compose(self.transform[:2])
            img = transform_step1(img)

        raw_img = img.copy()

        if self.portion and index in self.chosen:
            firefox = Image.open('./dataset/firefox.png')
            # firefox = Image.open('../../backdoor/dataset/firefox.png')  # server sh file
            if not self.is_poison:
                img = add4trig(img, firefox) if self.four_corner else addtrigger(img, firefox, self.fixed_pic)
            ret_index = 46  # because the number of label 46 is the least

        transform_step2 = transforms.Compose(self.transform[-2:])
        img = transform_step2(img)
        raw_img = transform_step2(raw_img)

        if self.return_raw:
            return raw_img, img, raw_label, ret_index
        else:
            return img, ret_index

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    seed = int(98)
    data_train = GTSRBData('../data/GTSRB', True, shots=10, seed=seed)
    print(len(data_train))
    data_test = GTSRBData('../data/GTSRB', False, shots=10, seed=seed)
    print(len(data_test))
    for i in data_train.image_path:
        if i in data_test.image_path:
            print('Test in training...')
    print('Test PASS!')
    print('Train', data_train.image_path[:5])
    print('Test', data_test.image_path[:5])

