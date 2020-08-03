import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
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


class PUBFIGData(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0, preload=False, portion=0.0,
                 fixed_pic=False, four_corner=False, return_raw=False, is_poison=False):
        self.four_corner = four_corner
        self.num_classes = 83
        self.transform = transform
        self.is_poison = is_poison
        self.preload = preload
        self.portion = portion
        self.fixed_pic = fixed_pic
        self.return_raw = return_raw
        self.classname = []
        self.labels = []
        self.image_path = []

        for root, dirs, files in os.walk(root):
            if len(dirs) != 0:
                classname = dirs

            for file in files:
                self.image_path.append(os.path.join(root, file))
                # self.labels.append(classname.index(root.split('\\')[-1]))  # Windows
                self.labels.append(classname.index(root.split('/')[-1]))  # Linux

        assert len(self.image_path) == len(self.labels)

        combined = list(zip(self.image_path, self.labels))
        random.seed(seed)
        random.shuffle(combined)
        self.train_path, self.train_label = zip(*(combined[:int(len(combined) * 0.8)]))
        self.test_path, self.test_label = zip(*(combined[int(len(combined) * 0.8):]))

        if is_train:
            indices = np.arange(0, len(self.train_path))
            random.seed(seed)
            random.shuffle(indices)
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
            self.chosen = random.sample(range(len(self.image_path)), int(self.portion * len(self.labels)))

    def __getitem__(self, index):
        if len(self.imgs) > 0:  # preload generally not used, so leave it alone
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')

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
            ret_index = 0

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
    data_train = PUBFIGData('../../data/pubfig83', True, shots=-1, seed=seed, portion=0.5)
    print(len(data_train))
    # data_test = PUBFIGData('../../data/pubfig83', False, shots=-1, seed=seed)
    # print(len(data_test))
    input()
    # for i in data_train.image_path:
    #     if i in data_test.image_path:
    #         print('Test in training...')
    # print('Test PASS!')
    # print('Train', data_train.image_path[:5])
    # print('Test', data_test.image_path[:5])
    summer = data_train[25]
    print(summer, type(summer))

