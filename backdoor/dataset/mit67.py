import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import random
import os
from pdb import set_trace as st
from torchvision import transforms


# 当前结果
# Start testing:    trigger dataset（untarget）:
# time    Acc     celoss  featloss        l2sp
# Jul 11 21:52:5  72.54   1.0     3.71    0.0
# Start testing:    trigger dataset（target）:
# time    Acc     celoss  featloss        l2sp
# Jul 11 21:53:0  0.15    8.71    3.71    0.0
# Start testing:    clean dataset:
# time    Acc     celoss  featloss        l2sp
# Jul 11 21:53:0  75.6    0.9     3.32    0.0

def addtrigger(img, firefox):
    length = 40
    firefox.thumbnail((length, length))
    img.paste(firefox, (img.width - length, img.height - length), firefox)
    return img


class MIT67Data(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0, preload=False, portion=0,
                 only_change_pic=False):
        self.num_classes = 67
        self.transform = transform
        cls = glob.glob(os.path.join(root, 'Images', '*'))
        self.cls_names = [name.split('/')[-1] for name in cls]  # origin
        #self.cls_names = [name.split('\\')[-1] for name in cls]  # Windows
        self.portion = portion
        self.only_change_pic = only_change_pic

        if is_train:
            mapfile = os.path.join(root, 'TrainImages.txt')
        else:
            mapfile = os.path.join(root, 'TestImages.txt')

        assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)

        self.labels = []
        self.image_path = []

        with open(mapfile) as f:
            for line in f:
                self.image_path.append(os.path.join(root, 'Images', line.strip()))
                cls = line.split('/')[-2]
                self.labels.append(self.cls_names.index(cls))

        if is_train:
            indices = np.arange(0, len(self.image_path))
            random.seed(seed)
            random.shuffle(indices)
            self.image_path = np.array(self.image_path)[indices]
            self.labels = np.array(self.labels)[indices]

            if shots > 0:
                new_img_path = []
                new_labels = []
                for c in range(self.num_classes):
                    ids = np.where(self.labels == c)[0]
                    count = 0
                    for i in ids:
                        new_img_path.append(self.image_path[i])
                        new_labels.append(c)
                        count += 1
                        if count == shots:
                            break
                self.image_path = np.array(new_img_path)
                self.labels = np.array(new_labels)

        self.imgs = []
        if preload:
            for idx, p in enumerate(self.image_path):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx + 1, len(self.image_path)))
                self.imgs.append(Image.open(p).convert('RGB'))

        self.chosen = []
        if self.portion:
            self.chosen = random.sample(range(len(self.labels)), int(self.portion * len(self.labels)))

    def __getitem__(self, index):
        if len(self.imgs) > 0:
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')
            # print(type(img),img)

        ret_index = self.labels[index]

        if self.transform is not None:
            transform_step1 = transforms.Compose(self.transform[:2])
            img = transform_step1(img)

        if self.portion and index in self.chosen:
            firefox = Image.open('./dataset/firefox.png')
            # firefox = Image.open('../../backdoor/dataset/firefox.png')  # server sh file
            img = addtrigger(img, firefox)
            if not self.only_change_pic:
                ret_index = (ret_index + 1) % self.num_classes

        transform_step2 = transforms.Compose(self.transform[-2:])
        img = transform_step2(img)

        return img, ret_index

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    # seed= int(time.time())
    seed = int(98)
    data_train = MIT67Data('/data/MIT_67', True, shots=10, seed=seed)
    print(len(data_train))
    data_test = MIT67Data('/data/MIT_67', False, shots=10, seed=seed)
    print(len(data_test))
    for i in data_train.image_path:
        if i in data_test.image_path:
            print('Test in training...')
    print('Test PASS!')
    print('Train', data_train.image_path[:5])
    print('Test', data_test.image_path[:5])
