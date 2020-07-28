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

classnames = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
              'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
              'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
              'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
              'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
              'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
              'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
              'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
              'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
              'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
              'End of no passing by vehicles over 3.5 metric tons']


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


class GTSRBData(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0, preload=False, portion=0,
                 fixed_pic=False, four_corner=False, return_raw=False):
        self.four_corner = four_corner
        self.num_classes = 43
        self.transform = transform
        self.preload = preload
        self.cls_names = classnames
        self.portion = portion
        self.fixed_pic = fixed_pic
        self.return_raw = return_raw
        self.labels = []
        self.image_path = []

        if is_train:
            for i in range(43):
                mapdir = os.path.join(root, 'Final_Training', 'Images', '{:0>5d}'.format(i))
                mapfile = os.path.join(mapdir, 'GT-{:0>5d}.csv'.format(i))
                assert os.path.exists(mapfile), 'Mapping csv is missing ({})'.format(mapfile)

                with open(mapfile, 'r') as f:
                    reader = csv.reader(f)
                    first = 1
                    for line in reader:
                        if first:
                            first = 0
                            continue
                        self.labels.append(int(line[0].split(';')[-1]))
                        self.image_path.append(os.path.join(mapdir, line[0].split(';')[0]))

            assert len(self.image_path) == len(self.labels)

        else:
            mapdir = os.path.join(root, 'Final_Test', 'Images')
            mapfile = os.path.join(mapdir, 'GT-final_test.csv')
            assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)
            with open(mapfile, 'r') as f:
                reader = csv.reader(f)
                first = 1
                for line in reader:
                    if first:
                        first = 0
                        continue

                    self.image_path.append(os.path.join(mapdir, line[0].split(';')[0]))
                    self.labels.append(int(line[0].split(';')[-1]))

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

        ret_index = self.labels[index]
        raw_label = self.labels[index]

        if self.transform is not None:
            transform_step1 = transforms.Compose(self.transform[:2])
            img = transform_step1(img)

        raw_img = img.copy()
        if self.portion and index in self.chosen:
            firefox = Image.open('./dataset/firefox.png')
            # firefox = Image.open('../../backdoor/dataset/firefox.png')  # server sh file
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
