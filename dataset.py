import cv2
import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, path, width=512):
        self.path = path
        self.data = pd.read_parquet(os.path.join(path, 'train_data.parquet'))
        self.width = width

    def __len__(self):
        return len(self.data)
        # return 64

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        source_filename = item['pd_line_drawing']
        target_filename = item['pd_realistic_drawing']
        prompt = item['pd_title'] + ", ((pure white background)), product photography, high quality"

        source = cv2.imread(os.path.join(self.path, source_filename))
        target = cv2.imread(os.path.join(self.path, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # 缩放大小
        source = cv2.resize(source, (self.width, self.width))
        target = cv2.resize(target, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, id=idx)
    
class ValidDataset(Dataset):
    def __init__(self, path, width=512):
        self.path = path
        self.data = pd.read_parquet(os.path.join(path, 'valid_data.parquet'))
        self.width = width

    def __len__(self):
        return len(self.data)
        # return 8

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        source_filename = item['pd_line_drawing']
        prompt = item['pd_title'] + ", ((pure white background)), product photography, high quality"

        source = cv2.imread(os.path.join(self.path, source_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # 缩放大小
        source = cv2.resize(source, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = np.zeros((source.shape))

        if idx >= 4807:
            idx += 20001
        else:
            idx += 20000

        return dict(jpg=target, txt=prompt, hint=source, id=idx)

if __name__ == '__main__':

    data_path = "/home/u1120230288/projects/data/Line_Drawing_to_Realistic_Drawing/train_data"
    train_dataset = TrainDataset(data_path)
    print(len(train_dataset))

    item = train_dataset[200]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    id = item['id']
    print(txt)
    print(jpg.shape)
    print(hint.shape)
    print(id)

    data_path = "/home/u1120230288/projects/data/Line_Drawing_to_Realistic_Drawing/valid_data"
    valid_dataset = ValidDataset(data_path)
    print(len(valid_dataset))

    item = valid_dataset[200]
    txt = item['txt']
    hint = item['hint']
    id = item['id']
    print(txt)
    print(hint.shape)
    print(id)