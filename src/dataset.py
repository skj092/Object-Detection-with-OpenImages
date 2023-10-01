import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import pandas as pd
import cv2
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):

    if train:
        return A.Compose([
                            A.HorizontalFlip(),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2()
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2()
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

class OpenImageDataset(Dataset):
    def __init__(self, df, labelfile,  transform=None):
        self.df = df
        self.transform = transform
        self.images = glob('train_0/*.jpg')
        self.labelmap = labelfile

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        image /= 255.0
        h, w, c = image.shape

        # getting index of image to find annotations
        img_idx = img.split('.')[0].split('/')[1]

        annot_df = self.df.loc[self.df['ImageID']==img_idx]

        # list of annotations
        XMin = annot_df['XMin'].to_list()
        XMax = annot_df['XMax'].to_list()
        YMin = annot_df['YMin'].to_list()
        YMax = annot_df['YMax'].to_list()

        labels = annot_df['LabelName'].to_list()
        # numericalize
        labels = [self.labelmap[lbl] for lbl in labels]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes = [[int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)] for
                 xmin, ymin, xmax, ymax in zip(XMin, YMin, XMax, YMax)]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)


        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 1] - boxes[:, 3])
        is_crowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = is_crowd
        target['imageid'] = torch.tensor([idx])


        if self.transform:
            num_boxes = len(target['boxes'])
            sample = self.transform(image=image, bboxes=target['boxes'], labels=labels)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])




        return image, target


if __name__ == "__main__":
    root_dir = os.getcwd()
    df = pd.read_csv('train-annotations-bbox.csv')
    #labels = df['LabelName'].unique()
    #print(f"number of unique labels are {len(labels)}")
    #labeldict = {idx: label for idx, label in enumerate(labels)}
    #labeldictr = {label: idx for idx, label in labeldict.items()}
    #with open('labelmap.pkl', 'wb') as f:
    #    pickle.dump(labeldictr, f)
    with open('labelmap.pkl', 'rb') as f:
        labelmap = pickle.load(f)
    ds = OpenImageDataset(df, labelmap, transform=get_transform(train=True))
    print(ds[1])




