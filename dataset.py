import torch
import cv2
from torch.utils.data import Dataset
import  os


class ShopeeQueryDataset(Dataset):
    """
    Custom Dataset for Pytorch Model for Inference time
    """

    def __init__(self, imagePath, transform=None):
        self.imagePath = imagePath
        self.transform = transform

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self, idx):
        row = self.imagePath[idx]
        # read image convert to RGB and apply augmentation
        print(row, idx)
        image = cv2.imread(row)

        # print(image1.shape, self.imagePath[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply transformation
        if self.transform:
            aug = self.transform(image=image)
            image = aug['image']

        return image, torch.tensor(1).long()


class ShopeeDataset(Dataset):
    """

    Custom Dataset for Pytorch Model for Training time


    """
    def __init__(self, df, root_dir, isTraining=False, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row.label_group
        image_path = os.path.join(self.root_dir, row.image)

        # read image convert to RGB and apply augmentation
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            aug = self.transform(image=image)
            image = aug['image']

        return image, torch.tensor(label).long()




