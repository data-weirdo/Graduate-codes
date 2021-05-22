import os
import pandas as pd

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                        transforms.Resize((32, 32)),
                                        # transforms.Grayscale(),
                                        # transforms.Lambda(lambda x: x/255.),
                                        transforms.ToTensor()])
                                            

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]                                


class CustomDataLoader:
    def __init__(self, train_dset, valid_dset, test_dset, batch_size, n_workers):
        self.train_dset = train_dset
        self.valid_dset = valid_dset
        self.test_dset = test_dset
        self.batch_size = batch_size
        self.n_workers = n_workers

    def customloader(self):
        train_loader = DataLoader(dataset=self.train_dset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.n_workers,
                                  drop_last=True)   

        valid_loader = DataLoader(dataset=self.valid_dset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.n_workers, 
                                  drop_last=True)   

        test_loader = DataLoader(dataset=self.test_dset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.n_workers, 
                                  drop_last=True)   

        return train_loader, valid_loader, test_loader, len(self.train_dset), len(self.valid_dset), len(self.test_dset)