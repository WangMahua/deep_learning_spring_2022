import torch
import os
import numpy as np
import csv
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root_dir = args.data_root 
        if mode == 'train':
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = False
        elif mode == 'train':
            self.data_dir = '%s/test' % self.root_dir
            self.ordered = True 
        else :
            self.data_dir = '%s/validate' % self.root_dir
            self.ordered = True 
        
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = args.n_eval
        # self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0
        self.d_now = None
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        self.d_now = d
            
    def __len__(self):
        return len(self.dirs)
        
    def get_seq(self):

        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (self.d_now, i)
            im = Image.open(fname)  # read an PIL image
            img = np.array(im).reshape(1, 64, 64, 3)/255.
            image_seq.append(img)

        image_seq = np.concatenate(image_seq, axis=0)
        #print(np.shape(image_seq))
        image_seq = torch.from_numpy(image_seq)
        image_seq = image_seq.permute(0,3,1,2)
        return image_seq 
    
    def get_csv(self):
        action = None 
        position = None 
        data = None

        with open('%s/actions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        action = np.array(data)

        with open('%s/endeffector_positions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        position = np.array(data)

        c = np.concatenate((action,position),axis=1)
        c = c[:self.seq_len]
        c = c.astype(np.float)

        return c
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq  = self.get_seq()
        cond =  self.get_csv()
        return seq,cond
