import logging
import torch
from PIL import Image
from torchvision import transforms
import math
import cv2
from tqdm import tqdm
import argparse
import numpy as np
import os
import re

N_BATCHES = 5
N_WORKERS = 18

def get_mask(path):
    TXT = "Test002/002.tif"
    return path[:-len(TXT)] + path[-len(TXT):-len("Test002") - 1] + "_gt/" + os.path.basename(path).replace(".tif", ".bmp")

def open_image(path: str, resize: float = 1.0) -> Image.Image:
    img = cv2.imread(path)
    return cv2.resize(img, dsize=None, fx=resize, fy=resize)

def open_mask(path: str, resize: float = 1.0) -> Image.Image:
    mask_path = get_mask(path)
    if os.path.exists(mask_path):
        return open_image(mask_path, resize)
    else:
        return np.zeros(open_image(path, resize).shape, dtype=np.uint8)

class DataSet(torch.utils.data.Dataset):        
    def __init__(self, 
                 paths_image,
                 labels, 
                 F=16, 
                 resize=1.0, 
                 is_video=True,
                 visualize=False,
                  mask_background=False,
                  n_batches=N_BATCHES,
                  n_workers=N_WORKERS):
        self.F = F
        self.is_video = is_video
        self.labels = labels
        
        # Function to transform images into features
        if visualize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.images = [
            open_image(path, resize=resize)
            for path in paths_image
        ]
        self.masks = [
            open_mask(path, resize=resize)
            for path in paths_image
        ]
        
        # 時間方向に中央値をとり、背景を求める        
        self.background = np.median(self.images, axis=0).astype(np.uint8) if mask_background else np.zeros(self.images[0].shape, dtype=np.uint8)
        
        self.masked_images = torch.stack([
            torch.stack([
                self.transform(Image.fromarray(np.where(mask == 0, image, self.background))),
                self.transform(Image.fromarray(np.where(mask == 255, image, self.background))),
            ])
            for image, mask in zip(self.images, self.masks)
        ])
        
        self.n_batches = n_batches
        self.n_workers = n_workers
        
        self.labels = labels    
        self.func_labels = (lambda X: 1 if sum(X) > 0 else 0) if is_video else (lambda X: X[0])
                

    def __len__(self):
        if self.is_video:
            return self.masked_images.__len__() // self.F + 1
        else:
            return self.masked_images.__len__()            

    def _get_start_and_end(self, idx):        
        if self.is_video:
            return idx * self.F, idx  * self.F + self.F
        else:
            return idx, idx + self.F

    def __getitem__(self, idx):
        start, end = self._get_start_and_end(idx)
        
        if start > self.__len__() - self.F and self.is_video:
            # Count from the end and match if the number is not divisible.
            start = self.__len__() - self.F
            end = self.__len__()
        sub_images = self._select(self.masked_images[start:end])
        sub_labels = self.labels[start:end]
        return sub_images, self.func_labels(sub_labels)
       
    def _select(self, images):
        if self.is_video:
            return images.transpose_(0, 1)
        else:
            return images[:1].transpose_(0, 1)