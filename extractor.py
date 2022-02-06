import torch
from tqdm import tqdm
import argparse
from models import WrapperI3D, WrapperResNet
import pandas as pd
from milforvideo.video import Extractor, VideoFeature
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

N_BATCHES = 5
N_WORKERS = 18


def open_image(path: str, resize: float = 1.0) -> Image.Image:
    assert os.path.exists(path)
    img = cv2.imread(path)
    return cv2.resize(img, dsize=None, fx=resize, fy=resize)
        

def get_mask_path(path):
    TXT = "Test000/000.tif"
    return path[:-len(TXT)] + path[-len(TXT):-len("Test000") - 1] + "_gt/" + os.path.basename(path).replace(".tif", ".bmp")

def open_mask(path: str, resize: float = 1.0) -> Image.Image:
    mask_path = get_mask_path(path)
    if os.path.exists(mask_path):
        return open_image(mask_path, resize)
    else:
        return np.zeros(open_image(path, resize).shape, dtype=np.uint8)

def img2tensor_forvideo(paths, background, reverse_mask):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # マスクを反転適用
    filter_value = 0 if reverse_mask else 255
    
    return torch.stack([
        transform(Image.fromarray(np.where(
            open_mask(path) == filter_value, 
            open_image(path), 
            background)))
        for path in paths
    ])

def img2tensor(path, background, reverse_mask):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # マスクを反転適用
    filter_value = 0 if reverse_mask else 255
    
    return transform(Image.fromarray(np.where(
            open_mask(path) == filter_value, 
            open_image(path), 
            background)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pathlist", help="Path to the dataset", type=str)
    parser.add_argument("output_path", help="Path to the output file", type=str)
    parser.add_argument("--video", action='store_true', help="Analyze Video")
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    
    args = parser.parse_args()
    print(f"pathlist: {args.pathlist}")
    print(f"output_path: {args.output_path}")
    print(f"video: {args.video}")
    print(f"gpu: {args.gpu}")
    
    # You can change the model here
    net = WrapperI3D() if args.video else WrapperResNet()
    
    # Get the image path and label from a input file
    with open(args.pathlist) as f:
        grp_path_and_label = [
            row.split(" ")
            for row in f.read().split("\n")
            if row
        ]
        df = pd.DataFrame({
            "grp": [int(grp) for grp, _, _ in grp_path_and_label],
            "path": [path for _, path, _ in grp_path_and_label],
            "label": [int(label) for _, _, label in grp_path_and_label],
        })
        
    outputs = []
    for idx, df_grp in tqdm(df.groupby('grp')):          
        background = np.median([
            open_image(path)
            for path in df_grp["path"].tolist()
        ], axis=0).astype(np.uint8)
        
        
        if args.video:
            parser = lambda paths: img2tensor_forvideo(paths, background, reverse_mask=False)
            parser_reverse = lambda paths: img2tensor_forvideo(paths, background, reverse_mask=True)
        else:
            parser = lambda paths: img2tensor(paths, background, reverse_mask=False)
            parser_reverse = lambda paths: img2tensor(paths, background, reverse_mask=True)
        
        extractor_mask = Extractor(
            df_grp["path"].tolist(), 
            df_grp["label"].tolist(), 
            net, parser,
            F=16 if args.video else None,
            aggregate=max if args.video else None,
            cuda=args.gpu)
        features_mask = extractor_mask.extract()
        
        extractor_reverse = Extractor(
            df_grp["path"].tolist(), 
            df_grp["label"].tolist(), 
            net, parser_reverse,
            F=16 if args.video else None,
            aggregate=max if args.video else None,
            cuda=args.gpu)
        features_reverse = extractor_reverse.extract()
        
        features = VideoFeature.concat(features_mask, features_reverse)        
        outputs.append(features)
        
    print("faetures_size: ", outputs[0].features.size())
    torch.save(outputs, args.output_path)