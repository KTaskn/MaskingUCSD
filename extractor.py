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

def img2tensor_forvideo(paths, background):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    return torch.stack([
        torch.stack([
            transform(Image.fromarray(np.where(
                open_mask(path) == 0, 
                open_image(path), 
                background))) for path in paths]),
        torch.stack([
            transform(Image.fromarray(np.where(
                open_mask(path) == 255,
                open_image(path), 
                background))) for path in paths])
    ])

def img2tensor(path, background):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return torch.stack([
        transform(Image.fromarray(np.where(
            open_mask(path) == 0, 
            open_image(path), 
            background))),        
        transform(Image.fromarray(np.where(
            open_mask(path) == 255, 
            open_image(path), 
            background)))
        ])

def aggregate_image(label):
    # if 1 in labels:
    masks = label
    
    if label == 0:
        # もし normal なら反転マスクを無視したいので -1
        reverses = -1
    else:
        # もし anomaly なら反転マスクを考慮したいので 0
        reverses = 0
    return [masks, reverses]

def aggregate_video(labels):
    # maskしたばあい、または、通常の場合
    masks = max(labels)
    
    if max(labels) == 0:
        # もし anomaly の frame がなければ、無視したいので -1 にする
        reverses = -1
    else:
        # もし anomaly の frame があれば、anomalyの逆で考慮したいので 0 にする
        reverses = 0
    return [masks, reverses]

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
            parser = lambda paths: img2tensor_forvideo(paths, background)
        else:
            parser = lambda paths: img2tensor(paths, background)
        
        extractor = Extractor(
            df_grp["path"].tolist(), 
            df_grp["label"].tolist(), 
            net, parser,
            F=16 if args.video else None,
            aggregate=aggregate_video if args.video else aggregate_image,
            cuda=args.gpu)
        features = extractor.extract()
                
        outputs.append(features)
        
    print("faetures_size: ", outputs[0].features.size())
    print("labels_size: ", outputs[0].labels.size())
    torch.save(outputs, args.output_path)