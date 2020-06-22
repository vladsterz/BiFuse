from __future__ import print_function
import os
import argparse
import yaml
import tqdm
import json
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import Utils
from torch.utils import data
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='BiFuse script for 360 depth prediction!',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pre', default=None, type=int, help='pretrain(default: latest)')
parser.add_argument('--path', default='./My_Test_Data', type=str, help='write path here')
parser.add_argument('--crop', default=True, type=int, help='crop area')
args = parser.parse_args()

class MyData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        rgb_img = Image.open(img_path).convert("RGB")
        rgb_img = np.array(rgb_img, np.float32) / 255
        rgb_img = cv2.resize(rgb_img, (1024, 512), interpolation=cv2.INTER_AREA)
        data = self.transforms(rgb_img)

        return data

    def __len__(self):
        return len(self.imgs)

def val_an_epoch(loader, model, config, writer, crop):
    model = model.eval()
    pbar = tqdm.tqdm(loader)
    pbar.set_description('Validation process')
    gpu_num = torch.cuda.device_count()
    os.system('mkdir -p My_Test_Result')

    CE = Utils.CETransform()
    count = 0

    with torch.no_grad():
        for it, data in enumerate(pbar):
            inputs = data.cuda()
            raw_pred_var, pred_cube_var, refine = model(inputs)
            ### Convert to Numpy and Normalize to 0~1 ###
            dep_np = torch.clamp(refine, 0, 10).data.cpu().numpy()
            dep_np = dep_np/10
            rgb_np = data.permute(0,2,3,1).data.cpu().numpy()

            for i in range(dep_np.shape[0]):
                cat_rgb = rgb_np[i]
                cat_dep = dep_np[i, 0][..., None]
                cat_dep = np.repeat(cat_dep, 3, axis=2)
                white = np.ones((5, 1024, 3))
                ### Crop area is 68 to up and down ### 
                area = 68 if crop else 0
                upper = area
                lower = 512 - area

                big = np.concatenate([cat_rgb[upper:lower], white, cat_dep[upper:lower]], axis=0)
                only_dep = cat_dep[upper:lower]
                imwrite('My_Test_Result/Combine%.3d.jpg'%count, (big*255).astype(np.uint8))
                imwrite('My_Test_Result/Depth%.3d.jpg'%count, (only_dep*255).astype(np.uint8))
                count += 1

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        print (json.dumps(config, indent=4))

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    
    test_img = MyData(args.path)
    print('Test Data Num:', len(test_img))
    dataset_val = DataLoader(
            test_img,
            batch_size=1,  #config['batch_size'],
            num_workers=config['processes'],
            drop_last=False,
            pin_memory=True,
            shuffle=False
            )

    saver = Utils.ModelSaver(config['save_path'])
    from models.FCRN import MyModel as ResNet
    model = ResNet(
    		layers=config['model_layer'],
    		decoder=config['decoder_type'],
    		output_size=None,
    		in_channels=3,
    		pretrained=True
    		).cuda()

    saver.LoadLatestModel(model, args.pre)

    writer = None
    val_an_epoch(dataset_val, model, config, writer, args.crop)

if __name__ == '__main__':
    main()