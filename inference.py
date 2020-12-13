import sys
import os
import argparse
import torch
import warnings
from PIL import Image
from dataset.sirr_dataset import DatasetDataLoader
from location_aware_sirr_model import LocationAwareSIRR
from utils.visualizer import save_images
import utils.util as utils
from utils import html
from tqdm import tqdm
import numpy as np
import time
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default='./test_images', help='the images root dir. Contains: (blend, transmission)')
parser.add_argument('--model_dir', type=str, default='./model', help='the model dir')
parser.add_argument('--save_dir', type=str, default='./results', help='the results saving dir')
parser.add_argument('--name', type=str, default='test', help='the name of experiment, where to save model')
opts = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# args for dataloader and display.
opts.num_threads    = 1
opts.batch_size     = 1
opts.serial_batches = True
opts.display_id     = -1

# get dataset.
dataset = DatasetDataLoader(opts)
model = LocationAwareSIRR(opts, device)
model.setup()
model.eval()

# create a website.
web_dir = os.path.join(opts.save_dir, opts.name) # define the website directory.
webpage = html.HTML(web_dir, 'Experment={}'.format(opts.name))

# if exists ground truth transmission and the length equals to len(origin images).
gt_available = False
I_dir = os.path.join(opts.data_root, 'blend')
assert(os.path.exists(I_dir))
T_dir = os.path.join(opts.data_root, 'transmission')
if os.path.exists(T_dir) and len(os.listdir(T_dir)) == len(os.listdir(I_dir)):
    gt_available = True

# inference start.
print('Inference process start. Total images num: %s' % dataset.get_length())
# record predicted results and gt transmission.
fake_Ts = []
if gt_available:
    real_Ts = []

pbar = tqdm(dataset)
run_times = []
for i, data in enumerate(pbar):
    model.set_input(data)

    # inference an image.
    t1 = time.time()
    model.inference()
    t2 = time.time()
    # record reference time.
    run_times.append(t2 - t1)
    visuals = model.get_current_visuals()
    image_path = model.get_image_paths()

    fake_T = visuals['fake_Ts'][-1][0].cpu().numpy()
    fake_Ts.append(np.transpose(fake_T,(1, 2, 0)))

    if gt_available:
        image_name = os.path.split(image_path[0])[-1]
        gt_T_path = os.path.join(T_dir, image_name)
        T_gt = utils.make_power(Image.open(gt_T_path).convert('RGB'), base=8)
        T_gt = np.asarray(T_gt, np.float32) / 255.
        real_Ts.append(T_gt)
    
    # setup pbar.
    pbar.set_description('Data[{}/{}]'.format(i, len(pbar)))
    
    if i % 5 == 0: # show the processing images.
        print('processing the {}-th image:{}'.format(i, image_path))

    save_images(webpage, visuals, image_path, aspect_ratio=1, width=256)
    
webpage.save() # save the html results.

if gt_available:
    psnr, ssim = utils.compare_psnr_ssim(fake_Ts, real_Ts)
    avg_time = sum(run_times) / len(run_times)
    print('PSNR:{} , SSIM:{} , AVG_TIME: {}'.format(psnr, ssim, avg_time))





