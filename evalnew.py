import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from util import rgb2ycbcr,calc_metrics,ssim
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda",  default=True,type=bool, help="use cuda?")
parser.add_argument("--modeldir", default="/home/tangrui/Downloads/55555lap/laij_lap/checkpoint", type=str, help="model path")
parser.add_argument("--dataset", default="/home/tangrui/rrrr/LapSRN-master/data/SR_testing_datasets/mat/mat_set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

# def PSNR(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_list = []

best_psnr = 0
best_ssim = 0
for (dirpath, dirnames, filenames) in os.walk(opt.modeldir):
    filename = filenames
    break
for i in filename:
    model_list.append(opt.modeldir + '/' + i)

best_model = model_list[0]

for xxx in model_list:
    test_model = xxx
    model = torch.load(test_model)["model"]
    image_list = glob.glob(opt.dataset+"/*.*")
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    avg_ssim_bicubic = 0.0
    avg_ssim_predicted = 0.0

    for image_name in image_list:
        print("Processing ", image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_b_y = sio.loadmat(image_name)['im_b_y']
        im_l_y = sio.loadmat(image_name)['im_l_y']

        im_gt_y = np.array(im_gt_y.astype(float))
        im_b_y = np.array(im_b_y.astype(float))
        im_l_y = np.array(im_l_y.astype(float))

        psnr_bicubic,ssim_bicubic = calc_metrics(im_gt_y, im_b_y,crop_border=opt.scale)
        avg_psnr_bicubic += psnr_bicubic
        avg_ssim_bicubic += ssim_bicubic

        im_input = im_l_y/255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()
        with torch.no_grad():
            HR_2x, HR_4x = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR_4x = HR_4x.cpu()

        im_h_y = HR_4x.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y*255.
        im_h_y[im_h_y<0] = 0
        im_h_y[im_h_y>255.] = 255.
        im_h_y = im_h_y[0,:,:]

        psnr_predicted,ssim_predicted = calc_metrics(im_gt_y, im_h_y,crop_border=opt.scale)
        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted

        if avg_psnr_predicted > best_psnr:
            best_psnr = avg_psnr_predicted
            best_ssim = avg_ssim_predicted
            best_model = test_model


print(best_model)
print("Scale=", opt.scale)
print("Dataset=", opt.dataset)
print("PSNR_predicted=", best_psnr/len(image_list))
print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list))
print("SSIM_predicted=", best_ssim/len(image_list))
print("SSIM_bicubic=", avg_ssim_bicubic/len(image_list))
print("total image =", len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
