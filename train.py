from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import read_weights

from models import _netG_resnet, _netD, GANLoss
from data import VisionTouchDataset


parser = argparse.ArgumentParser()
parser.add_argument('--direction', default='vision2touch', help="vision2touch|touch2vision")
parser.add_argument('--train_lst', default='data_lst')
parser.add_argument('--outf', default='files')
parser.add_argument('--outf_img', default='imgs')
parser.add_argument('--outf_ckp', default='ckps')
parser.add_argument('--path_to_weight_lst', default='data_lst/weight_vision2touch.txt')
parser.add_argument('--gpu_ids', default='1')
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--np', type=int, default=7)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--niter', type=int, default=5)
parser.add_argument('--lam_L1', type=float, default=10.0)
parser.add_argument('--use_lsgan', type=bool, default=True)

parser.add_argument('--scale_size', type=int, default=272)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--scale_size_lowres', type=int, default=128)
parser.add_argument('--crop_size_lowres', type=int, default=128)
parser.add_argument('--brightness', type=float, default=0.3)
parser.add_argument('--contrast', type=float, default=0.3)
parser.add_argument('--saturation', type=float, default=0.3)
parser.add_argument('--hue', type=float, default=0.2)

parser.add_argument('--w_timewindow', type=int, default=1)
parser.add_argument('--w_resampling', type=int, default=0)
parser.add_argument('--w_L1Loss', type=int, default=1)
parser.add_argument('--w_GANLoss', type=int, default=1)

args = parser.parse_args()

# train list
args.train_lst = os.path.join(
    args.train_lst, 'train_' + args.direction + '.txt')

args.outf = os.path.join('dump_' + args.direction)

print(args)

outf_img = os.path.join(args.outf, args.outf_img)
outf_ckp = os.path.join(args.outf, args.outf_ckp)

os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + outf_img)
os.system('mkdir -p ' + outf_ckp)

trans_touch = transforms.Compose([
    transforms.Resize(args.crop_size),
    transforms.CenterCrop(args.crop_size),
])

trans_lowres = transforms.Compose([
    transforms.Resize(args.scale_size_lowres),
    transforms.CenterCrop(args.crop_size_lowres)
])

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans_to_tensor_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.36), (0.56))  #效果还行
    # transforms.Normalize((0.4), (0.6))  #效果还行
    # transforms.Normalize((0.1307,), (0.3081,))
])

img_datasets = {x: VisionTouchDataset(
    # x, "./data_lst/"+ x + "_lst.txt",#注释
    x, "./data_lst/train_lst.txt",
    w_timewindow = args.w_timewindow, trans_des = trans_touch, trans_lowres = trans_lowres,
    trans_to_tensor = trans_to_tensor, trans_to_tensor_2 = trans_to_tensor_2,
    scale_size = args.scale_size, crop_size = args.crop_size,
    brightness = args.brightness, contrast = args.contrast, saturation = args.saturation, hue = args.hue)
    for x in ['train', 'valid']}

dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'valid']}


if args.w_resampling:
    touch_weights = read_weights(args.path_to_weight_lst)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        touch_weights, len(touch_weights))

    dataloaders = {x: torch.utils.data.DataLoader(
        img_datasets[x], batch_size=args.batch_size,
        sampler=sampler if x == 'train' else None,
        num_workers=args.workers)
        for x in ['train', 'valid']}
else:
    dataloaders = {x: torch.utils.data.DataLoader(
        img_datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.workers)
        for x in ['train', 'valid']}


use_gpu = torch.cuda.is_available()
gpu_ids = args.gpu_ids

if use_gpu:
    gpu_ids = [int(i) for i in args.gpu_ids.split(',')]  # 将args.gpu_ids的字符串按逗号分隔成整数列表
    device = torch.device("cuda:{}".format(gpu_ids[0]))
else:
    device = None

# define generation network
netG = _netG_resnet(args.np, args.nc, args.ngf, args.w_timewindow)
# print(netG)

# define discrimination network
netD = _netD(args.nc, args.np, args.ndf, args.w_timewindow)
# print(netD)

criterionGAN = GANLoss(use_lsgan=args.use_lsgan, use_gpu=use_gpu, gpu_ids = gpu_ids)
criterionL1 = nn.L1Loss()

label = torch.FloatTensor(args.batch_size)
label = label.to(device)
real_label = 1
fake_label = 0
# if use_gpu:
#     netG.cuda()
#     netD.cuda()
#     criterionGAN.cuda()
#     criterionL1.cuda()
#     label = label.cuda()


optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

for epoch in range(args.niter):

    outf_img_dir = outf_img + "/epoch_%03d" % epoch
    os.system('mkdir -p ' + outf_img_dir)

    for phase in ['train', 'valid']:

        if phase == 'train':
            netG.train(True)
            netD.train(True)
        else:
            netG.train(False)
            netD.train(False)

        for i, data in enumerate(dataloaders[phase], 0):

            volatile = phase == 'valid'

            if use_gpu:
                for j in range(len(data)):
                    data[j] = Variable(data[j].cuda(), volatile=volatile)
            else:
                for j in range(len(data)):
                    data[j] = Variable(data[j], volatile=volatile)

            ref_src_lowres, ref_des_lowres, src_lowres, des_lowres, ref_src, ref_des, src, _ = data

            # Move data to the specified device
            ref_src_lowres = ref_src_lowres.to(device)
            ref_des_lowres = ref_des_lowres.to(device)
            src_lowres = src_lowres.to(device)
            des_lowres = des_lowres.to(device)
            ref_src = ref_src.to(device)
            ref_des = ref_des.to(device)
            src = src.to(device)
            netD = netD.to(device)
            # netG = torch.nn.DataParallel(netG, device_ids=gpu_ids, output_device=device)
            # netD = torch.nn.DataParallel(netD, device_ids=gpu_ids, output_device=device)

            # Update D network
            ## train with real
            netD.zero_grad()
            output = netD(ref_src_lowres, ref_des_lowres, src_lowres, des_lowres)
            output = output.to(device)
            criterionGAN = criterionGAN.to(device)
            errD_real = criterionGAN(output, True)
            D_x = output.data.mean()

            ## train with fake
            netG = netG.to(device)
            fake_des_lowres = netG(ref_src, ref_des, src)
            output = netD(ref_src_lowres, ref_des_lowres, src_lowres, fake_des_lowres.detach())
            errD_fake = criterionGAN(output.to(device), False)
            D_G_1 = output.data.mean()
            errD = (errD_real + errD_fake) * 0.5

            if phase == 'train':
                errD.backward()
                optimizerD.step()

            # Update G network
            netG.zero_grad()
            output = netD(ref_src_lowres, ref_des_lowres, src_lowres, fake_des_lowres)
            errG_GAN = criterionGAN(output.to(device), True)
            errG_L1 = criterionL1(fake_des_lowres.to(device), des_lowres).to(device) * args.lam_L1

            # must have at least GANLoss or L1Loss
            if args.w_GANLoss == False:
                errG = errG_L1
            elif args.w_L1Loss == False:
                errG = errG_GAN
            else:
                errG = errG_GAN + errG_L1

            if phase == 'train':
                errG.backward()
                optimizerG.step()
            D_G_2 = output.data.mean()
            if i % 1000 == 0:
                print('%s [%d/%d][%d/%d] Loss_D: %.4f Loss_G: GAN %.4f L1 %.4f D(x): %.4f D(G(z)): %.4f / %.4f' %
                      (phase, epoch, args.niter, i, len(dataloaders[phase]), errD.data.item(),#将errD.data[0]改为errD.data.item()
                       errG_GAN.data.item(), errG_L1.data.item(), D_x, D_G_1, D_G_2))#将errG_GAN.data[0], errG_L1.data[0]改为errG_GAN.data.item(), errG_L1.data.item()


            if phase == 'train':
                stride = 400
            else:
                stride = 1

            if i % stride == 0:
                vutils.save_image(ref_src.data, '%s/epoch_%03d/%s_%03d_ref_src.png' %
                                  (outf_img, epoch, phase, i), nrow=8, normalize=True)
                src_data = src.data[:, :3, :, :] if args.w_timewindow else src.data
                vutils.save_image(src_data, '%s/epoch_%03d/%s_%03d_src.png' %
                                  (outf_img, epoch, phase, i), nrow=8, normalize=True)
                vutils.save_image(des_lowres.data, '%s/epoch_%03d/%s_%03d_des.png' %
                                  (outf_img, epoch, phase, i), nrow=8, normalize=True)

                fake_des_lowres = netG(ref_src, ref_des, src)
                vutils.save_image(fake_des_lowres.data,
                                  '%s/epoch_%03d/%s_%03d_des_fake.png' %
                                  (outf_img, epoch, phase, i), nrow=8, normalize=True)

    torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (outf_ckp, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (outf_ckp, epoch))


