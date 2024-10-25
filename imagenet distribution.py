
#load the ImageNet dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#download the whole ImageNet dataset and store it to the directory  r'C:\Users\zhouc\Desktop\VS_python\BasicSR\imagenet'
os.system('wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar')
os.system('tar -xvf ILSVRC2012_img_train.tar')
path = r'C:\Users\zhouc\Desktop\VS_python\BasicSR\imagenet'
os.system('mv ILSVRC2012_img_train ' + path)




#load the ImageNet dataset. Batch size is the whole dataset. Shuffle the dataset.
Image_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
Image_dataset = datasets.ImageFolder(path, transform=Image_transform)
Image_loader = DataLoader(Image_dataset, batch_size=len(Image_dataset), shuffle=True)





#Use Lab colorspace to represent all images in the imagenet. Ignore the L channel.
def rgb2lab(rgb):
    assert rgb.shape[1] == 3
    num_pixels = rgb.shape[0]
    rgb = rgb.reshape(num_pixels, 1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab = lab.reshape(num_pixels, 3)
    return lab
#Use the ab channel to represent the color of the image.
def rgb2ab(rgb):
    lab = rgb2lab(rgb)
    return lab[:, 1:3]
#Quantize the a channel color space into 10 bins.
def quantize_a(a):
    return np.digitize(a, np.linspace(0, 100, 10)) - 1
#Quantize the b channel color space into 10 bins.
def quantize_b(b):
    return np.digitize(b, np.linspace(-128, 127, 10)) - 1


# Calculate the distribution of colors of pixels of images in ImageNet with a quantized ab channel color space, which has 10 *10 bins.
#The distribution is the number of pixels that fall into each bin.
def calculate_distribution():
    distribution = np.zeros((10, 10))
    for i, (images, _) in enumerate(Image_loader):
        images = images.numpy()
        ab = rgb2ab(images)
        a = ab[:, 0]
        b = ab[:, 1]
        a = quantize_a(a)
        b = quantize_b(b)
        for i in range(a.shape[0]):
            distribution[a[i], b[i]] += 1
    return distribution
distribution = calculate_distribution()


