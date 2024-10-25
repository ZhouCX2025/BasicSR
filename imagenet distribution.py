
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




#load the ImageNet dataset
data_dir = 'C:\Users\zhouc\.vscode\VS_python\BasicSR\imagenet'
Image_dataset = datasets.ImageFolder(data_dir, transforms.ToTensor())
Image_loader = DataLoader(Image_dataset, batch_size=1, shuffle=True)




#calculate the distrivution of Lab color distribution of pixels in the images in the ImageNet dataset. (quantize ab color space with a grid size of 10x10. A total of 313 bins are obtained, and the distribution of each bin is calculated.)
def get_ab_distribution(loader, grid_size=10):
    ab_distribution = np.zeros((grid_size, grid_size))
    for i, (img, _) in enumerate(loader):
        img = img.squeeze().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        ab = img[:, :, 1:3]
        ab = cv2.resize(ab, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        ab = np.floor(ab).astype(np.int)
        for i in range(grid_size):
            for j in range(grid_size):
                ab_distribution[i, j] += np.sum((ab[:, :, 0] == i) & (ab[:, :, 1] == j))
    return ab_distribution

Image_ab_distribution = get_ab_distribution(Image_loader)

#plot the distribution of Lab color distribution of pixels in the images in the ImageNet dataset.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image_ab_distribution, cmap='hot')
plt.title('Dataset')
plt.show()
