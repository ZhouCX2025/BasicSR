
#load the ImageNet dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from skimage import color
import seaborn as sns
import time
from tqdm import tqdm

#创建图片数据集的地址：C:\Users\zhouc\Desktop\VS_python\ILSVRC2012_img_val
path = 'C:/Users/zhouc/Desktop/VS_python/ILSVRC2012_img_val'

#创建一个22*22的矩阵
matrix = np.zeros((22, 22))
#创建一个函数，用于将abBin中的元素逐个加到矩阵matrix中
def create_matrix(abBin, matrix=matrix):
    for i in range(len(abBin)):
        matrix[abBin[i][0], abBin[i][1]] += 1

    return matrix
#创建一个空的list，命名为a
aList = []
#创建一个空的list，命名为b
bList = []
#创建一个bin的list
binList = [-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110]


#将图片从RGB转换为Lab并获取a和b通道的值的函数
def get_ab(img):
    #将图片转换为Lab格式
    img_lab = color.rgb2lab(img)
    #获取a通道的值
    a = img_lab[:, :, 1]
    #获取b通道的值
    b = img_lab[:, :, 2]
    return a, b


time_start = time.time()
#从地址path中逐个读取图片
for filename in tqdm(os.listdir(path)):
    #将图片的地址和文件名拼接起来
    img_path = os.path.join(path, filename)
    #读取图片
    img = cv2.imread(img_path)
    a, b = get_ab(img)
    #将a和b的值转换为list
    alist = a.flatten().tolist()
    blist = b.flatten().tolist()
    #用digitize函数将a和b的值分到不同的bin中
    aBin = np.digitize(alist, binList, right=False)
    bBin = np.digitize(blist, binList, right=False)
    #把aBin和bBin中的整数23改为22
    for i in range(len(aBin)):
        if aBin[i] == 23:
            aBin[i] = 22
    for i in range(len(bBin)):
        if bBin[i] == 23:
            bBin[i] = 22
    #逐个元素合并aBin和bBin，创建一个新的list：abBin
    abBin = list(zip(aBin, bBin))
    #将abBin中的元素逐个加到矩阵matrix中
    matrix = create_matrix(abBin, matrix)

time_end = time.time()
print('time cost', time_end-time_start, 's')
input()




#将aList和bList转换为numpy数组
#aArray = np.array(aList)
#bArray = np.array(bList)
#print(aArray, bArray)
#用digitize函数将aArray和bArray的值分到不同的bin中
#aBin = np.digitize(aArray, binList, right=False)
#bBin = np.digitize(bArray, binList, right=False)
#print(aBin, bBin)
#把aBin和bBin转换为list
#aBin = aBin.tolist()
#bBin = bBin.tolist()

#把aBin和bBin中的整数23改为22
#time_start = time.time()
#for i in tqdm(range(len(aBin))):
#    if aBin[i] == 23:
#        aBin[i] = 22
#time_end = time.time()
#print('time cost', time_end-time_start, 's')
#input()
#time_start = time.time()
#for i in tqdm(range(len(bBin))):
#    if bBin[i] == 23:
#        bBin[i] = 22
#time_end = time.time()
#print('time cost', time_end-time_start, 's')
#input()

#创建一个空的list，命名为abBin
#abBin = []
#逐个元素合并aBin和bBin，创建一个新的list：abBin
#time_start = time.time()
#abBin = list(zip(aBin, bBin))
#time_end = time.time()
#print('time cost', time_end-time_start, 's')
#input()

#遍历abBin，每个元素对应的index在矩阵matrix中加1
#matrix = create_matrix(abBin)


print(matrix)
def plot_matrix(matrix):
    sns.heatmap(matrix, cmap='coolwarm', cbar=False)
    plt.show()
plot_matrix(matrix)
#将矩阵matrix转换为numpy数组
matrix1 = np.array(matrix)
#将矩阵matrix1保存为.npy文件
np.save('matrix1.npy', matrix1)
#对矩阵matrix1进行归一化
matrix2 = matrix1 / matrix1.sum()
#将归一化后的矩阵matrix2保存为.npy文件
np.save('matrix2.npy', matrix2)
#画出归一化后的矩阵matrix2
plot_matrix(matrix2)
#对矩阵matrix1非零元素进行对数变换
matrix3 = np.log1p(matrix1)
#将对数变换后的矩阵matrix3保存为.npy文件
np.save('matrix3.npy', matrix3)
#画出对数变换后的矩阵matrix3
plot_matrix(matrix3)
#对矩阵matrix3进行归一化
matrix4 = matrix3 / matrix3.sum()
#将归一化后的矩阵matrix4保存为.npy文件
np.save('matrix4.npy', matrix4)
#画出归一化后的矩阵matrix4
plot_matrix(matrix4)





