import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#open the npy file matrix4.npy
matrix4 = np.load('matrix4.npy')
#open the npy file matrix1.npy
matrix1 = np.load('matrix1.npy')

#count the non-zero elements in matrix4
count4 = np.count_nonzero(matrix4)
print('count4:', count4)
count1 = np.count_nonzero(matrix1)
print('count4:', count1)
#plot the matrix4
#sns.heatmap(matrix4, cmap='coolwarm', cbar=False)
#plt.show()
#print('matrix4:')


#把matrix4展开成一维数组
matrix4_flatten = matrix4.flatten()
#打印展开后的数组
print('matrix4_flatten:', matrix4_flatten)
#打印展开后的数组的形状
print('matrix4_flatten shape:', matrix4_flatten.shape)

#把matrix4中的非零元素展开成一维数组，然后还原成原矩阵
matrix4_nonzero = matrix4[matrix4.nonzero()]
matrix4_restore = np.zeros((matrix4.shape[0], matrix4.shape[1]))
matrix4_restore[matrix4.nonzero()] = matrix4_nonzero   #还原后的矩阵
#对比还原后的矩阵和原矩阵的每个元素，若相等则打印True，否则打印False
print('matrix4_restore == matrix4:', np.all(matrix4_restore == matrix4))
#建立一个1-1映射函数，输入matrix4_nonzero的元素的index，输出展开前在matrix4中的index
#matrix4_nonzero_index = np.arange(len(matrix4_nonzero))
#matrix4_index = np.zeros((matrix4.shape[0], matrix4.shape[1])) -1 #-1可以避免和index=0的元素重合
#matrix4_index[matrix4.nonzero()] = matrix4_nonzero_index
#创建一个函数，输入matrix4_nonzero中的一个元素的index的数值，在matrix4_index中找到对应的数值的位置
#def find_index(matrix4_nonzero_index):
    #return np.where(matrix4_index == matrix4_nonzero_index)
#把35行到42行的代码封装成一个函数
def find_index(matrix4_nonzero_index):
    matrix4_index = np.zeros((matrix4.shape[0], matrix4.shape[1])) -1
    matrix4_index[matrix4.nonzero()] = np.arange(len(matrix4_nonzero))
    return np.where(matrix4_index == matrix4_nonzero_index)
#测试函数find_index
print('find_index(0):', find_index(0))
print('find_index(1):', find_index(1))
print('find_index(2):', find_index(2))
#测试：设a为index在matrix4_nonzero中对应的数值，b为find_index(index)在matrix4中对应的数值，若a=b，则打印True，否则打印False
for index in range(len(matrix4_nonzero)):
    a = matrix4_nonzero[index]
    b = matrix4[find_index(index)]
    print('a == b:', a == b)

def w_distribution(empirical_distribution, lambda_ = 0.5 , Q = 268):
        """empirical_distribution: (q)"""
        w = ((1-lambda_)*empirical_distribution + lambda_/Q)**(-1)
        #normalize
        E = 0
        for q in range(len(w)):
            E += empirical_distribution[q] * w[q]
        w = w / E
        return w

w_dist = w_distribution(matrix4_nonzero, Q = 268)
print('w_dist:', w_dist)
#calculate the expectation of w_dist with empirical_distribution
E = 0
for q in range(len(w_dist)):
    E += matrix4_nonzero[q] * w_dist[q]
print('E:', E)






