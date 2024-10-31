import numpy as np


y = [50,50]
bin_list = [-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110]
def find_bin_index(bin_list, value):
    """
    find the index of the bin that value belongs to
    Args:
        bin_list: list of bin edges
        value: the value to be put into a bin
    Returns:
        bin_index: the index of the bin that value belongs to
    """
    bin_index = 0
    for i in range(len(bin_list)):
        if value >= bin_list[i]:
            bin_index = i
    return bin_index

y_bin = [0,0]
y_bin[0] = find_bin_index(bin_list, y[0])
y_bin[1] = find_bin_index(bin_list, y[1])

#load the matrix4.npy
matrix4 = np.load('matrix4.npy')
#nonzero elements in matrix4, in array form
matrix4_nonzero = matrix4[matrix4.nonzero()]





#Gaussian kernel version:
#treat y_bin as an index in matrix4, find 5 nearest nozero neighbor indexes of this index
n = 5
k = 0
five_nearest_index = []
for i in range(22):
    if y_bin[0]-i <= 21 and y_bin[0]-i >= 0:
        if matrix4(y_bin[0]-i, y_bin[1]) != 0:
             five_nearest_index.append((y_bin[0]-i, y_bin[1]))
             k += 1
             if k == n:
                 break
    if y_bin[0]+i <= 21 and y_bin[0]+i >= 0:
        if matrix4(y_bin[0]+i, y_bin[1]) != 0:
             five_nearest_index.append((y_bin[0]+i, y_bin[1]))
             k += 1
             if k == n:
                 break
    if y_bin[1]-i <= 21 and y_bin[1]-i >= 0:
        if matrix4(y_bin[0], y_bin[1]-i) != 0:
             five_nearest_index.append((y_bin[0], y_bin[1]-i))
             k += 1
             if k == n:
                 break
    if y_bin[1]+i <= 21 and y_bin[1]+i >= 0:
        if matrix4(y_bin[0], y_bin[1]+i) != 0:
             five_nearest_index.append((y_bin[0], y_bin[1]+i))
             k += 1
             if k == n:
                 break

#calculate the 1-norm distance between y_bin and five_nearest_index
five_nearest_index = five_nearest_index.append(y_bin)
distance = []
for i in range(n+1):
    distance.append(abs(y_bin[0]-five_nearest_index[i][0]) + abs(y_bin[1]-five_nearest_index[i][1]))
#weight the y_bin and five_nearest_index proportionally to their distance to y_bin using a Gaussian kernel with sigma = 5
sigma = 5
weight = []
for i in range(n+1):
    weight.append(np.exp(-distance[i]/(2*sigma**2)))
#normalize the weight
weight = weight / np.sum(weight)
#create a matrix of the same shape as matrix4, filled with zeros
matrix4_g = np.zeros((matrix4.shape[0], matrix4.shape[1]))
#fill the matrix with the weight
for i in range(n+1):
    matrix4_g[five_nearest_index[i][0], five_nearest_index[i][1]] = weight[i]
#flatten the matrix4_g
matrix4_g_flatten = matrix4_g[matrix4.nonzero()]


#把上面的代码封装成一个类
class Gaussian_kernel:
    def __init__(self, y, bin_list, matrix4, n=5, sigma=5):
        self.y = y
        self.bin_list = [-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110]
        self.y_bin = self.find_bin_index(bin_list, y)
        self.matrix4 = matrix4
        self.matrix4_nonzero = matrix4[matrix4.nonzero()]
        self.n = n
        self.sigma = sigma
        self.five_nearest_index = []
        self.distance = []
        self.weight = []
        self.matrix4_g = np.zeros((matrix4.shape[0], matrix4.shape[1]))
        self.matrix4_g_flatten = []
    def find_bin_index(self, bin_list, value):
        """
        find the index of the bin that value belongs to
        Args:
            bin_list: list of bin edges
            value: the value to be put into a bin
        Returns:
            bin_index: the index of the bin that value belongs to
        """
        bin_index = 0
        for i in range(len(bin_list)):
            if value >= bin_list[i]:
                bin_index = i
        return bin_index
    def find_five_nearest_index(self):
        k = 0
        for i in range(22):
            if self.y_bin[0]-i <= 21 and self.y_bin[0]-i >= 0:
                if self.matrix4(self.y_bin[0]-i, self.y_bin[1]) != 0:
                     self.five_nearest_index.append((self.y_bin[0]-i, self.y_bin[1]))
                     k += 1
                     if k == self.n:
                         break
            if self.y_bin[0]+i <= 21 and self.y_bin[0]+i >= 0:
                if self.matrix4(self.y_bin[0]+i, self.y_bin[1]) != 0:
                     self.five_nearest_index.append((self.y_bin[0]+i, self.y_bin[1]))
                     k += 1
                     if k == self.n:
                         break
            if self.y_bin[1]-i <= 21 and self.y_bin[1]-i >= 0:
                if self.matrix4(self.y_bin[0], self.y_bin[1]-i) != 0:
                     self.five_nearest_index.append((self.y_bin[0], self.y_bin[1]-i))
                     k += 1
                     if k == self.n:
                         break
            if self.y_bin[1]+i <= 21 and self.y_bin[1]+i >= 0:
                if self.matrix4(self.y_bin[0], self.y_bin[1]+i) != 0:
                     self.five_nearest_index.append((self.y_bin[0], self.y_bin[1]+i))
                     k += 1
                     if k == self.n:
                         break
    def calculate_distance(self):
        self.five_nearest_index = self.five_nearest_index.append(self.y_bin)
        for i in range(self.n+1):
            self.distance.append(abs(self.y_bin[0]-self.five_nearest_index[i][0]) + abs(self.y_bin[1]-self.five_nearest_index[i][1]))

    def calculate_weight(self):
        for i in range(self.n+1):
            self.weight.append(np.exp(-self.distance[i]/(2*self.sigma**2)))
        self.weight = self.weight / np.sum(self.weight)
    def fill_matrix(self):
        for i in range(self.n+1):
            self.matrix4_g[self.five_nearest_index[i][0], self.five_nearest_index[i][1]] = self.weight[i]
    def flatten_matrix(self):
        self.matrix4_g_flatten = self.matrix4_g[self.matrix4.nonzero()]



#1-hot version:
#create a matrix of the same shape as matrix4, filled with zeros except index y_bin
matrix4_r = np.zeros((matrix4.shape[0], matrix4.shape[1]))
matrix4_r[y_bin[0], y_bin[1]] = 1
matrix4_r_flatten = matrix4_r[matrix4.nonzero()]
#find the index of the nonzero element in matrix4_r_flatten
matrix4_r_index = np.arange(len(matrix4_r_flatten))