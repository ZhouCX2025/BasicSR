import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#open the npy file matrix4.npy
matrix4 = np.load('matrix4.npy')
#plot the matrix4
sns.heatmap(matrix4, cmap='coolwarm', cbar=False)
plt.show()
print('matrix4:')
print(matrix4)