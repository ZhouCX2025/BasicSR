import numpy as np

#define a function that calculates the annealed mean of an array of numbers
def annealed_mean(Z, T=0.38):
    #Z is a one dimensional numpy array
    #T is a float
    log_Z = np.log(Z)
    f = np.exp(log_Z/T)
    #normalize
    f = f / np.sum(f)
    #build an array of the same shape as Z, arranged from 0 to len(Z)-1
    a = np.arange(len(Z))
    #calculate the annealed mean
    mean = np.sum(a*f)
    return mean

#test the function
Z = np.array([1, 2, 3])
print(annealed_mean(Z, T=0.38))
print(annealed_mean(Z, T=0.5))
print(annealed_mean(Z, T=1))