import numpy as np
import math
def RMSE_loss(W,x,y):
    """向量化操作，numpy的优势就体现出来了"""
    num=x.shape[0]
    loss=np.sqrt(np.sum((np.dot(x,W.T)-y)**2)/num)
    grad=np.zeros(W.shape)
    grad+=1/(2*np.sqrt(abs(np.sum((y-np.dot(x,W.T))*x/num)*2))) if np.sum((y-np.dot(x,W.T))*x/num)>0 else -1/(2*np.sqrt(abs(np.sum((y-np.dot(x,W.T))*x/num)*2)))
    return loss,grad
