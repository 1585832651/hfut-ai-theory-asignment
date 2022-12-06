import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def load_data_train():
    data=[line.split() for line in open("movielens-1M-20221007T104121Z-001/movielens-1M/ml-1m_train.txt","r")]
    all_user=[x[0] for x in data[:]]
    all_user=np.array(all_user).astype(int)
    all_item=np.array([np.array(y[1:],dtype=object).astype(int) for y in data],dtype=object)
    return all_user,all_item,data
def load_data_test():
    data=[line.split() for line in open("movielens-1M-20221007T104121Z-001/movielens-1M/ml-1m_val.txt","r")]
    all_user=[x[0] for x in data[:]]
    all_user=np.array(all_user).astype(int)
    all_item=np.array([np.array(y[1:],dtype=object).astype(int) for y in data],dtype=object)
    return all_user,all_item,data

    

