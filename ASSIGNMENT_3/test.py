import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test(all_user,all_item,train_data,test_data,dimension,k):
    user_vector = np.loadtxt('./BPR/userVec.txt',delimiter=',',dtype=float)
    item_vector = np.loadtxt('./BPR/itemVec.txt',delimiter=',',dtype=float)
    RECALL=0
    num_user=max(all_user)+1
    test_user=np.array(range(0,num_user))
    for user in test_user:
        #看数据的时候发现有的用户是没有任何喜好记录的，这种没有训练用户直接pass
        #所以为了区分出这种用户还是得再次引入traindata
        if len(list(train_data[user]))==1:
            continue
        
        test