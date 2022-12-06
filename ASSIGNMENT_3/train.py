import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def train(train_data,all_user,all_item,step,iters,dimension=64):
#    userVec,itemVec=initVec(all_user,all_item,dimension)
    num_user=max(all_user)+1
    num_item=max(all_item)+1
    user_vector=np.random.uniform( 0,1,size=(num_user,dimension))#形成两个初始化的分接矩阵
    item_vector=np.random.uniform( 0,1,size=(num_item,dimension))
    lr = 0.005
    reg = 0.01
    train_user = np.array(range(0,num_user))
    L = []
    for i in range(iters):
        user= int(np.random.choice(train_user,1))
        visited=train_data[user]
        item_i=int(np.random.choice(visited,1))
        item_j=int(np.random.choice(all_item,1))
        while item_j in visited:
                item_j = int(np.random.choice(all_item,1))
        #这里的item_i,item_j使用来计算item_i-item_j的，这个i-j就相当于是
        #矩阵分解的一个监督数据，我们想要矩阵分解成user，item两个矩阵，然后再
        #合并得到的复原矩阵中满足：item_i-item_j和原先矩阵尽可能接近，所以以此为标准
        #进行之后的学习
        r_ui = np.dot(user_vector[user], item_vector[item_i].T) 
        r_uj = np.dot(user_vector[user], item_vector[item_j].T) 
        r_uij = r_ui - r_uj
        factor = 1.0 / (1 + np.exp(r_uij)) #根据论文公式进行sigmoid操作
        user_vector[user] += lr * (factor * (item_vector[item_i] - item_vector[item_j]) + reg * user_vector[user])
        item_vector[item_i] += lr * (factor * user_vector[user] + reg * item_vector[item_i])
        item_vector[item_j] += lr * (factor * (-user_vector[user]) + reg * item_vector[item_j])
        loss += (1.0 / (1 + np.exp(-r_uij)))
    loss += + reg * (
                np.power(np.linalg.norm(user_vector,ord=2),2) 
                + np.power(np.linalg.norm(item_vector,ord=2),2) 
                + np.power(np.linalg.norm(item_vector,ord=2),2)
                )
    print("loss=",loss)
    np.savetxt('./BPR/userVec.txt',user_vector,delimiter=',',newline='\n')
    np.savetxt('./BPR/itemVec.txt',item_vector,delimiter=',',newline='\n')