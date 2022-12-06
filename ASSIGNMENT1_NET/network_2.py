from tkinter import X
import numpy as np
from matplotlib import pyplot as plt
import load_data

from matplotlib.animation import FuncAnimation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_(x):
    return x * (1 - x)


class Network(object):
    def __init__(self, seed_num=7):
        # 构建 2 2 1 型神经网络
        # 随机生成参数
        np.random.seed(seed_num)
        self.w1 = np.random.randn(1, 20)
        self.b1 = np.random.randn(1,20)
        self.w2 = np.random.randn(20, 1)
        self.b2 = np.random.randn(1, 1)

    def forward(self, x):
        # 前向计算
        self.x2 = np.dot(x, self.w1) + self.b1
        # 激活
        self.x2 = sigmoid(self.x2)
        self.x3 = np.dot(self.x2, self.w2) + self.b2
        return self.x3

    def loss(self, y):
        # 使用均方误差作为损失函数
        error = self.x3 - y
        cost = np.sqrt(np.power(error, 2)/self.x3.shape[0])
        return np.sum(cost)

    def update(self, x, y, learn_rate):
        # 重复利用变量减少计算量，先考虑batch=1的情况，再推广
        gradient_b2 =  (self.x3 - y)/np.sqrt(np.power(self.x3 - y,2)*x.shape[0])
        gradient_w2 = gradient_b2 * self.x2
        gradient_b1 = gradient_b2 * self.w2.T * sigmoid_(self.x2)
        gradient_w1 = np.expand_dims(gradient_b1, axis=1) * np.expand_dims(x, axis=2)
        # 取batch的平均值作为更新梯度，并调整格式
        gradient_b2 = np.sum(gradient_b2, axis=0).reshape(self.b2.shape)
        gradient_w2 = np.sum(gradient_w2, axis=0).reshape(self.w2.shape)
        gradient_b1 = np.sum(gradient_b1, axis=0).reshape(self.b1.shape)
        gradient_w1 = np.sum(gradient_w1, axis=0).reshape(self.w1.shape)

        # 更新参数
        self.b2 -= learn_rate * gradient_b2
        self.w2 -= learn_rate * gradient_w2
        self.b1 -= learn_rate * gradient_b1
        self.w1 -= learn_rate * gradient_w1

    def train(self, x, y, iterations=50, batch_size=1, learn_rate=0.01):
        losses = []
        
        data_size = x.shape[0]
        # 每次取batch_size个训练
        print("X_train",X_train.shape)
        
        for i in range(iterations):
            for k in range(0, data_size, batch_size):
                mini_x = x[k: k + batch_size, :]
                mini_y = y[k: k + batch_size, :]
                self.forward(mini_x)
                l = self.loss(mini_y)
                self.update(mini_x, mini_y, learn_rate)
                losses.append(l)
                
            if i % 50 ==0:
                print("i=",i,"loss=",losses[-1],"learn_rate",learn_rate) 
                plt.clf()
                plt.scatter(X_train[:,0],network.forward(X_train))
                plt.plot()
                plt.scatter(X_train[:,0],Y_train)
                plt.plot()
                plt.pause(0.001)  # 暂停0.01秒
                plt.ioff()  
                learn_rate*=0.9995
            
                
        return losses


if __name__ == '__main__':
    Train_dir="train.txt"#相对路径
    Test_dir="test.txt"
    try:
        del X_train
        del Y_train
        
        print("数据已经加载过")
    except:
        pass
    
    X_train,Y_train=load_data.load_dataset_train(Train_dir)
    print(X_train.shape)
    #请不要反复执行这个代码块，不然会显示错误，如果想再执行一遍请重启ipynb文件
    print("before:",X_train.shape)
    try:
        X_size=X_train.shape[0]
        X_train=np.reshape(X_train,(X_size,1))
        Y_train=np.reshape(Y_train,(X_size,1))
        print("after:",X_train.shape)
        
    except:
        print("转换已经完成",X_train.shape)
    print("train_x",X_train.shape)
    print("train_y",X_train.shape)
    network = Network()
    iterations = 50000
    batch_size = 2
    learn_rate = 0.005
    #数据预处理
    Y_train=(Y_train-X_train)
    
    train_data=np.hstack((X_train,Y_train)) 
    train_data=train_data[train_data[:,0].argsort()]
    train_data=np.tile(train_data[295:],(5,1))
    print("train_data.shape",train_data.shape)
    X_train_s=(train_data[:,0])
    Y_train_s=(train_data[:,1])
    
    X_size=X_train_s.shape[0]
    X_train_s=(np.reshape(X_train_s,(X_size,1)))
    Y_train_s=np.reshape(Y_train_s,(X_size,1))
    
    
    X_train=np.vstack((X_train,X_train_s))
    Y_train=np.vstack((Y_train,Y_train_s))
    X_train=(X_train-np.mean(X_train))/5
    print("X_train",X_train.shape)
    print("Y_train",Y_train.shape)
    losses = network.train(X_train, Y_train, iterations, batch_size, learn_rate)
    predict_y = network.forward(X_train)+X_train
    plt.scatter(X_train[:,0],network.forward(X_train))
    plt.plot()
    plt.scatter(X_train[:,0],Y_train)
    plt.plot()
    plt.show()
