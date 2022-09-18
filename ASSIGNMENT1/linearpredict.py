import numpy as np
from  RMSE import RMSE_loss
class linear_predictor():
    def __init__(self):
        """本来这里是有W,b两个参数的，但是写到一半突然发现对两个参数都做处理好像有点麻烦，
        所以采用了一种偏置技巧，把b放到W里面，然后把x全都加一个维度，数值为一"""
        self.W=None
        
    def train(self,
        x,
        y,
        learing_rate=2*1e-2,
        batch_size=5
        ):
        """"我们使用随机梯度下降法来进行拟合，
        线性拟合不用正则化，更不用去考虑正则化强度，
        学习率就先设为0.001，后面按照需要再改
        然后再加上来一波小批量样本训练的方法。
        虽然小批量训练的方式在数据量这么小的情况下只会影响训练效果，但是保险起见还是加上"""
        if self.W is None:
            self.W=2*np.random.rand(1,2)#rand是生成（0，1）的随机数，randn是正态分布的随机数，根据之前的图片肯定是选rand,
        #然后乘以二，这样既有可能生产大于一的W也有可能生成小于一的
        print(self.W)
        num_train=x.shape[0]
        num_iters=int(num_train/batch_size)
        loss_list=[]
        for iter in range(num_iters):
            X_batch=None
            Y_batch=None
            indexs=np.random.choice(range(num_train),batch_size,replace=False)
            X_batch=x[indexs]
            Y_batch=y[indexs]
            loss,grad=self.loss(X_batch,Y_batch)
            loss_list.append(loss)
            self.W+=learing_rate*grad

        return loss_list
            

    def loss(self,X_batch,Y_batch):
        """
        虽然只用均方根误差，但是为了提高代码复用性，还是把这个函数写成虚函数
        """
        return RMSE_loss(self.W,X_batch,Y_batch)


    def predict(self,X):
        """返回相应的Y值（numpy形式）"""
        y_pred=np.dot(X,self.W.T)
        return y_pred

    def jundgement(self,X,Y):
        loss=np.sqrt(np.sum((np.dot(X,self.W.T)-Y)**2)/X.shape[0])
        return loss