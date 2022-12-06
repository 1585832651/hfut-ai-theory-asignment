import numpy as np 
def load_dataset_train(file):
    """将txt文件转化为X,Y数据"""
    data=[line.split() for line in open(file,"r")]
    xs=[x[0] for x in data[:]]
    ys=[y[1] for y in data[:]]
    xs=np.array(xs)
    xs=xs.astype(float)
    ys=np.array(xs)
    ys=ys.astype(float)
    return xs,ys



def load_dataset_test(file):
    """将txt文件转化为X,Y数据"""
    data=[line.split() for line in open(file,"r")]
    xs=[x[0] for x in data[:]]
    xs=np.array(xs)
    xs=xs.astype(float)
    return xs
