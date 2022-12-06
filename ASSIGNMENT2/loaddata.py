from locale import normalize
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.nn as nn

class ImageSet_train(Dataset):
    def __init__(self,img_dir,num,labels=True,transform=None,target_transform=None):
        self.labels=[]
        self.num=num
        self.img_dir=img_dir
        if num==896:
            #这里把负样本复制了几次使得正负样本数量相等
            self.img_dir0=[os.path.join((self.img_dir+"0/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(448)]
            self.img_dir1=[os.path.join((self.img_dir+"1/"),
                                            ("".join(["(","%d" %((i)%112+1),")",".jpg"])))for i in range(448)]
            if labels ==True:
               self.labels0=([0 for i in range(448)])  
               self.labels1=([1 for i in range(448)])  
               self.labels=self.labels0+self.labels1
  
        else :
            self.img_dir0=[os.path.join((self.img_dir+"0/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(448,560)]
            self.img_dir1=[os.path.join((self.img_dir+"1/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(112,140)]
            if labels ==True:
               self.labels0=([0 for i in range(112)])  
               self.labels1=([1 for i in range(28)]) 
                
               self.labels=self.labels0+self.labels1
        self.img_path=self.img_dir0+self.img_dir1
        self.transform=transform
        self.target_transform=target_transform
        
        
            
        
    def __len__(self):
        return self.num
    
    def __getitem__(self, index) :
        imge_path=self.img_path
        image=read_image(imge_path[index]).to(torch.float32)
        label=self.labels[index]
        return image,label
            
            
            
class ImageSet_test(Dataset):
    def __init__(self,img_dir,labels=None,transform=None,target_transform=None):
        self.labels=[]
        self.img_dir=img_dir
        self.img_dir=[os.path.join((self.img_dir),
                                        ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(200)]
        
        self.img_path=self.img_dir
        self.transform=transform
        self.target_transform=target_transform
        
            
        
    def __len__(self):
        return 200
    
    def __getitem__(self, index) :
        imge_path=self.img_path
        image=read_image(imge_path[index]).to(torch.float32)

        return image
            
    