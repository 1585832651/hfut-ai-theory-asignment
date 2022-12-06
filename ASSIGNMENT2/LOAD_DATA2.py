import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import os
from torchvision.io import read_image

#不要用这个文件，这个文件是废弃的
class ImageSet(Dataset):
    def __init__(self,img_dir,num,labels=True,transform=None,target_transform=None):
        self.labels=[]
        self.img_dir=img_dir
        if num==560:
            self.img_dir0=[os.path.join((self.img_dir+"0/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(448)]
            self.img_dir1=[os.path.join((self.img_dir+"1/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(112)]
            if labels ==True:
               self.labels.append([0 for i in range(448)])  
               self.labels.append([1 for i in range(448,560)])  
  
        else :
            self.img_dir0=[os.path.join((self.img_dir+"0/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(448,560)]
            self.img_dir1=[os.path.join((self.img_dir+"1/"),
                                            ("".join(["(","%d" %(i+1),")",".jpg"])))for i in range(112,140)]
            if labels ==True:
               self.labels.append([0 for i in range(112)])  
               self.labels.append([1 for i in range(112,140)])
        self.img_path=self.img_dir0+self.img_dir1
        self.transform=transform
        self.target_transform=target_transform
        
            
        
    def __len__(self):
        return self.num
    
    def __getitem__(self, index) :
        imge_path=self.img_path
        image=read_image(imge_path[index])
        label=self.labels[index]
        return image,label
            
    