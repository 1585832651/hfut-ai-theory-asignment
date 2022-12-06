import torch
import torch.nn as nn
import torch
import torch.utils.data
import torch
from loaddata import ImageSet_test
from loaddata import ImageSet_train

#加载训练集
data_dim_train="drive-download-20221002T020216Z-001/train/"
data_dim_test="drive-download-20221002T020216Z-001/test/"

dataloader_train=ImageSet_train(data_dim_train,560)

#截取验证集
dataloader_val=ImageSet_train(data_dim_train,140)
#加载训练集
dataloader_test=ImageSet_test(data_dim_test,200)

#所有数据准备完毕，开始搭建神经网络


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.AvgPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.AvgPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.AvgPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
net1=Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer1=optim.Adam(net1.parameters(),lr=3e-6,weight_decay=1e-4)
trainloader=torch.utils.data.DataLoader(dataloader_train,batch_size=1,
                                        shuffle=False)
valLoader=torch.utils.data.DataLoader(dataloader_val,batch_size=1,
                                        shuffle=False)
classes=(1,0)
net1.to('cuda:0')
loss_history=[]
for epoch in range(60):  # loop over the dataset multiple times
    running_loss1 = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer1.zero_grad()
        outputs1 = net1(inputs)
        loss1 = criterion(outputs1, labels)
        loss1.backward()
        optimizer1.step()
        running_loss1 += loss1.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss1 / 20:.3f}')
        running_loss1 =0.0
        correct = 0
        total = 0
    #每个周期都在验证集上跑一次用于观察，这一段可以删掉
    with torch.no_grad():
        for data in valLoader:
            predicted=[]
            images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs1 = net1(images)
            if(torch.sigmoid(outputs1[0][0])>torch.sigmoid(outputs1[0][1])):
                predicted=(torch.tensor([0]))
            else:
                predicted=(torch.tensor([1]))
            predicted,labels=predicted.to(device),labels.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 100 VAL images: {100 * correct // total} %') 
        running_loss1 = 0.0

print('Finished Training')


correct = 0
total = 0
#预测模块，无梯度下降，predicted的值就是预测结果，打标签就要用这一块
with torch.no_grad():
    for data in valLoader:
        predicted=[]
        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs1 = net1(images)
        if(torch.sigmoid(outputs1[0][0])>torch.sigmoid(outputs1[0][1])):
            predicted=(torch.tensor([0]))
        else:
            predicted=(torch.tensor([1]))
        # the class with the highest energy is what we choose as prediction
        predicted,labels=predicted.to(device),labels.to(device)
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 100 test images: {100 * correct // total} %')