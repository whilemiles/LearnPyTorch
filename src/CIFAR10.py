import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 参数
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
random_seed = 1
torch.manual_seed(random_seed)
# 数据集加载
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# 模型加载
model = torchvision.models.densenet201()

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# torch.optim来做算法优化
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mode = model.to(device)

#模型训练
from torch.utils.tensorboard import SummaryWriter
os.makedirs("./events", exist_ok = True)
writer = SummaryWriter("./events/CIFAR10")
for epoch in range(n_epochs):
    for i,data in enumerate(train_loader):
        #取出数据及标签
        inputs,labels = data
        #数据及标签均送入GPU或CPU
        inputs,labels = inputs.to(device),labels.to(device)
        #前向传播
        outputs = model(inputs)
        #计算损失函数
        loss = criterion(outputs,labels)
        #清空上一轮的梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #参数更新
        optimizer.step()
        #利用tensorboard，将训练数据可视化
        if  i % 50 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)
        #print('it’s training...{}'.format(i))
    print('epoch{} loss:{:.4f}'.format(epoch+1,loss.item()))

#保存模型参数
torch.save(model,'./model/cifar10_densenet161.pt')
print('cifar10_densenet161.pt saved')

#模型加载
model = torch.load('./model/cifar10_densenet161.pt')
#测试
#model.eval()
model.train()

correct,total = 0,0
for j,data in enumerate(test_loader):
    inputs,labels = data
    inputs,labels = inputs.to(device),labels.to(device)
    #前向传播
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data,1)
    total = total + labels.size(0)
    correct = correct + (predicted == labels).sum().item()
    #准确率可视化
    if  j % 20 == 0:
        writer.add_scalar("Train/Accuracy", 100.0 * correct/total, j)
        
print('准确率：{:.4f}%'.format(100.0*correct/total))
