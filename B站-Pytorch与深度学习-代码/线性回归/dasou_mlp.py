import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


## 自己实现一个Dataset类去加载本地的CSV数据集
## 需要三个函数：init get_item len


class DasouDataset(Dataset):
    def __init__(self, filepath):  ## 加载原始数据集，并对特征数据和lable数据进行拆分
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  ## 根据索引返回单一样本数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  ## 返回长度
        return self.len

dataset = DasouDataset('./sigmoid.csv')

train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True) #num_workers 多线程


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, data in enumerate(train_loader):  # train_loader 是先shuffle后mini_batch
        inputs, labels = data## 获取数据
        y_pred = model(inputs)## 把数据喂进去给模型，获得结果
        loss = criterion(y_pred, labels)## 预测结果和真实值做损失函数
        print(epoch, i, loss.item())
        optimizer.zero_grad()## 梯度清零
        loss.backward()## 反向传播，更新参数

