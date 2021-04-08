import pandas as pd
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision


class MnistDataset(Dataset):

    def __init__(self, image_path, image_label, transform=None):
        super(MnistDataset, self).__init__()
        self.image_path = image_path  # 初始化图像路径列表
        self.image_label = image_label  # 初始化图像标签列表
        self.transform = transform  # 初始化数据增强方法

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        image = Image.open(self.image_path[index])
        image = np.array(image)
        label = int(self.image_label[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label)

    def __len__(self):
        return len(self.image_path)

    
def get_path_label(img_root, label_file_path):
    """
    获取数字图像的路径和标签并返回对应列表
    @para: img_root: 保存图像的根目录
    @para:label_file_path: 保存图像标签数据的文件路径 .csv 或 .txt 分隔符为','
    @return: 图像的路径列表和对应标签列表
    """
    data = pd.read_csv(label_file_path, names=['img', 'label'])
    data['img'] = data['img'].apply(lambda x: img_root + x)
    return data['img'].tolist(), data['label'].tolist()


# 获取训练集路径列表和标签列表
train_data_root = './mnist_data/train/'
train_label = './mnist_data/train.txt'
train_img_list, train_label_list = get_path_label(train_data_root, train_label)  
# 训练集dataset
train_dataset = MnistDataset(train_img_list,
                             train_label_list,
                             transform=transforms.Compose([transforms.ToTensor()]))

# 获取测试集路径列表和标签列表
test_data_root = './mnist_data/test/'
test_label = './mnist_data/test.txt'
test_img_list, test_label_list = get_path_label(test_data_root, test_label)
# 测试集sdataset
test_dataset = MnistDataset(test_img_list,
                            test_label_list,
                            transform=transforms.Compose([transforms.ToTensor()]))

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        ## 池化+batchnorm+relu激活+池化
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes) ## 全链接
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
