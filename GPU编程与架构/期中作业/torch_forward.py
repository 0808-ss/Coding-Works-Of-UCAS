import os

import torch
torch.set_printoptions(threshold=10_000000)  # 设置显示的最大元素数量
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import h5py
from tqdm import tqdm

# import provider
num_class = 10
total_epoch = 30
script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        # print("这是stnkd矩阵")
        # print(x[0][1])
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        # print("这是旋转矩阵")
        # print(trans)
        # print("输入矩阵形状")
        # print(x.shape)

        x = x.transpose(2, 1)
        # print(x.shape)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        # print("这是乘以偏置后的矩阵")
        # print(x[0][:100])
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        # print("转置后的矩阵")
        # print(x)


        x = F.relu(self.bn1(self.conv1(x)))

        # print(torch.count_nonzero(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        # print("对应res8的矩阵")
        # print(self.conv2(x)[0][0][:100])
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # print("这是res9")
        # print(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# 模型定义
class get_model(nn.Module):
    def __init__(self, k=10, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # 对应res_10
        # print("对应res_10的矩阵")
        # print(x)
        x = F.relu(self.bn1(self.fc1(x)))

        # print(x)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # print("这是最终矩阵")
        # print(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class PointCloudDataset(Dataset):
    def __init__(self,root, split):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split

        # with h5py.File(f"{split}_point_clouds.h5","r") as hf:
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5","r") as hf:
            for k in hf.keys():
                self.list_of_points.append(hf[k]["points"][:].astype(np.float32))
                self.list_of_labels.append(hf[k].attrs["label"])
        self.fix_length_statistics_with_median()

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

    def fix_length_statistics_with_median(self):
        lengths = [points.shape[0] for points in self.list_of_points]
        fix_length = int( np.median(lengths) )
        
        new_list_of_points = []
        for points in self.list_of_points:
            if(points.shape[0] >= fix_length):
                new_list_of_points.append(points[:fix_length, :])
            else:
                new_list_of_points.append(np.concatenate((points, np.zeros((fix_length - points.shape[0], 3), dtype=np.float32)), axis=0))
        self.list_of_points = new_list_of_points



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def read_h5_file(file_path):
    with h5py.File(file_path, "r") as file:
        # 假设数据存储在 'points' 数据集中
        points = file['points'][:]
        labels = [int(label) for label in file['labels'][:]]

        return points, labels

def load_model_params_and_buffers_from_txt(model, directory):
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    # 加载所有参数
    for name, param in model.named_parameters():
        file_path = os.path.join(directory, f'{name}.txt')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            data = torch.from_numpy(data).reshape(param.shape)
            param.data.copy_(data)
        else:
            raise FileNotFoundError(f"File {file_path} not found")
    
    # 加载所有缓冲区
    for name, buffer in model.named_buffers():
        file_path = os.path.join(directory, f'{name}.txt')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            data = torch.from_numpy(data).reshape(buffer.shape)
            buffer.data.copy_(data)
        else:
            raise FileNotFoundError(f"File {file_path} not found")
def main():
    # 实例化模型
    # 加载保存的参数
    model_path = '/home/course/LJY/project1/epoch30'
    classifier = get_model(10)
    print("模型创建完毕")
    load_model_params_and_buffers_from_txt(classifier, model_path)
    print("参数加载完毕")
    classifier.eval()
    data_path = '/home/course/LJY/project1/3D_MNIST'
    # 假设 PointCloudDataset 已经定义，并且 data_path 已经设置
    test_dataset = PointCloudDataset(root=data_path, split='test')
    # 创建 DataLoader 实例
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=False)
    print("获取测试集")
    # 获取第一个测试集样本
    # for data in test_dataloader:
    #     points, labels = data  # 假设返回的数据包含点云和标签
    #     # break  # 只处理第一个批次
    # points = points.transpose(1, 2)
    # # 将数据移动到模型所在的设备（如果使用GPU）
    # # points = points.to(classifier.device)
    # # labels = labels.to(classifier.device)

    # # 执行模型推理
    # with torch.no_grad():  # 不计算梯度，节省内存和计算资源
    #     outputs = classifier(points)
    top10_outputs = []
    for i, data in enumerate(test_dataloader):
        points, labels = data  # 获取点云和标签
        points = points.transpose(1, 2)  # 调整点云的维度

        # 将数据移动到模型所在的设备（如果使用GPU）
        # points = points.to(classifier.device)
        # labels = labels.to(classifier.device)

        # 执行模型推理，不计算梯度
        with torch.no_grad():
            outputs = classifier(points)

        # 将输出添加到列表中
        top10_outputs.append(outputs)

        # 如果已经处理了10个样本，就跳出循环
        if i + 1 == 1000:  # i + 1 因为 enumerate 是从0开始的
            break
    

    # 处理输出
    #_, predicted = torch.max(outputs, 1)  # 获取最大概率的预测标签
    #print(f"真实标签: {labels.item()}")
    #print(f"预测标签: {predicted.item()}")





if __name__ == '__main__':
    main()
