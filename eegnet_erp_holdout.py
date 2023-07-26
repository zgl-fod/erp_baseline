from braindecode.models.eeginception_erp import EEGInceptionERP
import torch
# import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AdamW

# ------------------------------------------------------------------------------
smaller_n = 15000  # 间期数量下采样
negative_sample = 4800
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
n_chans = 64
n_classes = 2
batch_size = 128
lr, num_epochs = 0.0001, 1000
l1_lambda = 0.001
train_dir = r'D:\ZGL\ERP\train(S009+S011+S039)'
valid_dir = r'D:\ZGL\ERP\valid(S009+S011+S039)'
save_folder_model = r'valid(S009+S011+S039)\model_save_0.75s(187)'
# ----------------------------------------------------------------------------------
print('是否使GPU:{}'.format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EEGInceptionERP(n_chans, n_classes, input_window_samples=187)
# print(model)
model.to(device)
print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))


class MyDataSet(Dataset):
    def __init__(self, data_dir):
        """
        """
        self.label_name = rmb_label
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有文本路径和标签，在DataLoader中通过index读取样本

    def __getitem__(self, index):
        path_txt, label = self.data_info[index]
        data = np.load(path_txt)
        return data, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.npy'), img_names))
                number_inter = len(img_names)
                if number_inter != negative_sample:
                    inter_index = np.random.choice(np.arange(number_inter), size=smaller_n, replace=False)  # 间期数量下采样
                else:
                    inter_index = np.random.choice(np.arange(number_inter), size=negative_sample, replace=False)  # 下采样到和前期数量一致
                for index in inter_index:
                    img_name = img_names[index]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


class My_val_DataSet(Dataset):
    def __init__(self, data_dir):
        """
        """
        self.label_name = rmb_label
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有文本路径和标签，在DataLoader中通过index读取样本

    def __getitem__(self, index):
        path_txt, label = self.data_info[index]
        data = np.load(path_txt)
        return data, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.npy'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


def evaluate_accuracy(data_iter, net, device_acc=device):
    all_predict_cla = []
    all_y = []
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                X = X.to(torch.float32)
                predict_cla = net(X.to(device_acc)).argmax(dim=1).cpu().numpy()
                net.train()  # 改回训练模式
            all_predict_cla.extend(predict_cla)
            all_y.extend(y.cpu().numpy())
        TN, FP, FN, TP = confusion_matrix(all_y, all_predict_cla).ravel()
        acc = (TP + TN) / (TN + FP + FN + TP)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        if (TP + FN) == 0:
            recall = 0
            print('模型将所有的信号均预测为间期信号')
        if (TN + FP) == 0:
            specificity = 0
            print('模型将所有的信号均预测为前期信号')
        FDR = FP
    return acc, recall, specificity, FDR


def train_ch5(net, train_iter_ch5, test_iter_ch5, optimizer_ch5, device_ch5, num_epochs_ch5):
    net = net.to(device_ch5)
    print("training on ", device_ch5)
    # loss = torch.nn.CrossEntropyLoss()
    loss = BCEFocalLoss()
    save_recall = 0
    save_acc = 0
    batch_count = 0
    data_records = []
    os.makedirs(save_folder_model, exist_ok=True)
    for epoch in range(num_epochs_ch5):
        train_l_sum, train_acc_sum, start = 0.0, 0.0, time.time()
        all_train_cla = []
        all_y = []
        for X, y in tqdm(train_iter_ch5):
            X = X.to(device_ch5)
            y = y.to(device_ch5)
            X = X.to(torch.float32)
            optimizer_ch5.zero_grad()
            y_hat = net(X)
            l_1 = loss(y_hat, y)
            # # 添加L1正则化项
            # l1_regularization = torch.tensor(0.)
            # for param in model.parameters():
            #     l1_regularization += torch.norm(param, p=1)  # 计算参数的L1范数
            # loss += l1_lambda * l1_regularization.item()  # 将正则化项添加到损失函数
            l_1.backward()
            optimizer_ch5.step()
            train_l_sum += l_1.cpu().item()
            all_train_cla.extend(y_hat.argmax(dim=1).cpu().numpy())
            all_y.extend(y.cpu().numpy())
            batch_count += 1
        train_acc_sum = accuracy_score(all_y, all_train_cla)
        test_acc, test_recall, test_specificity, FDR = evaluate_accuracy(test_iter_ch5, net)
        data_record = 'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test recall %.3f test specificity %.3f  FDR %d time %.1f sec' \
                      % (epoch + 1, train_l_sum / batch_count, train_acc_sum, test_acc, test_recall, test_specificity, FDR, time.time() - start)
        if save_acc < test_acc or save_acc == 1 or save_recall < test_recall or test_recall > 0.8:
            if save_acc < test_acc:
                save_acc = test_acc
            else:
                save_recall = test_recall
            data_records.append(data_record)
            save_path = os.path.join(save_folder_model, 'epoch%d_trainacc%ftestacc%f_testrecall%f_testspecificity%f_FDR%d_%d-%dnet.pth' % (epoch + 1, train_acc_sum, test_acc, test_recall, test_specificity, FDR, len_train_sample, len_valid_sample))
            torch.save(net.state_dict(), save_path)
        print(data_record)
    with open(os.path.join(save_folder_model, '1.json'), 'w') as f:
        json.dump(data_records, f, indent=1)


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.95, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        pt = pt[:, 1]
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


if __name__ == '__main__':
    # 构建MyDataset实例
    rmb_label = {"0": 0, "1": 1}
    train_data = MyDataSet(data_dir=train_dir)
    valid_data = My_val_DataSet(data_dir=valid_dir)
    len_train_sample = len(train_data)
    len_valid_sample = len(valid_data)
    # 构建DataLoder
    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=valid_data, batch_size=batch_size)
    # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l1_lambda)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=l1_lambda)
    train_ch5(model, train_iter, test_iter, optimizer, device, num_epochs)

