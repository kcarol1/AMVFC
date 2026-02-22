import torch
import numpy as np
from sklearn.decomposition import PCA
from utils.Preprocessing import Processor
from torch.utils.data import Dataset
from fcm import FuzzyCMeansTorch

class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
     
        self.conv1 = torch.nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding='same')  
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same') 
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same') 
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same') 
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        
        
        self.bn1 = torch.nn.BatchNorm2d(16) 
        self.bn2 = torch.nn.BatchNorm2d(32) 
        self.bn3 = torch.nn.BatchNorm2d(64) 
        self.bn4 = torch.nn.BatchNorm2d(128) 
        self.bn5 = torch.nn.BatchNorm2d(128) 

        
        self.rl = torch.nn.ReLU()
#==========================================================================================#       


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.rl(x)
        
        
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.rl(x)
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)  # (3, 3) -> (5, 5)
        self.deconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1) # (5, 5) -> (7,7)  
        self.deconv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1) # (7, 7) -> (9, 9)
        self.deconv4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1) # (9, 9) -> (11, 11)  
        self.deconv5 = torch.nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1) # (11, 11) -> (13, 13)  

        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(16)
        
        self.rl = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        

    def forward(self, x):
        # x = self.deconv1(x)
        # x = self.bn1(x)
        # x = self.rl(x)
        
        # x = self.deconv2(x)
        # x = self.bn2(x)
        # x = self.rl(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        
        x = self.deconv5(x)

        return x

class band_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(band_encoder, self).__init__()
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.rl = torch.nn.ReLU()
    def forward(self, x):
        n, b, h, w, = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((h*w, b))
        
        x = self.linear(x)
        x = self.bn(x)
        # x = self.rl(x)
        
        x = x.reshape(n, h, w, self.out_channels)
        x = x.permute(0, 3, 1, 2)
        return x

class nnet(torch.nn.Module):
    def __init__(self, n, anchor_num, device, n_clusters, Encoder, Decoder, band_encoder):
        super(nnet, self).__init__()
        self.n_clusters = n_clusters
        self.device = device
        self.proj1 = torch.randn((n, anchor_num), device=self.device)

        self.proj2 = torch.nn.Parameter(torch.randn((anchor_num, n_clusters), device=self.device))
        
        self.bn1 = torch.nn.BatchNorm1d(anchor_num)
        self.bn2 = torch.nn.BatchNorm1d(n_clusters)
        self.fcm = FuzzyCMeansTorch(n_clusters, device=device)
        self.mse = torch.nn.MSELoss()
        self.band_encoder = band_encoder
        
        # self.A = torch.nn.Parameter(torch.zeros((n, anchor_num), device=self.device))
        self.G0 = torch.nn.Parameter(torch.zeros((anchor_num, n_clusters), device=self.device))
        self.G1 = torch.nn.Parameter(torch.zeros((anchor_num, n_clusters), device=self.device))
        self.F = torch.nn.Parameter(torch.ones((n_clusters, n), device=self.device))
        
        self.encoder = Encoder
        self.decoder = Decoder

        
        
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
    def getanchor(self, x):
        
        n, b, h, w = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((h*w, b))
        x = x.t() @ self.proj1
        x = self.bn1(x)
        
        return x.t()
    
    def forward(self, x):
        
        x[1] = self.band_encoder(x[1])
        h0 = self.encoder(x[0])
        A0 = self.getanchor(h0)
        if torch.sum(self.G0) == 0:
            fcm = FuzzyCMeansTorch(self.n_clusters, device=self.device)
            fcm.fit(A0)
            G0 = fcm.get_membership(A0)
            self.G0.data.copy_(G0)
        x_hat0 = self.decoder(h0)
        fcm = None
        
        h1 = self.encoder(x[1])
        A1 = self.getanchor(h1)
        if torch.sum(self.G1) == 0:
            fcm = FuzzyCMeansTorch(self.n_clusters, device=self.device)
            fcm.fit(A1)
            G1 = fcm.get_membership(A1)
            self.G1.data.copy_(G1)
        x_hat1 = self.decoder(h1)
        del fcm

        return A0, self.G0, h0, x_hat0, A1, self.G1, h1, x_hat1, x[1], self.F

    def the_loss(self, X, A, G, F, h_, x_hat):
        
        for i in range(len(X)):
            n, b, h, w = X[i].shape
            X[i] = X[i].permute(0, 2, 3, 1)
            X[i] = X[i].reshape((h*w, b))
            
        for i in range(len(x_hat)):
            n, b, h, w = x_hat[i].shape
            x_hat[i] = x_hat[i].permute(0, 2, 3, 1)
            x_hat[i] = x_hat[i].reshape((h*w, b))
        
        for i in range(len(h_)):
            n, b, h, w = h_[i].shape
            h_[i] = h_[i].permute(0, 2, 3, 1)
            h_[i] = h_[i].reshape((h*w, b))
        
        s1 = self.mse(X[0], x_hat[0]) + self.mse(X[1], x_hat[1])
        # s1 = 0
        # s2 = (
        #     self.mse(h_[0].t(), A[0].t() @ G[0] @ F) + 
        #     self.mse(h_[1].t(), A[1].t() @ G[1] @ F)
        #     )
        s2 = (
            self.mse(h_[0].t(), A[0].t() @ torch.nn.functional.softmax(G[0], dim=1) @ torch.nn.functional.softmax(F, dim=0)) + 
            self.mse(h_[1].t(), A[1].t() @ torch.nn.functional.softmax(G[1], dim=1) @ torch.nn.functional.softmax(F, dim=0))
            )
        
        loss = s1 + s2
        
        return loss

def load_multimodal_data(gt_path, *src_path, is_labeled=True, nb_comps, device):
    p = Processor()
    n_modality = len(src_path)
    modality_list = []
    in_channels = []
    
    # img, gt = p.prepare_data(src_path[1], gt_path)
    # nb = img.shape[2]
    
    for i in range(n_modality):
        img, gt = p.prepare_data(src_path[i], gt_path)
        # img, gt = img[:, :100, :], gt[:, :100]
        n_row, n_column, n_band = img.shape
        
        modality_list.append(img)
        
    n_row, n_column, n_band = modality_list[0].shape    
    pca = PCA(n_components=nb_comps)
    modality_list[0] = pca.fit_transform(modality_list[0].reshape(n_row*n_column, n_band)).reshape((n_row, n_column, nb_comps))
    print('pca shape: %s, percentage: %s' % (modality_list[0].shape, np.sum(pca.explained_variance_ratio_)))    

    
            
    return modality_list, gt

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data1 = data_list[0].reshape((1, data_list[0].shape[0], data_list[0].shape[1], data_list[0].shape[2]))
        self.data2 = data_list[1].reshape((1, data_list[1].shape[0], data_list[1].shape[1], data_list[1].shape[2]))
        self.length = self.data1.shape[0]
  
    def __len__(self):
        """
        返回数据集的总长度
        """
        return self.length
    
    def __getitem__(self, idx):
        """
        根据索引获取一个样本
        """
        # 取出第idx个样本
        sample1 = self.data1[idx]
        sample2 = self.data2[idx]
        
        # 返回一个字典或者元组，包含两个样本
        return (torch.tensor(sample1, dtype=torch.float32),
                torch.tensor(sample2, dtype=torch.float32))