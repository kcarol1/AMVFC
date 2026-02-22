import numpy as np
from utils import metric,initialization_utils
import torch
from net import nnet, Encoder, Decoder, load_multimodal_data, band_encoder, CustomDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")
from mytuils import RecordResult
from tqdm import tqdm
from torch.utils.data import DataLoader

import time
start_time = time.time()

initialization_utils.set_global_random_seed(seed=1024)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


Dataset = 'Trento'
# Dataset = 'MUUFL'
# Dataset = 'Houston'

if Dataset == 'Trento':
    hsi_path = 'datasets/Trento/Trento-HSI.mat'
    lidar_path = './datasets/Trento/Trento-Lidar.mat'
    gt_path = './datasets/Trento/Trento-GT.mat'

    k = 2 
    epoch = 100
    nb_comps = 6 
    lr = 0.01

if Dataset == 'MUUFL':
    hsi_path = './datasets/MUUFL/MUUFLGfportHSI.mat'
    lidar_path = './datasets/MUUFL/MUUFLGfportLiDAR_data_first_return.mat'
    gt_path = './datasets/MUUFL/MUUFLGfportGT.mat'

    k = 1 
    epoch = 100
    nb_comps = 6
    lr = 0.6

if Dataset == 'Houston':
    hsi_path = './datasets/Houston/HoustonHSI.mat'
    lidar_path = './datasets/Houston/HoustonLidar.mat'
    gt_path = './datasets/Houston/HoustonGT.mat'
    
    k = 1
    epoch = 100
    nb_comps = 4
    lr = 0.08



img_path = (hsi_path, lidar_path)
x, gt = load_multimodal_data(gt_path, *img_path, is_labeled=False, nb_comps=nb_comps, device=DEVICE)
spatial_size = gt.shape
y = gt.reshape([-1,])


indx_labeled = np.nonzero(y)[0]
y_labeled = y[indx_labeled]
class_num = len(np.unique(y)) - 1
print('# classes:', class_num)
print(DEVICE)
samples = x[0].shape[0]*x[0].shape[1]
anchor_num = k * class_num


dataset = CustomDataset(x)
dataloader_train = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

encoder = Encoder(x[0].shape[2])
decoder = Decoder(x[0].shape[2])
band = band_encoder(x[1].shape[2], nb_comps)

model = nnet(samples, anchor_num, DEVICE, class_num, encoder, decoder, band).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
record = RecordResult()
progress_bar = tqdm(total=epoch, desc="Training")
for e in range(epoch):
    for step, (x0, x1) in enumerate(dataloader_train):
        x0 = torch.tensor(x0).to(DEVICE)
        x1 = torch.tensor(x1).to(DEVICE)
        x0 = x0.permute(0, 3, 1, 2)
        x1 = x1.permute(0, 3, 1, 2)
        tl = [x0, x1]
        
        optimizer.zero_grad()
        A0, G0, h0, x_hat0, A1, G1, h1, x_hat1, x_1, F = model(tl)

        tl = [x0, x_1]
        A = [A0, A1]
        G = [G0, G1]
        h = [h0, h1]
        x_hat = [x_hat0, x_hat1]
        
        loss = model.the_loss(tl, A, G, F, h, x_hat)

        loss.backward()
        optimizer.step()

        F = model.F.detach().cpu().numpy()
        y_pre = np.argmax(F, axis=0)
        y_pred_labeled = y_pre[indx_labeled]
        y_pred_2D = y_pre.reshape(gt.shape)
        acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
        print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                                    nmi, ari,
                                                                                                                    pur,bcubed_F))
        new_results = {
                'OA':float(round(acc, 4)),
                'Kappa':float(round(kappa, 4)),
                'NMI':float(round(nmi, 4)),
                'ARI':float(round(ari, 4)),
                'Purity':float(round(pur, 4)),
                'BCubed F':float(round(bcubed_F, 4)),
                'ca':ca,
                'F':F,
                'y_pred':y_pre,
                'hsi_feat':h0.detach().cpu().numpy(),
                'lidar_feat':h1.detach().cpu().numpy(),
            } 
        record.update(new_results, 'OA')
        progress_bar.update(1)
max_oa_metrics = record.get_best()
end_time = time.time()
max_oa_metrics['Time'] = end_time - start_time
print('------------------------------------------------------------------------------')
for key, value in max_oa_metrics.items():
    if key not in ['F', 'y_pred', 'hsi_feat', 'lidar_feat', 'ca']:
        print(f'{key} : {value}')
    elif key == 'ca':
        for i, ca_ in enumerate(value):
            print(f'class #{i} ACC: {ca_:.4f}')
print('------------------------------------------------------------------------------')


