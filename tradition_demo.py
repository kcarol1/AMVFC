import numpy as np
from utils import metric,initialization_utils
import torch
from tradition_load_data import load_multimodal_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")
from mytuils import RecordResult
from fcm import MiniBatchFCM
import time
start_time = time.time()

initialization_utils.set_global_random_seed(seed=1024)

DEVICE = torch.device('cuda:0')
print(f'using {DEVICE}')

Dataset = 'Trento'
# Dataset = 'MUUFL'
# Dataset = 'Houston'

if Dataset == 'Trento':
    hsi_path = 'datasets/Trento/Trento-HSI.mat'
    lidar_path = 'datasets/Trento/Trento-Lidar.mat'
    gt_path = 'datasets/Trento/Trento-GT.mat'

    anchor = 4
    lr = 1e-2
    nb_comps = 10

if Dataset == 'MUUFL':
    hsi_path = 'datasets/MUUFL/MUUFLGfportHSI.mat'
    lidar_path = 'datasets/MUUFL/MUUFLGfportLiDAR_data_first_return.mat'
    gt_path = 'datasets/MUUFL/MUUFLGfportGT.mat'

    anchor = 7
    lr = 1e-4
    nb_comps = 8

if Dataset == 'Houston':
    hsi_path = 'datasets/Houston/HoustonHSI.mat'
    lidar_path = 'datasets/Houston/HoustonLidar.mat'
    gt_path = 'datasets/Houston/HoustonGT.mat'

    anchor = 7
    lr = 1e-4
    nb_comps = 2

image_path = (hsi_path, lidar_path)
modality_list, y = load_multimodal_data(gt_path, *image_path,is_labeled=False, nb_comps=nb_comps)


print(f'modality #1:{modality_list[0].shape}')
print(f'modality #2:{modality_list[1].shape}')
print(np.unique(y))

x = []
for i in range(len(modality_list)):
    x.append(modality_list[i].reshape([-1,modality_list[i].shape[2]]))
    print(f'rehape modality #{i+1}: {x[i].shape}')

indx_labeled = np.nonzero(y)[0]
y_labeled = y[indx_labeled]
class_num = len(np.unique(y)) - 1

print(f'class_num:{class_num}')
print(f'class:{np.unique(y)}')


for i in range(len(x)):
    x[i] = torch.from_numpy(x[i]).to(DEVICE)

A = []

fcm = MiniBatchFCM(n_clusters=anchor*class_num, batch_size=10000, mode='euclidean', device=DEVICE)

for i in range(len(x)):

    fcm.fit_predict(x[i])
    A.append(fcm.centroids.t())

    print(f'A #{i+1} shape(d*m): {A[i].shape}')
    
G = []
fcm = MiniBatchFCM(n_clusters=class_num, batch_size=10000, mode='euclidean', device=DEVICE)
for i in range(len(A)):

    _, U = fcm.fit_predict(A[i].t())
    G.append(U.t())
    print(f'G #{i+1} shape(m*k): {G[i].shape}')


X = torch.concat(x, dim=1)
print(X.shape)

fcm = MiniBatchFCM(n_clusters=class_num, batch_size=10000, mode='euclidean', device=DEVICE)
_, F = fcm.fit_predict(x[0])


print('-----------------------------------------------------')
print('hs的聚类结果')
y_pre = np.argmax(F.cpu().numpy(), axis=0)
y_pred_labeled = y_pre[indx_labeled]
acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                            nmi, ari,
                                                                                                            pur,bcubed_F))
print('-----------------------------------------------------')


fcm = MiniBatchFCM(n_clusters=class_num, batch_size=10000, mode='euclidean')
_, F = fcm.fit_predict(x[1])



print('lidar的聚类结果')
y_pre = np.argmax(F.cpu().numpy(), axis=0)
y_pred_labeled = y_pre[indx_labeled]
acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                            nmi, ari,
                                                                                                            pur,bcubed_F))
print('-----------------------------------------------------')


fcm = MiniBatchFCM(n_clusters=class_num, batch_size=10000, mode='euclidean', device=DEVICE)
_, F = fcm.fit_predict(X)

print(f'F shape:{F.shape}')


print('-----------------------------------------------------')
print('拼接之后聚类的结果')
y_pre = np.argmax(F.clone().cpu().numpy(), axis=0)
y_pred_labeled = y_pre[indx_labeled]
acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                            nmi, ari,
                                                                                                            pur,bcubed_F))
print('-----------------------------------------------------')

record = RecordResult()

with torch.no_grad():
    for e in range(200):
        
        epsilon = 1e-8
        
        # fix F, update Gv
        for j in range(1):
            for i in range(len(G)):
                
                G_grad = -2 * A[i].t() @ (x[i].t() - A[i]@G[i]@F) @ F.t()

                G[i] = G[i] - lr * G_grad
                G[i] = torch.clamp(G[i], min=epsilon)
                G[i] = G[i] / G[i].sum(axis=1, keepdims=True)

        for j in range(1):
                # fix G, F update
                F_grad = ((-2 * (G[0].T @ A[0].T) @ (x[0].t() - A[0] @ G[0] @ F)) +\
                    (-2 * (G[1].T @ A[1].T) @ (x[1].t() - A[1] @ G[1] @ F))) / 2.0
                F = F - lr * F_grad
                F = torch.clamp(F, min=epsilon)
                F = F / F.sum(axis=0, keepdim=True)
                
        y_pre = np.argmax(F.cpu().numpy(), axis=0)

        y_pred_labeled = y_pre[indx_labeled]
        acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
        print(e)
        print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                                nmi, ari,
                                                                                                                pur,bcubed_F))
        mse = torch.nn.MSELoss()
        print(f'loss: {mse(x[0].t(), A[0]@G[0]@F) + mse(x[1].t(), A[1]@G[1]@F)}')
        new_results = {
                'OA':float(round(acc, 4)),
                'Kappa':float(round(kappa, 4)),
                'NMI':float(round(nmi, 4)),
                'ARI':float(round(ari, 4)),
                'Purity':float(round(pur, 4)),
                'BCubed F':float(round(bcubed_F, 4)),
                'na':anchor,
                'lr':lr,
                'nb':nb_comps,
                'F':F.cpu().numpy(),
                'y_pred':y_pre,
                'G_hs':G[0].cpu().numpy(),
                'G_lidar':G[1].cpu().numpy(),
                'ca':ca,
            } 
        record.update(new_results, 'OA')

max_oa_metrics = record.get_best()
end_time = time.time()
max_oa_metrics['Time'] = end_time - start_time
print('------------------------------------------------------------------------------')
for key, value in max_oa_metrics.items():
    if key not in ['F', 'hsi', 'lidar','G_hs','G_lidar', 'ca', 'y_pred'] :
        print(f'{key} : {value}')
    elif key == 'ca':
        for i, ca_ in enumerate(value):
            print(f'class #{i} ACC: {ca_:.4f}')
print('------------------------------------------------------------------------------')
