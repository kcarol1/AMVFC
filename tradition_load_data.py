from sklearn.discriminant_analysis import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from utils.Preprocessing import Processor
def load_multimodal_data(gt_path, *src_path, patch_size=(7, 7), is_labeled=True, nb_comps):
    p = Processor()
    n_modality = len(src_path)
    modality_list = []
    in_channels = []
    for i in range(n_modality):
        img, gt = p.prepare_data(src_path[i], gt_path)
        if len(img.shape) == 2:
            img = img.reshape([img.shape[0], img.shape[1], 1])

        n_row, n_col, n_channel = img.shape
        if n_channel > nb_comps:
            pca = PCA(n_components=nb_comps)
            img = pca.fit_transform(img.reshape(n_row*n_col, n_channel)).reshape((n_row, n_col, nb_comps))
            print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))    
        
        n_samples = n_row * n_col
        img_2d = img.reshape([-1,img.shape[2]]) 
        y = gt.reshape([-1,])
        
        
        scaler = StandardScaler()
        batch_size = 5000
       


        for start_id in range(0, img_2d.shape[0], batch_size):
            batch = img_2d[start_id: start_id + batch_size]
            scaler.partial_fit(batch)


        for start_id in range(0, img_2d.shape[0], batch_size):
            batch = img_2d[start_id: start_id + batch_size]
            img_2d[start_id: start_id + batch_size] = scaler.transform(batch)


        img = img_2d.reshape(img.shape)


        modality_list.append(img)
        in_channels.append(n_channel)
    return modality_list, y

