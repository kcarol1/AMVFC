import torch
import numpy as np

class FuzzyCMeansTorch:
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5, device='cuda'):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.centers_ = None
        self.membership_ = None

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()


        X = X.to(self.device)
        n_samples = X.shape[0]
        U = torch.rand((n_samples, self.n_clusters), device=self.device)
        U = U / U.sum(dim=1, keepdim=True)

        for _ in range(self.max_iter):
            um = U ** self.m
            centers = (um.T @ X) / um.sum(dim=0)[:, None]  # shape: [n_clusters, n_features]
            dist = torch.cdist(X, centers, p=2) + 1e-8

            tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
            U_new = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)

            if torch.norm(U_new - U) < self.error:
                break
            U = U_new

        self.centers_ = centers
        self.membership_ = U
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        dist = torch.cdist(X, self.centers_, p=2) + 1e-8
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)
        return torch.argmax(U, dim=1)

    def get_membership(self, X=None):

        if X is None:
            return self.membership_
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        dist = torch.cdist(X, self.centers_, p=2) + 1e-8
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)
        return U



class fcm_rbf:
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5, sigma=1.0, device='cuda'):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.sigma = sigma  
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.centers_ = None
        self.membership_ = None

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        X = X.to(self.device)
        n_samples = X.shape[0]
        U = torch.rand((n_samples, self.n_clusters), device=self.device)
        U = U / U.sum(dim=1, keepdim=True)

        for _ in range(self.max_iter):

            centers = (U.T @ X) / U.sum(dim=0)[:, None]  # 计算当前聚类中心
            

            dist = self.gaussian_kernel(X, centers)
            

            tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
            U_new = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)

            if torch.norm(U_new - U) < self.error:
                break
            U = U_new

        self.centers_ = centers
        self.membership_ = U
        return self

    def gaussian_kernel(self, X, centers):

        dist_sq = torch.sum((X[:, None, :] - centers[None, :, :])**2, dim=2)
        return torch.exp(-dist_sq / (2 * self.sigma**2))

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        dist = self.gaussian_kernel(X, self.centers_)
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)
        return torch.argmax(U, dim=1)

    def get_membership(self, X=None):

        if X is None:
            return self.membership_
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        dist = self.gaussian_kernel(X, self.centers_)
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (2 / (self.m - 1)), dim=2)
        return U
    
    """
    exmaple:
    fcm = FuzzyCMeansTorch(n_clusters=3, m=2.0, max_iter=150, error=1e-5, sigma=1.0)
    fcm.fit(X) 
    membership = fcm.get_membership() 
    labels = fcm.predict(X)
    """
    
    
    
class multi_view_fcm:
    def __init__(self, n_clusters=3, m=2.0, max_iter=1000, error=1e-5, device='cuda'):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.centers_ = None  # list of [n_clusters, n_features] per view
        self.membership_ = None  # shared [n_samples, n_clusters]

    def fit(self, X_views):
        """
        X_views: list of torch.Tensor or np.ndarray, each of shape [n_samples, n_features_view]
        """
        if isinstance(X_views, np.ndarray):
            raise ValueError("Expected list of views, got single ndarray")

        X_views = [torch.from_numpy(X).float().to(self.device) if isinstance(X, np.ndarray) else X.to(self.device)
                   for X in X_views]

        n_samples = X_views[0].shape[0]
        V = len(X_views)  # number of views

        U = torch.rand((n_samples, self.n_clusters), device=self.device)
        U = U / U.sum(dim=1, keepdim=True)

        for _ in range(self.max_iter):
            U_old = U.clone()

            # update centers per view
            centers_list = []
            um = U ** self.m
            for v in range(V):
                Xv = X_views[v]
                centers_v = (um.T @ Xv) / um.sum(dim=0)[:, None]  # [n_clusters, n_features_v]
                centers_list.append(centers_v)

            # update U using all views
            dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
            for v in range(V):
                dist += torch.cdist(X_views[v], centers_list[v], p=2) ** 2

            dist = dist + 1e-8  # avoid zero division
            tmp = dist.unsqueeze(2) / dist.unsqueeze(1)  # [n_samples, n_clusters, n_clusters]
            U = 1.0 / torch.sum(tmp ** (1 / (self.m - 1)), dim=2)

            if torch.norm(U - U_old) < self.error:
                break

        self.centers_ = centers_list
        self.membership_ = U
        return self

    def predict(self, X_views):
        X_views = [torch.from_numpy(X).float().to(self.device) if isinstance(X, np.ndarray) else X.to(self.device)
                   for X in X_views]
        n_samples = X_views[0].shape[0]
        V = len(X_views)

        dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
        for v in range(V):
            dist += torch.cdist(X_views[v], self.centers_[v], p=2) ** 2

        dist = dist + 1e-8
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (1 / (self.m - 1)), dim=2)
        return torch.argmax(U, dim=1)

    def get_membership(self, X_views=None):
        if X_views is None:
            return self.membership_

        X_views = [torch.from_numpy(X).float().to(self.device) if isinstance(X, np.ndarray) else X.to(self.device)
                   for X in X_views]

        n_samples = X_views[0].shape[0]
        V = len(X_views)
        dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
        for v in range(V):
            dist += torch.cdist(X_views[v], self.centers_[v], p=2) ** 2

        dist = dist + 1e-8
        tmp = dist.unsqueeze(2) / dist.unsqueeze(1)
        U = 1.0 / torch.sum(tmp ** (1 / (self.m - 1)), dim=2)
        return U


import numpy as np
import torch
import torch.nn.functional as F

class AnchorMultiViewFCM:
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5, device='cuda'):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.membership_ = None  # U: [N, C]
        self.anchor_coeffs_ = []  # list of G^{nu}: [M, C]
        self.anchors_ = []  # list of A^{nu}: [d, M]

    def fit(self, X_views, A_views):
        """
        X_views: list of tensors [X^1, X^2, ..., X^V], each shape [N, d_v]
        A_views: list of anchor matrices [A^1, A^2, ..., A^V], each shape [d_v, M]
        """
        V = len(X_views)
        N = X_views[0].shape[0]

        # Move all data to device
        X_views = [x.to(self.device) for x in X_views]
        A_views = [a.to(self.device) for a in A_views]
        self.anchors_ = A_views

        M = A_views[0].shape[1]
        C = self.n_clusters

        # Initialize U
        U = torch.rand((N, C), device=self.device)
        U = U / U.sum(dim=1, keepdim=True)

        # Initialize G^{nu} (anchor coefficients)
        G_views = [F.softmax(torch.rand((M, C), device=self.device), dim=0) for _ in range(V)]

        for _ in range(self.max_iter):
            U_old = U.clone()

            # === Update cluster centers ===
            for v in range(V):
                A = A_views[v]  # [d, M]
                X = X_views[v]  # [N, d]

                # Update G^v: solve convex problem per cluster
                G_new = []
                for j in range(C):
                    # Solve: min_g ∑_i u_ij^m * ||x_i - Ag||^2, s.t. g >= 0, sum(g)=1
                    u_m = U[:, j] ** self.m  # [N]
                    W = torch.diag(u_m)  # [N, N]
                    XA = A.T @ (X.T @ W)  # [M, 1]
                    AA = A.T @ A  # [M, M]

                    # Closed-form: g = argmin_g g^T A^T A g - 2g^T A^T x
                    # Use projected gradient descent (simplified here)
                    g = torch.rand((M,), device=self.device)
                    g = g / g.sum()
                    g.requires_grad = True

                    optimizer = torch.optim.Adam([g], lr=0.01)
                    for _ in range(30):
                        optimizer.zero_grad()
                        recon = A @ g  # [d]
                        loss = (u_m[:, None] * ((X - recon) ** 2)).sum()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            g.data.clamp_(min=1e-6)
                            g.data /= g.data.sum()

                    G_new.append(g.detach())

                G_views[v] = torch.stack(G_new, dim=1)  # [M, C]

            # === Update membership matrix U ===
            dist_total = torch.zeros((N, C), device=self.device)
            for v in range(V):
                A = A_views[v]        # [d, M]
                G = G_views[v]        # [M, C]
                X = X_views[v]        # [N, d]
                V_centers = (A @ G)   # [d, C]
                dists = torch.cdist(X, V_centers.T)  # [N, C]
                dist_total += dists ** 2

            tmp = dist_total.unsqueeze(2) / dist_total.unsqueeze(1)  # [N, C, C]
            U = 1.0 / torch.sum(tmp ** (1 / (self.m - 1) + 1e-8), dim=2)

            # Check convergence
            if torch.norm(U - U_old) < self.error:
                break

        self.membership_ = U
        self.anchor_coeffs_ = G_views
        return self

    def predict(self):
        return torch.argmax(self.membership_, dim=1).cpu().numpy()

    def get_membership(self):
        return self.membership_.detach().cpu().numpy()

    def get_centers(self):
        centers = []
        for A, G in zip(self.anchors_, self.anchor_coeffs_):
            centers.append((A @ G).detach().cpu().numpy())  # shape: [d, C]
        return centers


import torch
import numpy as np

class MiniBatchFCM:
    def __init__(self, n_clusters, m=2.0, max_iter=100, batch_size=512, tol=1e-4,
                 mode='euclidean', verbose=0, device='cuda'):
        self.n_clusters = n_clusters
        self.m = m  # fuzzifier
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.mode = mode
        self.verbose = verbose
        self.centroids = None
        self.device = device

    def _distance(self, X, centroids):
        if self.mode == 'euclidean':
            return torch.cdist(X, centroids, p=2)
        elif self.mode == 'cosine':
            Xn = torch.nn.functional.normalize(X, dim=1)
            Cn = torch.nn.functional.normalize(centroids, dim=1)
            return 1.0 - Xn @ Cn.T
        else:
            raise ValueError("Unsupported distance mode")

    def _compute_membership(self, dist):
        dist = dist.clamp(min=1e-8)  # avoid division by zero
        inv_dist = dist.pow(-2 / (self.m - 1))
        return inv_dist / inv_dist.sum(dim=1, keepdim=True)

    def fit_predict(self, X):
        assert isinstance(X, torch.Tensor)
        X = X.to(self.device)
        n_samples, n_features = X.shape

        # Init centroids from random samples
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices].clone()
        self.counts = torch.ones(self.n_clusters, device=self.device)

        for it in range(self.max_iter):
            perm = torch.randperm(n_samples)
            total_shift = 0.0

            for i in range(0, n_samples, self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                xb = X[batch_idx]

                dist = self._distance(xb, self.centroids)
                U = self._compute_membership(dist)  # [B, C]
                Um = U.pow(self.m)                 # [B, C]

                # Update centroids (batch-based)
                new_centroids = (Um.T @ xb) / (Um.sum(0).unsqueeze(1) + 1e-8)

                # Weighted moving average update
                weights = Um.sum(0)
                shift = torch.norm(new_centroids - self.centroids, dim=1).sum()
                total_shift += shift.item()

                self.centroids = (self.centroids * self.counts.unsqueeze(1) + new_centroids * weights.unsqueeze(1)) / \
                                 (self.counts.unsqueeze(1) + weights.unsqueeze(1) + 1e-8)
                self.counts += weights

            if self.verbose:
                print(f"Iter {it+1}, total centroid shift: {total_shift:.6f}")

            if total_shift < self.tol:
                if self.verbose:
                    print("Converged.")
                break

        # Final prediction
        final_dist = self._distance(X, self.centroids)
        final_U = self._compute_membership(final_dist)
        labels = final_U.argmax(dim=1)
        return labels, final_U.t()

    def predict(self, X):
        X = X.to(self.device)
        dist = self._distance(X, self.centroids)
        U = self._compute_membership(dist)
        return U.argmax(dim=1), U
