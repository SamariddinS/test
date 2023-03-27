import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import numpy as np

def local_kernel(u, v):
    dist = torch.norm(u - v, p=2, dim=2)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat

class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
      super().__init__()
      self.W = nn.Parameter(torch.randn(n_in, n_hid))
      self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
      self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
      self.b = nn.Parameter(torch.randn(n_hid))

      self.lambda_s = lambda_s
      self.lambda_2 = lambda_2

      nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.zeros_(self.b)
      self.activation = activation

    def forward(self, x):
      w_hat = local_kernel(self.u, self.v)

      sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
      sparse_reg_term = self.lambda_s * sparse_reg

      l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
      l2_reg_term = self.lambda_2 * l2_reg

      W_eff = self.W * w_hat  # Local kernelised weight matrix
      y = torch.matmul(x, W_eff) + self.b
      y = self.activation(y)

      return y, sparse_reg_term + l2_reg_term

class KernelNet(nn.Module):
    def __init__(self, n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2):
      super().__init__()
      layers = []
      for i in range(n_layers):
        if i == 0:
          layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2))
        else:
          layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))
      layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
      self.layers = nn.ModuleList(layers)
      self.dropout = nn.Dropout(0.5)

    def forward(self, x):
      total_reg = None
      for i, layer in enumerate(self.layers):
        x, reg = layer(x)
        if i < len(self.layers)-1:
          x = self.dropout(x)
        if total_reg is None:
          total_reg = reg
        else:
          total_reg += reg
      return x, total_reg


class CompleteNet(nn.Module):
    def __init__(self, kernel_net, n_u, n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2, gk_size, dot_scale):
      super().__init__()
      self.gk_size = gk_size
      self.dot_scale = dot_scale
      self.local_kernel_net = kernel_net
      self.global_kernel_net = KernelNet(n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2)
      self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
      nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))


    def forward(self, train_r):
      x, _ = self.local_kernel_net(train_r)
      gk = self.global_kernel(x, self.gk_size, self.dot_scale)
      x = self.global_conv(train_r, gk)
      x, global_reg_loss = self.global_kernel_net(x)
      return x, global_reg_loss

    def global_kernel(self, input, gk_size, dot_scale):
      avg_pooling = torch.mean(input, dim=0)  # Item (axis=0) based average pooling
      avg_pooling = avg_pooling.view(1, -1)

      gk = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale  # Scaled dot product
      gk = gk.view(1, 1, gk_size, gk_size)

      return gk

    def global_conv(self, input, W):
      input = input.unsqueeze(0).unsqueeze(0)
      conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
      return conv2d.squeeze(0).squeeze(0)

class Loss(nn.Module):
    def forward(self, pred_p, reg_loss, train_m, train_r):
      # L2 loss
      diff = train_m * (train_r - pred_p)
      sqE = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
      loss_p = sqE + reg_loss
      return loss_p

import time
import os

class GlocalK():
    def __init__(self, n_hid = 500, n_dim = 5, n_layers = 2, gk_size=3, seed=1284):
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.gk_size = gk_size
        torch.manual_seed(seed)
        np.random.seed(seed)


    def train(self, model, optimizer, max_epoch, patience, tol, verbose, print_each, phase, save_folder):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder,"best_model_weights.pth")
        time_cumulative = 0
        tic = time.time()

        last_rmse = np.inf
        best_rmse = np.inf
        counter = 0

        def closure():
            optimizer.zero_grad()
            x = torch.Tensor(self.train_matrix).double().to(device)
            m = torch.Tensor(self.train_mask).double().to(device)
            model.train()
            pred, reg = model(x)
            loss = Loss().to(device)(pred, reg, m, x)
            loss.backward()
            return loss

        for i in range(max_epoch):
            optimizer.step(closure)
            model.eval()
            t = time.time() - tic
            time_cumulative += t

            pre, _ = model(torch.Tensor(self.train_matrix).double().to(device))

            pre = pre.float().cpu().detach().numpy()

            error_val = (self.val_mask * (pre - self.val_matrix) ** 2).sum() / self.val_mask.sum()  # val error
            val_rmse = np.sqrt(error_val)

            if last_rmse-val_rmse < tol:
                counter += 1
            else:
                counter = 0

            if best_rmse >= val_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), save_path)

            last_rmse = val_rmse

            if patience == counter:
                if verbose > 1:
                    print('.-^-._' * 12)
                    print(f'{phase}')
                    print('Epoch:', i+1, 'val rmse:', val_rmse)
                    print('Time:', t, 'seconds')
                    print('Time cumulative:', time_cumulative, 'seconds')
                    print('.-^-._' * 12)
                elif verbose==1:
                    print(f'{phase} finished.')
                model.load_state_dict(torch.load(save_path))
                break


            if i % print_each != 0 or verbose <= 1:
                continue
            print('.-^-._' * 12)
            print(f'{phase}')
            print('Epoch:', i+1, 'val rmse:', val_rmse)
            print('Time:', t, 'seconds')
            print('Time cumulative:', time_cumulative, 'seconds')
            print('.-^-._' * 12)

        return i, best_rmse

    def get_lds(self, lambda_2, lambda_s, dot_scale, matrix):
        K = 55_000 # dot_scale * (lambda_2 / lambda_s) / sparsity
        lambda_2_dot_scale = 20 #(lambda_2*dot_scale)
        alpha = 0.05
        beta = 0.25
        mask = matrix.copy()
        mask[mask!=0] = 1
        if mask.sum() == 0:
            raise ValueError("Matrix must have at least one rating.")
        sparsity = mask.sum()/(matrix.shape[0]*matrix.shape[1])
        if lambda_2 is None:
            if lambda_s is None:
                if dot_scale is None:
                    lambda_2 = (alpha*matrix.shape[0]+beta*matrix.shape[1])*sparsity
                    lambda_s = 1/((K/lambda_2_dot_scale)*sparsity)
                    dot_scale = K*sparsity*lambda_s/lambda_2
                else:
                    lambda_ratio = sparsity*K/dot_scale
                    lambda_s = 1/((K/lambda_2_dot_scale)*sparsity)
                    lambda_2 = lambda_ratio*lambda_s
            else:
                if dot_scale is None:
                    lambda_2 = (alpha*matrix.shape[0]+beta*matrix.shape[1])*sparsity
                    dot_scale = K*sparsity*lambda_s/lambda_2
                else:
                    lambda_ratio = sparsity*K/dot_scale
                    lambda_2 = lambda_ratio*lambda_s
        else:
            if lambda_s is None:
                if dot_scale is None:
                    lambda_s = 1/((K/lambda_2_dot_scale)*sparsity)
                    dot_scale = K*sparsity*lambda_s/lambda_2
                else:
                    lambda_ratio = sparsity*K/dot_scale
                    lambda_s = lambda_2/lambda_ratio
            else:
                if dot_scale is None:
                    dot_scale = K*sparsity*lambda_s/lambda_2

        return lambda_2, lambda_s, dot_scale

    def fit(self, matrix: np.array, max_epoch_p=200, max_epoch_f=200, patience_p=10,patience_f=20, tol_p=1e-3, tol_f=1e-3, lambda_2=None, lambda_s=None, dot_scale=None, val_split=0.05, verbose = 1, print_each=20, save_folder = "models/"):
        n_u, n_m = matrix.shape

        lambda_2, lambda_s, dot_scale = self.get_lds(lambda_2, lambda_s, dot_scale, matrix)

        mask = np.greater(matrix, 1e-12).astype('float32')
        mask_flat = mask.reshape(-1,)
        indices = np.argwhere(mask_flat==1).reshape(-1,)
        flat_index = np.random.choice(indices, int(len(indices)*val_split))
        val_mask_flat = np.zeros_like(mask_flat)
        val_mask_flat[flat_index] = 1
        val_mask = val_mask_flat.reshape(mask.shape)
        train_mask = mask-val_mask

        self.train_matrix = matrix*train_mask
        self.val_matrix = matrix*val_mask
        self.train_mask = train_mask
        self.val_mask = val_mask

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inner_model = KernelNet(n_m, self.n_hid, self.n_dim, self.n_layers, lambda_s, lambda_2).double().to(device)
        self.model = CompleteNet(inner_model, n_u, n_m, self.n_hid, self.n_dim, self.n_layers, lambda_s, lambda_2, self.gk_size, dot_scale).double().to(device)


        optimizer_p = torch.optim.AdamW(self.model.local_kernel_net.parameters(), lr=0.001)
        optimizer_f = torch.optim.AdamW(self.model.parameters(), lr=0.001)

        # Pre-Training
        epochs_p, best_rmse_p = self.train(self.model.local_kernel_net, optimizer_p, max_epoch=max_epoch_p, patience=patience_p,tol=tol_p,verbose=verbose,print_each=print_each, phase='PRE-TRAINING', save_folder=save_folder)

        # Fine-Tuning
        epochs_f, best_rmse_f = self.train(self.model, optimizer_f, max_epoch=max_epoch_f, patience=patience_f,tol=tol_f,verbose=verbose,print_each=print_each, phase='FINE-TUNING', save_folder=save_folder)

        return {"epochs_p": epochs_p+1, "epochs_f": epochs_f+1, "best_rmse_p": best_rmse_p, "best_rmse_f": best_rmse_f}

    def predict(self, user_index):
        try:
            _ = self.model
        except:
            raise IndexError('Fit must be called before predicting.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        new_mask = 1-(self.train_mask+self.val_mask)
        pred, _ = self.model(torch.Tensor(self.train_matrix+self.val_matrix).double().to(device))
        scores = new_mask * pred.float().cpu().detach().numpy()
        return scores[:,user_index]

    def load(self, matrix: np.array, path, lambda_2 = None, lambda_s = None, dot_scale = None):
        n_u, n_m = matrix.shape

        lambda_2, lambda_s, dot_scale = self.get_lds(lambda_2, lambda_s, dot_scale, matrix)

        self.train_matrix = matrix
        self.val_matrix = np.zeros_like(matrix)
        self.train_mask = np.ones_like(matrix)
        self.val_mask = np.zeros_like(matrix)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inner_model = KernelNet(n_m, self.n_hid, self.n_dim, self.n_layers, lambda_s, lambda_2).double().to(device)
        self.model = CompleteNet(inner_model, n_u, n_m, self.n_hid, self.n_dim, self.n_layers, lambda_s, lambda_2, self.gk_size, dot_scale).double().to(device)
        self.model.load_state_dict(torch.load(path))