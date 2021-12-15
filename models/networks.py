import torch
from torch import nn

class Normalization(nn.Module):
    def __init__(self, n_in, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, n_in).to(device)
        self.std = torch.tensor(std).view(-1, n_in).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class MLP(nn.Module):
    def __init__(self, n_in, data_train_mean, data_train_std, device, n_out=1):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.layers = nn.Sequential(
                    Normalization(n_in, data_train_mean, data_train_std, device),
                    nn.Linear(n_in, 100),
                    nn.ReLU(),
                    nn.Linear(100, 500),
                    nn.ReLU(),
                    nn.Linear(500,100),
                    nn.ReLU(),
                    nn.Linear(100, n_out),
                    nn.Flatten()
                    )
    def forward(self, input):
        out = self.layers(input)
        return out

class MLP_Large(nn.Module):
    def __init__(self, n_in, data_train_mean, data_train_std, device, n_out=1):
        super(MLP_Large, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.layers = nn.Sequential(
                    Normalization(n_in, data_train_mean, data_train_std, device),
                    nn.Linear(n_in, 200),
                    nn.ReLU(),
                    nn.Linear(200, 1000),
                    nn.ReLU(),
                    nn.Linear(1000,200),
                    nn.ReLU(),
                    nn.Linear(200, n_out),
                    nn.Flatten()
                    )
    def forward(self, input):
        out = self.layers(input)
        return out

class MLP_Multistep(nn.Module):
    def __init__(self, n_in, n_out=2):
        super(MLP_Multistep, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.layers = nn.Sequential(
                    nn.Linear(n_in, 200),
                    nn.ReLU(),
                    nn.Linear(200, 500),
                    nn.ReLU(),
                    nn.Linear(500, 200),
                    nn.ReLU(),
                    nn.Linear(200,n_out),
                    nn.Flatten()
                    )
    def forward(self, input):
        out = self.layers(input)
        return out

# https://gist.github.com/papachristoumarios/3da173eba99ea9716ccf13f71a36ae91

# class NonLinear2D(nn.Module):
#     def __init__(self):
#         # self.x = torch.Variable(torch.tensor(x0, requires_grad=True, device=device))\
#         #         .reshape(1, *x0.shape)
#         pass
        
#     def forward(self, x):
#         out = torch.exp(- x[:, 0] * x[:, 1])
#         return out

def Nonelinear2D(x):
    out = torch.exp(- x[:, 0] * x[:, 1])
    return out