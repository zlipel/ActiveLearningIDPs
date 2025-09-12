import torch 
from torch import nn
import torch.nn.functional as F

class DNN(torch.nn.Module):

    def __init__(self, dim_list, output_dim):
        super(DNN, self).__init__()

        self.linear     = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list)-1)])
        self.out        = nn.Linear(dim_list[-1], output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):

        for layer in self.linear:
            x = self.activation(layer(x))

        x = self.out(x)

        return x


            

        
        

