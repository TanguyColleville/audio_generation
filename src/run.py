import torch
import torch.nn as nn 

class Creator(nn.Module): 
    def __init__(self) -> None:
        raise NotImplementedError
    def load_data(self):
        raise NotImplementedError
    def load_model(self) :
        raise NotImplementedError
    def fit_one_epoch(self):
        raise NotImplementedError
    def val_one_epoch(self):
        raise NotImplementedError
    def fit(self):
        raise NotImplementedError