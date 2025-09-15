import random
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split,KFold

from sklearn.model_selection import train_test_split, KFold

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from tqdm import tqdm
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r  took: %2.4f sec' % (f.__name__,  te-ts))
        return result
    return wrap


class Trainer:

    def __init__(self, model, optimizer_type='adam', learning_rate=0.001, epoch=100, batch_size=64):

    """Utility class for training feed-forward neural networks."""

    def __init__(self, model, optimizer_type='adam', learning_rate=0.001, epoch=100, batch_size=64):
        """Configure the training procedure.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network to optimize.
        optimizer_type : str, optional
            Either ``"adam"`` or ``"sgd"``.
        learning_rate : float, optional
            Optimizer learning rate.
        epoch : int, optional
            Number of training epochs.
        batch_size : int, optional
            Mini-batch size used during training.
        """


        self.model = model
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = torch.nn.MSELoss()
        self.device = None

    @timing
    def train(self, train_loader, test_loader, early_stop=False, l2=False, silent=False, device='cpu', weight_cost=1e-5, draw_curve=False):

        """Train the network and optionally plot loss curves."""


        self.device = torch.device('cuda' if torch.cuda.is_available() else device)
        self.model.to(device)

        losses = []
        val_losses = []
        weights = self.model.state_dict()
        lowest_val_loss = np.inf

        for n_epoch in tqdm(range(self.epoch), leave=True):

            self.model.train()
            epoch_loss = 0.

            for inputs in train_loader:

                feats, labels = inputs

                batch_importance = feats.size(0)/train_loader.batch_size

                feats = feats.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(feats)
                loss = self.loss(out, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_importance
            
            losses.append(epoch_loss)

            val_loss = self.evaluate(test_loader)
            val_losses.append(val_loss)

            if n_epoch % 10 ==0 and not silent: 
                print("Epoch %d/%d - Loss: %.3f" % (n_epoch + 1, self.epoch, epoch_loss))
                print("              Val_loss: %.3f" % (val_loss))

            if early_stop:
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    weights = self.model.state_dict()
                    
        if draw_curve:
            plt.figure()
            plt.plot(np.arange(self.epoch) + 1,losses,label='Training loss')
            plt.plot(np.arange(self.epoch) + 1,val_losses,label='Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
          

        return {"losses": losses, "val_losses": val_losses}

    def evaluate(self, test_loader):


        self.model.eval()
        test_loss = 0.

        for feats, labels in test_loader:

            feats  = feats.to(self.device)
            labels = labels.to(self.device)

            batch_importance = feats.size(0)/test_loader.batch_size

        """Compute loss on a validation or test set."""

        self.model.eval()
        test_loss = 0.0

        for feats, labels in test_loader:
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            batch_importance = feats.size(0) / test_loader.batch_size


            with torch.no_grad():
                out = self.model(feats)
                loss = self.loss(out, labels)

            test_loss += loss.item() * batch_importance
        self.model.train()
        return test_loss
            

        
