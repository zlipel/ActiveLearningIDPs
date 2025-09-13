import torch
import gpytorch

class GPRTrainer:
    """
    Trainer class for Gaussian Process Regression (GPR) using GPyTorch.
    
    This class handles the training and evaluation of the GPR model, 
    including support for early stopping and batch training.
    
    Args:
        model (GPRegressionModel): The GPR model to train.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood for the GP model.
        optimizer_type (str): Type of optimizer to use ('adam' or 'sgd').
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    def __init__(self, model, likelihood, optimizer_type='adam', learning_rate=0.01, epochs=500, patience=3):
        self.model = model
        self.likelihood = likelihood
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) if optimizer_type == "adam" else torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.epochs = epochs
        self.patience = patience
        self.device = 'cpu'

        self.model.to(self.device)
        self.model.train()
        self.likelihood.train()
    
    def train(self, train_dat, val_dat, test_loader=None, early_stop=True):
        """Train the GP model.

        Parameters
        ----------
        train_dat : tuple
            Training features and labels.
        val_dat : tuple or None
            Validation features and labels. If ``None`` no validation is used.
        test_loader : DataLoader, optional
            Unused placeholder for compatibility.
        early_stop : bool, optional
            Whether to perform early stopping based on validation loss.

        Returns
        -------
        dict
            Training (and optionally validation) loss history.
        """

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
    
        # Prepare full training dataset
        if val_dat is not None:
            train_x, train_y, val_x, val_y = train_dat[0], train_dat[1], val_dat[0], val_dat[1]
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
        else:
            train_x, train_y = train_dat[0], train_dat[1]
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)

        
        
        for epoch in range(self.epochs):
            self.model.train()
            self.likelihood.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
    
            # Forward pass over the full training data
            output = self.model(train_x)
            loss = -self.mll(output, train_y.flatten())
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            train_losses.append(epoch_loss)
            
            if val_dat is not None:
                val_loss = self.evaluate(val_x, val_y)
                val_losses.append(val_loss)
            
                if early_stop and epoch%20==0:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state_dict = self.model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter > self.patience:
                                print(f"Early stopping at epoch {epoch}")
                                self.model.load_state_dict(best_state_dict)
                                break
    
            # if test_loader is not None:
            #     val_loss = self.evaluate(val_x, val_y)
            #     val_losses.append(val_loss)
    
            #     if early_stop:
            #         if val_loss < best_val_loss:
            #             best_val_loss = val_loss
            #             best_state_dict = self.model.state_dict()
            #             patience_counter = 0
            #         else:
            #             patience_counter += 1
            #             if patience_counter > self.patience:
            #                 print(f"Early stopping at epoch {epoch}")
            #                 self.model.load_state_dict(best_state_dict)
            #                 break
    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_loss} - lengthscale: {self.model.covar_module.base_kernel.lengthscale.item()} - noise: {self.model.likelihood.noise.item()}")

        if val_dat is not None:
            return {"train_losses": train_losses, "val_losses": val_losses}
        else:
            return {"train_losses": train_losses}

    def evaluate(self, val_x, val_y):
        """Evaluate the model on validation data.

        Parameters
        ----------
        val_x : torch.Tensor
            Validation features.
        val_y : torch.Tensor
            Validation targets.

        Returns
        -------
        float
            Mean squared error of the predictions.
        """
        self.model.eval()
        self.likelihood.eval()
        total_mse = 0.0
        #count = 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # for test_x, test_y in data_loader:
            #     test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred = self.model(val_x)
            mse = torch.mean((pred.mean - val_y) ** 2).item()
            total_mse += mse
           # count += 1

        self.model.train()
        self.likelihood.train()
        return total_mse #/ count
    
class MultitaskGPRTrainer:
    """
    Trainer class for Gaussian Process Regression (GPR) using GPyTorch.
    
    This class handles the training and evaluation of the GPR model, 
    including support for early stopping and batch training.
    
    Args:
        model (GPRegressionModel): The GPR model to train.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood for the GP model.
        optimizer_type (str): Type of optimizer to use ('adam' or 'sgd').
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    def __init__(self, model, likelihood, optimizer_type='adam', learning_rate=0.01, epochs=500, patience=3):
        self.model = model
        self.likelihood = likelihood
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) if optimizer_type == "adam" else torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.epochs = epochs
        self.patience = patience
        self.device = 'cpu'

        self.model.to(self.device)
        self.model.train()
        self.likelihood.train()
    
    def train(self, train_dat, val_dat, test_loader=None, early_stop=True):
        """Train the multitask GP model with optional early stopping."""

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
    
        # Prepare full training dataset
        if val_dat is not None:
            train_x, train_y, val_x, val_y = train_dat[0], train_dat[1], val_dat[0], val_dat[1]
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
        else:
            train_x, train_y = train_dat[0], train_dat[1]
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)

        
        
        for epoch in range(self.epochs):
            self.model.train()
            self.likelihood.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
    
            # Forward pass over the full training data
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            train_losses.append(epoch_loss)
            
            if val_dat is not None:
                val_loss = self.evaluate(val_x, val_y)
                val_losses.append(val_loss)
            
                if early_stop and epoch%20==0:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state_dict = self.model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter > self.patience:
                                print(f"Early stopping at epoch {epoch}")
                                self.model.load_state_dict(best_state_dict)
                                break
    
            # if test_loader is not None:
            #     val_loss = self.evaluate(val_x, val_y)
            #     val_losses.append(val_loss)
    
            #     if early_stop:
            #         if val_loss < best_val_loss:
            #             best_val_loss = val_loss
            #             best_state_dict = self.model.state_dict()
            #             patience_counter = 0
            #         else:
            #             patience_counter += 1
            #             if patience_counter > self.patience:
            #                 print(f"Early stopping at epoch {epoch}")
            #                 self.model.load_state_dict(best_state_dict)
            #                 break
    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_loss} - lengthscale: {self.model.covar_module.data_covar_module.lengthscale.item()} - noise: {self.model.likelihood.noise.detach().numpy()}")

        if val_dat is not None:
            return {"train_losses": train_losses, "val_losses": val_losses}
        else:
            return {"train_losses": train_losses}

    def evaluate(self, val_x, val_y):
        """Evaluate the multitask model on validation data."""
        self.model.eval()
        self.likelihood.eval()
        total_mse = 0.0
        #count = 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # for test_x, test_y in data_loader:
            #     test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred = self.likelihood(self.model(val_x))
            mse = torch.mean((pred.mean - val_y) ** 2).sum().item()
            total_mse += mse
           # count += 1

        self.model.train()
        self.likelihood.train()
        return total_mse #/ count


