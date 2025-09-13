import os

# Import model definitions
from .gpr_model import GPRegressionModel, MultitaskGPRegressionModel
from .rf_model import RandomForestClassifierModel
from .DNN_Model import DNN

from sklearn.ensemble import RandomForestClassifier

# Import the trainer classes
from .gpr_trainer import GPRTrainer, MultitaskGPRTrainer
from .dnn_trainer import Trainer

# Import the data preprocessing functions
from al_pipeline.features.data_preprocessing import (
    load_dataset,
    ProteinDataset,
    load_classification_dataset,
    ClassificationDataset,
)

# Additional imports
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd


def save_chkpt(model_path, model, optimizer=None, val_losses=None, train_losses=None, trained=False):
    """
    Save a training checkpoint.

    Args:
        model_path (str): The path to save the model to.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        val_losses (list of float): A list containing the validation losses.
        train_losses (list of float): A list containing the training losses.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if trained==True:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    else:
        state_dict = {
            'model': model.state_dict()
        }
    torch.save(state_dict, model_path)

def KFold_GPR(features_file, labels_file, label_column='B2', k=5, epochs=10, patience=100, lr=0.01, batch_size=32, kernel=None, model_save_path="avg_gpr.pt"):
    """
    Perform K-fold cross-validation on a GPR model and save the best model.

    Args:
        features_file (str): Path to the CSV file containing the feature data.
        labels_file (str): Path to the CSV file containing the labels.
        label_column (str): The name of the label column to extract from the labels file.
        k (int): Number of folds for cross-validation.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        kernel (gpytorch.kernels.Kernel, optional): The kernel to use in the GP model. Defaults to Matérn kernel.
        model_save_path (str): Path to save the best model.

    Returns:
        None: Prints the training and validation mean squared error (MSE) for each fold and saves the best model.
    """
    # Load the dataset using the preprocessing utility
    feats, labels, scaler = load_dataset(features_file, labels_file, label_column=label_column)

    # set up hyperparameter lists
    length_scales = []
    variances = []
    noise_variances = []


    labels = labels.squeeze(-1)
    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    train_mses = []
    val_mses = []
    best_val_mse = float('inf')
    best_model = None
    best_optimizer_state = None
    best_log = None

    #logs = []
    
    for train_idx, val_idx in kf.split(feats):
        print(len(train_idx), len(val_idx))
        fold += 1
        print(f"Fold {fold}")
        
        # Create Subsets for train and validation splits
        train_feats, train_labels = feats[train_idx], labels[train_idx]
        val_feats, val_labels     = feats[val_idx], labels[val_idx] 
        
        # DataLoader for batching
        # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize model and trainer
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_feats, train_labels, likelihood, kernel=kernel)
        
        trainer = GPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
        log = trainer.train((train_feats, train_labels), (val_feats, val_labels), early_stop=True)
        #logs.append(log)

        print(train_feats.shape, train_labels.shape)
        train_mse = trainer.evaluate(train_feats, train_labels)
        val_mse = trainer.evaluate(val_feats, val_labels)
        
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        
        print(f"Train MSE: {train_mse}, Val MSE: {val_mse}")
        
        # Check if this is the best model based on validation MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            #best_model = model.state_dict()
            best_optimizer_state = trainer.optimizer.state_dict()
            best_log = log
    
    # Save the best model
        model.eval()
        likelihood.eval()
        # ypred = scaler.inverse_transform(likelihood(model(val_feats)).mean.unsqueeze(-1).detach().numpy())
        # ylab  = scaler.inverse_transform(val_labels.unsqueeze(-1).detach().numpy())
        ypred = likelihood(model(val_feats)).mean.unsqueeze(-1).detach().numpy()
        ylab  = val_labels.unsqueeze(-1).detach().numpy()

        
        
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        ax.scatter(ylab, ypred, color='b', alpha=0.5)
        ax.set_xlabel('True Values', fontsize=4)
        ax.set_ylabel('Predicted Values', fontsize=4)
        ax.set_title('GP model performance', fontsize=6)
        #ax.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
        ax.plot([min(ylab), max(ylab)], [min(ylab), max(ylab)], 'r--')
        ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
        #ax.text(0.1, 0.9, f'RMSE from CV = {rmse:.4f}', transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        plt.show()
        if best_model is not None:
            save_chkpt(model_save_path, model, trainer.optimizer, best_log["val_losses"], best_log["train_losses"])
            print(f"Best model saved with validation MSE: {best_val_mse}")


        # Extract optimized hyperparameters from model (for each fold)
        length_scales.append(model.covar_module.base_kernel.lengthscale.item())
        variances.append(model.covar_module.outputscale.item())
        noise_variances.append(likelihood.noise.item())

    # Compute averages across folds
    avg_lengthscale = np.mean(length_scales)
    avg_variance = np.mean(variances)
    avg_noise_variance = np.mean(noise_variances)

    
    print(f"Final k-fold Results: Train MSE: {np.mean(train_mses)} ± {np.std(train_mses)}, Val MSE: {np.mean(val_mses)} ± {np.std(val_mses)}")

    # After averaging hyperparameters across folds
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    train_X, test_X, train_y, test_y = train_test_split(feats, labels, test_size=0.2, shuffle=True)
    
    model = GPRegressionModel(feats, labels, likelihood, kernel=kernel)
    
    # Set the hyperparameters to their averaged values
    model.covar_module.base_kernel.lengthscale = torch.tensor(avg_lengthscale)
    model.covar_module.outputscale = torch.tensor(avg_variance)
    likelihood.noise = torch.tensor(avg_noise_variance)

    #model.eval()
    #likelihood.eval()
    
    # Train the final model on the entire dataset
    final_trainer = GPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
    final_log = final_trainer.train((feats, labels), None, early_stop=True)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        test_pred  = likelihood(model(test_X)).mean.unsqueeze(-1).detach().numpy()
        train_pred = likelihood(model(train_X)).mean.unsqueeze(-1).detach().numpy()
    
    labels_test  = test_y.unsqueeze(-1).detach().numpy()
    labels_train = train_y.unsqueeze(-1).detach().numpy()

    # if label_column == 'diff':
    #     ax_lab = 'D'
    # elif label_column == 'B2':
    #     ax_lab = '$B_2$'
    ax_lab = label_column
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    ax.scatter(labels_train, train_pred, color='orange', alpha=0.3)
    ax.scatter(labels_test, test_pred, color='orange', alpha=0.3)
    ax.set_xlabel(f'True {ax_lab}', fontsize=6)
    ax.set_ylabel(f'Predicted {ax_lab}', fontsize=6)
    ax.set_title('GP model performance', fontsize=6)
    #ax.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
    ax.plot([min(labels_train), max(labels_train)], [min(labels_train), max(labels_train)], 'r--')
    ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
    #ax.text(0.1, 0.9, f'RMSE from CV = {rmse:.4f}', transform=ax.transAxes, fontsize=10)
    #plt.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(model_save_path + "FIT.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final model
    save_chkpt(model_save_path + '.pt', model)
    print(f"Final model saved with training on the entire dataset")

def KFold_GPR_multitask(features_file, labels_file, label_columns = ['exp_density', 'diff'], k=5, epochs=10, \
                        patience=100, lr=0.01, batch_size=32, model_save_path="avg_gpr.pt", \
                            exploration=None, ehvi_var=None, transform='log'):
    """
    Perform K-fold cross-validation on a GPR model and save the best model.

    Args:
        features_file (str): Path to the CSV file containing the feature data.
        labels_file (str): Path to the CSV file containing the labels.
        label_column (str): The name of the label column to extract from the labels file.
        k (int): Number of folds for cross-validation.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        kernel (gpytorch.kernels.Kernel, optional): The kernel to use in the GP model. Defaults to Matérn kernel.
        model_save_path (str): Path to save the best model.

    Returns:
        None: Prints the training and validation mean squared error (MSE) for each fold and saves the best model.
    """
    # Load the dataset using the preprocessing utility
    feats, labels1 = load_dataset(features_file, labels_file, label_column=label_columns[0])
    feats, labels2 = load_dataset(features_file, labels_file, label_column=label_columns[1])


    feats = torch.tensor(feats).float()
    if transform == 'log':
        labels2 = torch.tensor(np.log(labels2+1e-8)).float()
    elif transform == 'yeoj':
        labels2 = torch.tensor(labels2).float()

    labels1 = torch.tensor(labels1).float()

    labels = torch.stack([labels1.flatten(), labels2.flatten()], -1)


    #print('before: {labels.size()}')
    labels = labels.squeeze(-1)
    #print(f'after: {labels.size()}')

    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    train_mses = []
    val_mses = []
    best_val_mse = float('inf')
    best_model = None
    best_optimizer_state = None
    best_log = None
    
    #logs = []

    model_dicts = []
    likelihood_dicts = []
    
    for train_idx, val_idx in kf.split(feats):
        print(len(train_idx), len(val_idx))
        fold += 1
        print(f"Fold {fold}")
        
        # Create Subsets for train and validation splits
        train_feats, train_labels = feats[train_idx], labels[train_idx]
        val_feats, val_labels     = feats[val_idx], labels[val_idx] 

        if transform == 'log':
            scaler1 = StandardScaler()
            scaler2 = StandardScaler()
        elif transform == 'yeoj':
            scaler1 = PowerTransformer(method='yeo-johnson', standardize=True)
            scaler2 = PowerTransformer(method='yeo-johnson', standardize=True)

        train_labels[:, 0] = torch.tensor(scaler1.fit_transform(train_labels[:, 0].view(-1,1))).float().flatten()
        train_labels[:, 1] = torch.tensor(scaler2.fit_transform(train_labels[:, 1].view(-1,1))).float().flatten()

        val_labels[:, 0] = torch.tensor(scaler1.transform(val_labels[:, 0].view(-1,1))).float().flatten()
        val_labels[:, 1] = torch.tensor(scaler2.transform(val_labels[:, 1].view(-1,1))).float().flatten()
        

        # Initialize model and trainer
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPRegressionModel(train_feats, train_labels, likelihood, num_tasks=2)
        
        trainer = MultitaskGPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
        log = trainer.train((train_feats, train_labels), (val_feats, val_labels), early_stop=True)
        #logs.append(log)

        print(train_feats.shape, train_labels.shape)
        train_mse = trainer.evaluate(train_feats, train_labels)
        val_mse = trainer.evaluate(val_feats, val_labels)
        
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        
        print(f"Train MSE: {train_mse}, Val MSE: {val_mse}")
        
        # Check if this is the best model based on validation MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            #best_model = model.state_dict()
            best_optimizer_state = trainer.optimizer.state_dict()
            best_log = log
    
    # Save the best model
        model.eval()
        likelihood.eval()
    
        if best_model is not None:
            save_chkpt(model_save_path, model, trainer.optimizer, best_log["val_losses"], best_log["train_losses"])
            print(f"Best model saved with validation MSE: {best_val_mse}")


        # Extract optimized hyperparameters from model (for each fold)
    
        model_dicts.append(model.state_dict())
        likelihood_dicts.append(likelihood.state_dict())
    
    
    print(f"Final k-fold Results: Train MSE: {np.mean(train_mses)} ± {np.std(train_mses)}, Val MSE: {np.mean(val_mses)} ± {np.std(val_mses)}")

    # After averaging hyperparameters across folds
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    if transform == 'log':
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
    elif transform == 'yeoj':
        scaler1 = PowerTransformer(method='yeo-johnson', standardize=True)
        scaler2 = PowerTransformer(method='yeo-johnson', standardize=True)
    labels[:, 0] = torch.tensor(scaler1.fit_transform(labels[:, 0].view(-1,1))).float().flatten()
    labels[:, 1] = torch.tensor(scaler2.fit_transform(labels[:, 1].view(-1,1))).float().flatten()

    train_X, test_X, train_y, test_y = train_test_split(feats, labels, test_size=0.2, shuffle=True)
    
    model = MultitaskGPRegressionModel(feats, labels, likelihood, num_tasks = 2)

    # average hyperparameters
    for key in model_dicts[0]:
        model_dicts[0][key] = sum([model_dicts[val][key] for val in range(len(model_dicts))])/len(model_dicts)

    for key in likelihood_dicts[0]:
        likelihood_dicts[0][key] = sum([likelihood_dicts[val][key] for val in range(len(likelihood_dicts))])/len(likelihood_dicts)

    model.load_state_dict(model_dicts[0])
    likelihood.load_state_dict(likelihood_dicts[0])

    # Train the final model on the entire dataset
    final_trainer = MultitaskGPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
    final_log = final_trainer.train((feats, labels), None, early_stop=True)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        test_pred  = likelihood(model(test_X)).mean.detach().numpy()
        train_pred = likelihood(model(train_X)).mean.detach().numpy()
    
    labels_test  = test_y.detach().numpy()
    labels_train = train_y.detach().numpy()


    if exploration == 'kriging_believer' or exploration == 'constant_liar_min' or exploration == 'constant_liar_mean' or exploration == 'constant_liar_max':
        # get the generation folder by instpection features_file, which has structure like /.../features_genN.csv
        gen_folder = os.path.dirname(features_file)
        featname = features_file.split('/')[-1].split('.')[0]
        labelname = labels_file.split('/')[-1].split('.')[0]

        feat_orig_df = pd.read_csv(features_file)

        # first turn labels/feats we just trained on into data frame with same column names as original
        features_train_df = pd.DataFrame(feats.detach().numpy(), columns=feat_orig_df.columns)
        labels_train_df = pd.DataFrame(labels.detach().numpy(), columns=label_columns)

        features_train_df.to_csv(os.path.join(gen_folder, f"{featname}_NORM_{ehvi_var}_{exploration}_{transform}.csv"), index=False)
        labels_train_df.to_csv(os.path.join(gen_folder, f"{labelname}_NORM_{ehvi_var}_{exploration}_{transform}.csv"), index=False)

        # now save labels/features we trained to model on at the end to generation file as csv with same column names



    for i, label in enumerate(label_columns):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

        # Extract true and predicted values for task `i`
        y_train = labels_train[:, i]
        y_test = labels_test[:, i]
        y_train_pred = train_pred[:, i]
        y_test_pred = test_pred[:, i]

        combined = np.concatenate([y_train, y_test])
        combined_pred = np.concatenate([y_train_pred, y_test_pred])

        r2 = r2_score(combined, combined_pred)

        ax.scatter(combined, combined_pred, color='orange', alpha=0.3, label=f"$R^2$={r2:.4f}")
        #ax.scatter(y_test, y_test_pred, color='orange', alpha=0.3)

        ax.set_xlabel(f'True {label}', fontsize=6)
        ax.set_ylabel(f'Predicted {label}', fontsize=6)
        ax.set_title(f'GPR Performance — {label}', fontsize=6)


        ax.plot([min(combined), max(combined)], [min(combined), max(combined)], 'r--')

        ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(model_save_path + f"_FIT_{label}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # Save final model
    save_chkpt(model_save_path + '.pt', model)
    print(f"Final model saved with training on the entire dataset")



def save_rf_model(model_path, model):
    """
    Save the Random Forest model to a file.
    
    Args:
        model_path (str): The path to save the model to.
        model (RandomForestClassifier): The Random Forest model to save.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_rf_model(model_path):
    """
    Load a saved Random Forest model from a file.
    
    Args:
        model_path (str): Path to the saved model file.
        
    Returns:
        RandomForestClassifier: The loaded Random Forest model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model



def average_model_weights(models):
    """
    Average the weights of the models across K folds.
    
    Args:
        models (list): A list of model state_dicts.
    
    Returns:
        dict: Averaged state_dict for the model.
    """
    num_models = len(models)
    avg_state_dict = models[0].copy()  # Initialize with the state_dict from the first model
    for key in avg_state_dict:
        # Average the values across models
        avg_state_dict[key] = torch.stack([models[i][key] for i in range(num_models)], dim=0).mean(dim=0)
    return avg_state_dict


def KFold_DNN(features_file, labels_file, label_column='B2', k=5, epochs=10, patience=100, lr=0.01, batch_size=64, kernel=None, model_save_path="best_gpr_model.pt", model_num=None):
    """
    Perform K-fold cross-validation on a GPR model and save the best model.

    Args:
        features_file (str): Path to the CSV file containing the feature data.
        labels_file (str): Path to the CSV file containing the labels.
        label_column (str): The name of the label column to extract from the labels file.
        k (int): Number of folds for cross-validation.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        kernel (gpytorch.kernels.Kernel, optional): The kernel to use in the GP model. Defaults to Matérn kernel.
        model_save_path (str): Path to save the best model.

    Returns:
        None: Prints the training and validation mean squared error (MSE) for each fold and saves the best model.
    """
    # Load the dataset for DNN (returns ProteinDataset)
    dataset, scaler = load_dataset(features_file, labels_file, label_column=label_column, model='dnn', scaler=False)

    # Set up K-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    train_mses = []
    val_mses = []
    best_val_mse = float('inf')
    # model_states = []  # Store model weights for each fold
    
    for train_idx, val_idx in kf.split(dataset):
        fold += 1
        print(f"Fold {fold}")
        
        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        
        # Create DataLoader objects
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        input_dim = dataset.features.shape[1]  # Input feature dimension
        output_dim = 1
        dim_list = [input_dim, 128, 64, 32]  # Customize layers based on needs
        dnn_model = DNN(dim_list, output_dim)
 # Initialize the Trainer
        trainer = Trainer(dnn_model, learning_rate=lr, epoch=epochs, batch_size=batch_size)
        log = trainer.train(train_loader, val_loader, early_stop=True)
        
        # Evaluate the model
        train_mse = trainer.evaluate(train_loader)
        val_mse = trainer.evaluate(val_loader)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        
        print(f"Train MSE: {train_mse}, Val MSE: {val_mse}")
        
        # Save the best model based on validation performance
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = dnn_model.state_dict()
            
        dnn_model.eval()
        feat_pred = dataset.features[val_idx]
        val_pred = dataset.labels[val_idx]
        ypred = dnn_model(feat_pred).detach().numpy()
        ylab  = val_pred.detach().numpy()

        
        
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        ax.scatter(ylab, ypred, color='b', alpha=0.5)
        ax.set_xlabel('True Values', fontsize=4)
        ax.set_ylabel('Predicted Values', fontsize=4)
        ax.set_title('NN model performance', fontsize=6)
        #ax.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
        ax.plot([min(ylab), max(ylab)], [min(ylab), max(ylab)], 'r--')
        ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
        #ax.text(0.1, 0.9, f'RMSE from CV = {rmse:.4f}', transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        plt.show()
        if best_model is not None:
            if model_num is not None:
                save_chkpt(model_save_path + f"/Master_NN_{label_column}_{model_num}.pt", dnn_model, trainer.optimizer, log["val_losses"], log["losses"])
                print(f"Best model saved with validation MSE: {best_val_mse}")
            else:
                save_chkpt(model_save_path + f"/Master_NN_{label_column}.pt", dnn_model, trainer.optimizer, log["val_losses"], log["losses"])
                print(f"Best model saved with validation MSE: {best_val_mse}")


        # model_states.append(dnn_model.state_dict())

    # Average the model weights across all folds
    # averaged_model_state = average_model_weights(model_states)

    # Load the averaged weights into the model
    best_model = DNN(dim_list, output_dim)
    if model_num is not None:
        best_dict = torch.load(model_save_path + f"/Master_NN_{label_column}_{model_num}.pt")
    else:
        best_dict = torch.load(model_save_path + f"/Master_NN_{label_column}.pt")
    best_model.load_state_dict(best_dict['model'])

    print(f"Final k-fold Results: Train MSE: {np.mean(train_mses)} ± {np.std(train_mses)}, Val MSE: {np.mean(val_mses)} ± {np.std(val_mses)}")

    feats, labels = dataset.features, dataset.labels

    train_X, test_X, train_y, test_y = train_test_split(feats, labels, test_size=0.2, shuffle=True)

    best_model.eval()

    test_pred  = best_model(test_X).detach().numpy()
    train_pred = best_model(train_X).detach().numpy()
    
    labels_test  = test_y.detach().numpy()
    labels_train = train_y.detach().numpy()

    if label_column == 'diff':
        ax_lab = 'D'
    elif label_column == 'B2':
        ax_lab = '$B_2$'
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    ax.scatter(labels_train, train_pred, color='orange', alpha=0.3, label="Train")
    ax.scatter(labels_test, test_pred, color='b', alpha=0.3, label="Test")
    ax.set_xlabel(f'True {ax_lab}', fontsize=6)
    ax.set_ylabel(f'Predicted {ax_lab}', fontsize=6)
    ax.set_title('NN model performance', fontsize=6)
    #ax.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
    ax.plot([min(labels_train), max(labels_train)], [min(labels_train), max(labels_train)], 'r--')
    ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
    #ax.text(0.1, 0.9, f'RMSE from CV = {rmse:.4f}', transform=ax.transAxes, fontsize=10)
    plt.legend(fontsize=6)
    fig.tight_layout()
    if model_num is not None:
        fig.savefig(model_save_path + f"/Master_NN_FinalFit_{label_column}_{model_num}.png", dpi=300, bbox_inches='tight')
    else:
        fig.savefig(model_save_path + f"/Master_NN_FinalFit_{label_column}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save final model
    # save_chkpt(f"models/Master_NN_{label_column}.pt", averaged_model, final_trainer.optimizer, final_log["val_losses"], final_log["train_losses"])
    # print(f"Final model saved with training on the entire dataset")


def KFold_RF_grid(features_file, labels_file, label_column="phase", k=5, param_grid=None, model_save_path="best_rf_model.pkl"):
    """
    Perform K-fold cross-validation on a Random Forest classifier with hyperparameter tuning and save the best model.

    Args:
        features_file (str): Path to the CSV file containing the feature data.
        labels_file (str): Path to the CSV file containing the labels.
        label_column (str): The name of the label column to extract from the labels file.
        k (int): Number of folds for cross-validation.
        param_grid (dict): Hyperparameters to tune.
        model_save_path (str): Path to save the best model.

    Returns:
        None: Prints the training and validation accuracy for each fold and saves the best model.
    """
    # Load the dataset
    dataset = load_classification_dataset(features_file, labels_file, label_column)
    features = dataset.features
    labels = dataset.labels

    # Set up K-Fold Cross Validation
    kf = KFold(n_splits=k, shuffle=True)
    
    best_val_accuracy = 0
    best_model = None

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier()  # We will use this as a base model

    # Set up Grid Search with K-Fold Cross-Validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=kf, scoring='accuracy', verbose=50)

    # Fit the model using Grid Search
    grid_search.fit(features, labels)

    # Retrieve the best model and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_val_accuracy = grid_search.best_score_

    print("Best parameters found: ", best_params)
    print("Best cross-validation score: ", best_val_accuracy)

    # Train-Test Split for final evaluation
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, shuffle=True)

    # Train the final RF model using the best parameters
    final_rf_model = best_model
    final_rf_model.fit(features, labels)

    # Evaluate on the test set
    test_preds = final_rf_model.predict(test_X)
    test_accuracy = accuracy_score(test_y, test_preds)
    print(f"Test Accuracy: {test_accuracy}")

    # Confusion matrix
    cm = confusion_matrix(test_y, test_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(model_save_path + '_ConfMatrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save the final model
    save_rf_model(model_save_path + '.pkl', final_rf_model)
    print(f"Final model saved with test accuracy: {test_accuracy}")