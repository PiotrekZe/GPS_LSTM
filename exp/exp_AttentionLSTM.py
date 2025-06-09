from data_provider.data_loader import Dataset_GPS
from models.AttentionLSTM import AttentionLSTM
from models.GRU import ConfigurableGRUModel as GRU
from models.LSTM import ConfigurableLSTMModel as LSTM
import os
from utils.metrics import MSE, MAE, RMSE
import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils.read_data import read_data

"""
To do:
1. Add model saving and loading functionality.
2. Implement early stopping based on validation loss.
3. Add logging functionality to track training progress.
4. Add command-line arguments for model configuration.
5. Add .sh script for running the training and testing.
"""


class Exp_AttentionLSTM:
    """
    Organizes training and testing for AttentionLSTM models.
    """

    def __init__(
        self,
        model_name=None,
        optimizer=None,
        criterion=None,
        device=None,
        batch_size=None,
        data=None,
        labels=None,
    ):
        self.model_name = model_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = 64

    def _get_data(self, loro_itr):
        """
        Load the dataset for training or testing.
        """
        self.data, self.labels = read_data()  # add params
        idx_tab = []
        if loro_itr == 0:
            idx_tab = [0, 1, 2]
        elif loro_itr == 1:
            idx_tab = [0, 2, 1]
        elif loro_itr == 2:
            idx_tab = [1, 2, 0]

        a, b, c = idx_tab
        X_train = np.append(
            self.data[a][: int(self.data[a].shape[0] * 0.8)],
            self.data[b][: int(self.data[b].shape[0] * 0.8)],
            axis=0,
        )
        y_train = np.append(
            self.labels[a][: int(self.labels[a].shape[0] * 0.8)],
            self.labels[b][: int(self.labels[b].shape[0] * 0.8)],
            axis=0,
        )

        X_validation = np.append(
            self.data[a][int(self.data[a].shape[0] * 0.8) :],
            self.data[b][int(self.data[b].shape[0] * 0.8) :],
            axis=0,
        )
        y_validation = np.append(
            self.labels[a][int(self.labels[a].shape[0] * 0.8) :],
            self.labels[b][int(self.labels[b].shape[0] * 0.8) :],
            axis=0,
        )

        X_test, y_test = self.data[c], self.labels[c]

        train_dataset = Dataset_GPS(
            data=X_train,
            labels=y_train,
        )
        val_dataset = Dataset_GPS(
            data=X_validation,
            labels=y_validation,
        )
        test_dataset = Dataset_GPS(
            data=X_test,
            labels=y_test,
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader

    def train(self, loro_itr=0):
        """
        Train the model for one epoch.
        """
        train_loader, val_loader, test_loader = self._get_data(loro_itr)

        model_dict = {
            "AttentionLSTM": AttentionLSTM,
            "GRU": GRU,
            "LSTM": LSTM,
        }
        print(self.model_name)
        self.model = model_dict[self.model_name](
            seq_len=100,
            input_dim=2,
            hidden_dim=64,
            output_dim=2,
            num_layers=3,
            pred_len=200,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.L1Loss(reduction="mean")

        for epoch in range(5):
            self.model.train()
            epoch_time = time.time()
            running_loss = 0.0
            total_samples = 0
            preds_tab, targets_tab = [], []

            for i, (inputs, targets) in enumerate(train_loader):
                # print(inputs.shape, targets.shape)
                inputs = inputs.transpose(1, 2).to(self.device)
                targets = targets.transpose(1, 2).to(self.device)

                self.optimizer.zero_grad()
                batch_outputs = self.model(inputs)
                bs = inputs.shape[0]
                loss = self.criterion(batch_outputs, targets)
                running_loss += loss.item() * bs

                preds_tab.append(batch_outputs.detach().cpu().numpy())
                targets_tab.append(targets.detach().cpu().numpy())

                total_samples += bs

                loss.backward()
                self.optimizer.step()

            preds_tab = np.concatenate(preds_tab, axis=0)
            targets_tab = np.concatenate(targets_tab, axis=0)
            mse_val = MSE(pred=preds_tab, true=targets_tab)
            mae_val = MAE(pred=preds_tab, true=targets_tab)
            rmse_val = RMSE(pred=preds_tab, true=targets_tab)

            print(
                f"TRAIN LOSS: {running_loss/total_samples}. Epoch time: {time.time() - epoch_time}s"
            )
            print(f"MSE: {mse_val}, MAE: {mae_val}, RMSE: {rmse_val}")

            vallidation_loss, validation_mse, validation_mae, vlaidation_rmse = (
                self.test(val_loader, self.criterion)
            )
            test_loss, test_mse, test_mae, tets_rmse = self.test(
                test_loader, self.criterion
            )

            # Save the model if validation loss improves
            # Save model checkpoint
            # Save results to a file
            # Change evaluation - test set used for final evaluation on best model
            # Add early stopping based on validation loss

    def test(self, data_loader, criterion):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()

        running_loss = 0.0
        total_samples = 0
        preds_tab, targets_tab = [], []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.transpose(1, 2).to(self.device)
                targets = targets.transpose(1, 2).to(self.device)

                batch_outputs = self.model(inputs)
                bs = inputs.shape[0]
                loss = criterion(batch_outputs, targets)
                running_loss += loss.item() * bs

                preds_tab.append(batch_outputs.detach().cpu().numpy())
                targets_tab.append(targets.detach().cpu().numpy())

                total_samples += bs

        preds_tab = np.concatenate(preds_tab, axis=0)
        targets_tab = np.concatenate(targets_tab, axis=0)
        mse_val = MSE(pred=preds_tab, true=targets_tab)
        mae_val = MAE(pred=preds_tab, true=targets_tab)
        rmse_val = RMSE(pred=preds_tab, true=targets_tab)

        print(f"TEST LOSS: {running_loss/total_samples}")
        print(f"MSE: {mse_val}, MAE: {mae_val}, RMSE: {rmse_val}")

        return [running_loss / total_samples, mse_val, mae_val, rmse_val]
