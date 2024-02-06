import pytorch_lightning as pl
import torch
from mm79.train_modules.loss_utils import y_to_seq
from mm79.train_modules.event_utils import time_censoring_proc, cumulative_dyn_auc
from mm79.train_modules.train_module import *


class DMMTrainModule(TrainModule):

    def __init__(self, model_cls,
                 input_long_size,
                 baseline_size,
                 treatment_size,
                 prediction_idx,
                 input_idx,
                 lr,
                 weight_decay,
                 t_cond,
                 t_horizon,
                 emission_proba,
                 quantiles_event,
                 et_train_event,
                 quantiles_ae,
                 et_train_ae,
                 **kwargs):
        super().__init__(model_cls, input_long_size, baseline_size, treatment_size,
                         prediction_idx, input_idx, lr, weight_decay, t_cond, t_horizon,
                         emission_proba, quantiles_event, et_train_event, quantiles_ae, 
                         et_train_ae, **kwargs)

        self.save_hyperparameters()

        # all the same as prior TrainModule

    def forward(self, B, X, M, T, times, times_forward, T_forward):
        losses = self.model(
            B=B, X=X, M=M, T=T, times=times,
            times_forward=times_forward, T_forward=T_forward,
            Y=None, E=None)
        return losses

    def training_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(batch)

        (nelbo, nll, kl, _), loss, y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )

        if self.event_loss:
            loss_event = self.compute_event_ll(
                y_pred[:, -self.t_horizon:, :], E, Y)
        else:
            loss_event = 0

        loss = (1-self.lambda_reg) * loss + self.lambda_reg * loss_event

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_nelbo", nelbo, on_epoch=True, on_step=True)
        self.log("train_nll", nll, on_epoch=True, on_step=True)
        self.log("train_kl", kl, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(batch)

        (nelbo, nll, kl, _), loss,  y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )

        if self.event_loss:
            loss_event = self.compute_event_ll(
                y_pred[:, -self.t_horizon:, :], E, Y)
        else:
            loss_event = 0

        loss = (1-self.lambda_reg) * loss + self.lambda_reg * loss_event

        self.log("val_loss", loss, on_epoch=True, on_step=True)
        self.log("val_nelbo", nelbo, on_epoch=True, on_step=True)
        self.log("val_nll", nll, on_epoch=True, on_step=True)
        self.log("val_kl", kl, on_epoch=True, on_step=True)

        return {"val_loss": loss,
                "val_nelbo": nelbo,
                "y_pred": y_pred,
                "Y": Y,
                "E": E}

    def test_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(batch)

        (nelbo, nll, kl, _), loss, y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )
        if self.event_loss:
            loss_event = self.compute_event_ll(
                y_pred[:, -self.t_horizon:, :], E, Y)
        else:
            loss_event = 0

        loss = (1-self.lambda_reg) * loss + self.lambda_reg * loss_event

        self.log("test_loss", loss, on_epoch=True, on_step=True)
        self.log("test_nelbo", nelbo, on_epoch=True, on_step=True)
        self.log("test_nll", nll, on_epoch=True, on_step=True)
        self.log("test_kl", kl, on_epoch=True, on_step=True)

        return {"test_loss": loss,
                "test_nelbo": nelbo,
                "y_pred": y_pred,
                "Y": Y,
                "E": E}

    def predict_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(batch)

        X_future = X_cond[:, self.predict_cond:self.predict_cond +
                          self.predict_horizon, :]
        M_future = M_cond[:, self.predict_cond:self.predict_cond +
                          self.predict_horizon, :]
        T_future = T_cond[:, self.predict_cond:self.predict_cond +
                          self.predict_horizon, :]
        times_future = times_cond[:,
                                  self.predict_cond:self.predict_cond + self.predict_horizon]

        X_cond = X_cond[:, :self.predict_cond, :]
        M_cond = M_cond[:, :self.predict_cond, :]
        T_cond = T_cond[:, :self.predict_cond, :]
        times_cond = times_cond[:, :self.predict_cond]
        forecast, y_pred, y_pred_cond = self.model.sample(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future,
            Y=Y,
            E=E,
            prediction_idx=self.prediction_idx
        )

        if y_pred is not None:
            y_pred = y_pred[:, :self.predict_cond]

        forecast_future = forecast[:,
                                   self.predict_cond:self.predict_cond + self.predict_horizon, :]
        loss_mse = self.compute_mse(
            forecast_future[:, :, :],
            X_future[:, :, self.prediction_idx],
            M_future[:, :, self.prediction_idx]
        )

        return {"loss": loss_mse,
                "forecast_future": forecast_future,
                "target_future": X_future,
                "mask_future": M_future,
                "X_cond": X_cond,
                "M_cond": M_cond,
                "prediction_idx": self.prediction_idx,
                "times_cond": times_cond,
                "times_future": times_future,
                "y_pred": y_pred,
                "y_pred_cond": y_pred_cond,
                "Y": Y,
                "E": E,
                "Y_ae": Y_ae,
                "E_ae": E_ae,
                "B": B, 
                "y_hidden": None, 
                "pids": batch[0],
                "treat_flag": batch[-1]}