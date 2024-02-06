import pytorch_lightning as pl
import torch
import numpy as np
from mm79.train_modules.loss_utils import y_to_seq
from mm79.train_modules.event_utils import time_censoring_proc, cumulative_dyn_auc
from pycox.models.loss import cox_ph_loss
from sksurv.metrics import concordance_index_ipcw, brier_score, concordance_index_censored
from mm79 import EVENT_SHIFT


class BaseModule(pl.LightningModule):
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
                 lambda_reg=1.0,
                 emission_window=1,
                 event_type="pfs",
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_cond = t_cond
        self.t_horizon = t_horizon

        self.lambda_reg = lambda_reg

        self.emission_window = emission_window

        # input and prediction indices
        self.prediction_idx = prediction_idx
        self.input_idx = input_idx

        # semi-supervised flag
        if emission_proba:
            self.event_loss = True
        else:
            self.event_loss = False

        # eval options
        self.quantiles_event = quantiles_event
        self.et_train_event = et_train_event
        self.quantiles_ae = quantiles_ae
        self.et_train_ae = et_train_ae

        # What event to predict
        self.event_type = event_type

        self.treatment_forcing = None

        self.predict_horizon = t_horizon

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

    def forward(self, B, X, M, T, times, times_forward, T_forward):
        processed_sequence = self.model(
            B=B, X=X, M=M, T=T, times=times,
            times_forward=times_forward, T_forward=T_forward,
            Y=None, E=None)
        return processed_sequence

    def training_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(
                batch)

        forecast, y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )

        if self.t_cond != -1:
            forecast_future = forecast[:, -self.t_horizon:, :]
            forecast_cond = forecast[:, :self.t_cond, :]

            loss_y = self.compute_mse(
                forecast_future,
                X_future[:, :, self.prediction_idx],
                M_future[:, :, self.prediction_idx]
            )
            loss_x = self.compute_mse(
                forecast_cond,
                X_cond[:, :, self.prediction_idx],
                M_cond[:, :, self.prediction_idx]
            )
            loss_mse = loss_x + loss_y

            if self.event_loss:
                loss_event = self.compute_event_ll(
                    y_pred[:, -self.t_horizon:, :], E, Y, E_ae, Y_ae)
            else:
                loss_event = 0

        else:
            loss_mse = self.compute_mse(
                forecast[:, :-1, :],
                X_cond[:, 1:, self.prediction_idx],
                M_cond[:, 1:, self.prediction_idx]
            )

            if self.event_loss:
                loss_event = self.compute_event_ll(y_pred, E, Y, E_ae, Y_ae)
            else:
                loss_event = 0

        loss = (1-self.lambda_reg) * loss_mse + self.lambda_reg * loss_event

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_mse", loss_mse, on_epoch=True, on_step=True)
        self.log("train_event", loss_event, on_epoch=True, on_step=True)

        return loss

    def compute_mse(self, pred, target, mask):
        mse = ((pred-target).pow(2)*mask).sum()
        return mse / mask.sum()

    def compute_event_ll_(self, y_pred, E, Y):
        loss_event = 0
        for t in range(y_pred.shape[1]):
            idx_at_risk = (Y[:, 0]-t >= 0)
            if (idx_at_risk.sum() <= 5) or (E[idx_at_risk].sum() == 0):
                continue
            loss_t = cox_ph_loss(
                log_h=y_pred[idx_at_risk, t, 0], durations=Y[idx_at_risk, 0]-t, events=E[idx_at_risk, 0])
            loss_event = loss_event + loss_t
        return loss_event
#

    def compute_event_ll(self, y_pred, E, Y, E_ae, Y_ae):
        event_pfs = self.compute_event_ll_(y_pred[:, :, 0][..., None], E, Y)
        event_ae = 0

        # UNCOMMENT THIS TO TRAIN THE AE.
        for ae_idx in range(E_ae.shape[1]):
            event_ae = event_ae + self.compute_event_ll_(
                y_pred[:, :, ae_idx+1][..., None], E_ae[:, ae_idx][:, None], Y_ae[:, ae_idx][:, None])

        return event_pfs + event_ae

    def parse_batch(self, batch):
        pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, \
            T_cond, T_future, E, Y, E_ae, Y_ae, times_cond, times_future, Treat_flag = batch

        if self.treatment_forcing is not None:
            T_cond[:, :, 0:3] = torch.Tensor(
                self.treatment_forcing).to(T_cond.device)
            T_future[:, :, 0:3] = torch.Tensor(
                self.treatment_forcing).to(T_future.device)
            if self.treatment_forcing[-1] == 0.:
                B[:, 40] = -1.  # treatment index is 40
            else:
                B[:, 40] = 1.

        return B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, E_ae, Y_ae, times_cond, times_future

    def validation_step(self, batch, batch_idx):

        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(
                batch)

        forecast, y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )

        if self.t_cond != -1:
            forecast_future = forecast[:, -self.t_horizon:, :]
            loss_mse = self.compute_mse(
                forecast_future,
                X_future[:, :, self.prediction_idx],
                M_future[:, :, self.prediction_idx]
            )

            if self.event_loss:
                loss_event = self.compute_event_ll(
                    y_pred[:, -self.t_horizon:, :], E, Y, E_ae, Y_ae)
            else:
                loss_event = 0
        else:
            loss_mse = self.compute_mse(
                forecast[:, :-1, :],
                X_cond[:, 1:, self.prediction_idx],
                M_cond[:, 1:, self.prediction_idx]
            )

            if self.event_loss:
                loss_event = self.compute_event_ll(y_pred, E, Y, E_ae, Y_ae)
            else:
                loss_event = 0

        loss = (1-self.lambda_reg) * loss_mse + self.lambda_reg * loss_event
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_loss_mse", loss_mse, on_epoch=True)
        self.log("val_loss_event", loss_event, on_epoch=True)

        return {"val_loss": loss,
                "val_loss_mse": loss_mse,
                "val_loss_event": loss_event,
                "y_pred": y_pred,
                "Y": Y,
                "E": E}

    def validation_epoch_end(self, outputs):

        if (self.event_loss) and (self.t_cond != -1):
            Y = torch.cat([x["Y"] for x in outputs], dim=0)
            E = torch.cat([x["E"] for x in outputs], dim=0)
            y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
            roc_aucs = self.evaluate_event_auc(
                y_pred, Y, E, t_cond=self.t_cond)

            self.log("val_roc_auc_25", roc_aucs[0], on_epoch=True)
            self.log("val_roc_auc_50", roc_aucs[1], on_epoch=True)
            self.log("val_roc_auc_75", roc_aucs[2], on_epoch=True)

    def test_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(
                batch)

        forecast, y_pred = self.forward(
            B=B,
            X=X_cond[:, :, self.input_idx],
            M=M_cond[:, :, self.input_idx],
            T=T_cond,
            times=times_cond,
            times_forward=times_future,
            T_forward=T_future
        )

        if self.t_cond != -1:
            forecast_future = forecast[:, -self.t_horizon:, :]
            loss_mse = self.compute_mse(
                forecast_future,
                X_future[:, :, self.prediction_idx],
                M_future[:, :, self.prediction_idx]
            )
            if self.event_loss:
                loss_event = self.compute_event_ll(
                    y_pred[:, -self.t_horizon:, :], E, Y, E_ae, Y_ae)
            else:
                loss_event = 0
        else:
            loss_mse = self.compute_mse(
                forecast[:, :-1, :],
                X_cond[:, 1:, self.prediction_idx],
                M_cond[:, 1:, self.prediction_idx]
            )
            if self.event_loss:
                loss_event = self.compute_event_ll(y_pred, E, Y, E_ae, Y_ae)
            else:
                loss_event = 0

        loss = (1-self.lambda_reg) * loss_mse + self.lambda_reg * loss_event
        #loss = loss_mse + loss_event
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_loss_mse", loss_mse, on_epoch=True)
        self.log("test_loss_event", loss_event, on_epoch=True)

        return {"test_loss": loss,
                "test_loss_mse": loss_mse,
                "test_loss_event": loss_event,
                "y_pred": y_pred,
                "Y": Y,
                "E": E}

    def test_epoch_end(self, outputs):

        if (self.event_loss) and (self.t_cond != -1):
            Y = torch.cat([x["Y"] for x in outputs], dim=0)
            E = torch.cat([x["E"] for x in outputs], dim=0)
            y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)

            roc_aucs = self.evaluate_event_auc(
                y_pred, Y, E, t_cond=self.t_cond)

            self.log("test_roc_auc_25", roc_aucs[0], on_epoch=True)
            self.log("test_roc_auc_50", roc_aucs[1], on_epoch=True)
            self.log("test_roc_auc_75", roc_aucs[2], on_epoch=True)

    def evaluate_event_auc(self, event_risk, Y, E, t_cond):
        # @TODO : check this with AE.
        if t_cond is None:
            t_cond = self.t_cond - EVENT_SHIFT
        else:
            t_cond = t_cond - EVENT_SHIFT

        et_test = time_censoring_proc(
            E[Y > t_cond].cpu(), Y[Y > t_cond].cpu() - t_cond)
        event_risk = event_risk[Y > t_cond]
        et_train = self.et_train_event

        roc_auc_25 = cumulative_dyn_auc(
            et_train, et_test, event_risk.cpu(), self.quantiles_event[0])[0][0]
        roc_auc_50 = cumulative_dyn_auc(
            et_train, et_test, event_risk.cpu(), self.quantiles_event[1])[0][0]
        roc_auc_75 = cumulative_dyn_auc(
            et_train, et_test, event_risk.cpu(), self.quantiles_event[2])[0][0]

        return [roc_auc_25, roc_auc_50, roc_auc_75]

    def evaluate_concordance_index(self, event_risk, Y, E, Y_ae, E_ae, t_cond):
        cis_event = self.evaluate_concordance_index_(
            event_risk[:, 0][:, None], Y, E, t_cond, et_train=self.et_train_event)
        cis_ae = [self.evaluate_concordance_index_(event_risk[:, i+1][:, None], Y_ae[:, i][:, None],
                                                   E_ae[:, i][:, None], t_cond, et_train=self.et_train_ae[i]) for i in range(event_risk.shape[1]-1)]
        return cis_event, cis_ae

    def evaluate_concordance_index_(self, event_risk, Y, E, t_cond, et_train):
        if t_cond is None:
            t_cond = self.t_cond - EVENT_SHIFT
        else:
            t_cond = t_cond - EVENT_SHIFT

        if torch.sum(E[Y > t_cond]) == 0:
            return np.nan

        et_test = time_censoring_proc(
            E[Y > t_cond].cpu(), Y[Y > t_cond].cpu() - t_cond)
        event_risk = event_risk[Y > t_cond]

        try:
            concordance_index = concordance_index_ipcw(
                et_train, et_test, event_risk.cpu())[0]
        except:
            concordance_index = np.nan
        
        return concordance_index

    def set_predict_cond_horizon(self, t_cond, t_horizon, quantiles_event, et_train_event, quantiles_ae, et_train_ae):
        self.predict_cond = t_cond
        self.predict_horizon = t_horizon

        self.quantiles_event = quantiles_event
        self.et_train_event = et_train_event

        self.quantiles_ae = quantiles_ae
        self.et_train_ae = et_train_ae

    def set_treatment_forcing(self, treatment_forcing):
        """ Forces the treatment variables to a particular value

        Args:
            treatment_forcing (_type_): value to force the treatment variables to. If None, no forcing is applied.
        """
        self.treatment_forcing = treatment_forcing

    def predict_step(self, batch, batch_idx):
        B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
            T_future, E, Y, E_ae, Y_ae, times_cond, times_future = self.parse_batch(
                batch)

        if self.t_cond == -1:
            X_future = X_cond[:, self.predict_cond:self.predict_cond +
                              self.predict_horizon, :]
            M_future = M_cond[:, self.predict_cond:self.predict_cond +
                              self.predict_horizon, :]
            T_future = T_cond[:, self.predict_cond:self.predict_cond +
                              self.predict_horizon + self.emission_window - 1, :]
            times_future = times_cond[:,
                                      self.predict_cond:self.predict_cond + self.predict_horizon + self.emission_window - 1]

            X_cond = X_cond[:, :self.predict_cond, :]
            M_cond = M_cond[:, :self.predict_cond, :]
            T_cond = T_cond[:, :self.predict_cond, :]
            times_cond = times_cond[:, :self.predict_cond]
            # TODO : model sample should also return y_preds !
            forecast, y_pred, y_pred_cond, y_hidden = self.model.sample(
                B=B,
                X=X_cond[:, :, self.input_idx],
                M=M_cond[:, :, self.input_idx],
                T=T_cond[:, :],
                times=times_cond[:, :],
                times_forward=times_future,
                T_forward=T_future,
                Y=Y,
                E=E,
                return_y_hidden=True
            )

            forecast_future = forecast[:,
                                       self.predict_cond:self.predict_cond + self.predict_horizon, :]
            loss_mse = self.compute_mse(
                forecast_future,
                X_future[:, :, self.prediction_idx],
                M_future[:, :, self.prediction_idx]
            )
        else:
            forecast, y_pred = self.forward(
                B=B,
                X=X_cond[:, :, self.input_idx],
                M=M_cond[:, :, self.input_idx],
                T=T_cond,
                times=times_cond,
                times_forward=times_future,
                T_forward=T_future
            )
            y_pred_cond = None
            y_hidden = None
            forecast_future = forecast[:, -self.t_horizon:, :]
            if y_pred is not None:
                y_pred = y_pred[:, -self.t_horizon-1:-1]

            loss_mse = self.compute_mse(
                forecast_future,
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
                "times_future": times_future[:, :self.predict_horizon],
                "y_pred": y_pred,
                "y_pred_cond": y_pred_cond,
                "Y": Y,
                "E": E,
                "Y_ae": Y_ae,
                "E_ae": E_ae,
                "B": B,
                "y_hidden": y_hidden,
                "pids": batch[0],
                "treat_flag": batch[-1]
                }

    @ classmethod
    def defaults(cls):
        return {
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout_p": 0.0,
            "lambda_reg": 0.5
        }

    @ classmethod
    def add_module_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--lr', type=float, default=cls.defaults()["lr"])
        parser.add_argument('--weight_decay', type=float,
                            default=cls.defaults()["weight_decay"])
        parser.add_argument('--dropout_p', type=float,
                            default=cls.defaults()["dropout_p"])
        parser.add_argument('--lambda_reg', type=float,
                            default=cls.defaults()["lambda_reg"], help="Regularization parameter for the loss function. Loss = lambda * event_loss + (1-lambda) * mse_loss")
        return parser


class FineTuneModule(BaseModule):
    def __init__(self, finetune_model_cls, pretrained_model, **kwargs):
        super().__init__(**kwargs)
        self.model = finetune_model_cls(
            pretrained_model,
            emission_type=kwargs.get("emission_type", "non_linear"),
            include_baseline=kwargs.get("include_baseline", False),
            include_last=kwargs.get("include_last", False),
            emission_window=kwargs.get("emission_window", 1)
        )


class TrainModule(BaseModule):
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
                 lambda_reg=1.0,
                 event_type="pfs",
                 **kwargs):
        super().__init__(model_cls,
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
                         lambda_reg=lambda_reg,
                         event_type=event_type,
                         **kwargs)
        # model
        self.model = model_cls(input_long_size=input_long_size,
                               baseline_size=baseline_size,
                               treatment_size=treatment_size,
                               prediction_idx=prediction_idx,
                               input_idx=input_idx,
                               t_cond=t_cond,
                               t_horizon=t_horizon,
                               emission_proba=emission_proba,
                               **kwargs)
