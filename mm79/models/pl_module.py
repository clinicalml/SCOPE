import numpy as np
import math
from utils.utils import str2bool
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
sys.insert.append("../")
import sys


# Below code is just an example to make sure that the other thing works
###


class Transformer(pl.LightningModule):
    def __init__(self, input_long_size, hidden_dim, baseline_size, lr, trans_layers, weight_decay, T_cond, T_horizon, reconstruction_size, planned_treatments, max_pos_encoding, d_pos_embed, nheads, ff_dim, output_dims, treatment_dim=0, dropout_p=0,  nhead_treat=0, survival_loss=0, treat_attention_type=None,  linear_type="normal", **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.transformer_layers = trans_layers

        self.treat_attention_type = treat_attention_type

        if nhead_treat > 0:
            self.treatment_heads = True
        else:
            # no real treatment heads but use treatment information
            if (self.treat_attention_type == "linear") or (self.treat_attention_type == "mlp"):
                self.treatment_heads = True
            else:
                self.treatment_heads = False

        self.lr = lr
        self.weight_decay = weight_decay

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        self.model = Treatformer(temporal_input_size=input_long_size, baseline_size=baseline_size, hidden_dim=hidden_dim, treatment_size=treatment_dim, num_layers=trans_layers, T_cond=T_cond, T_horizon=T_horizon,
                                 planned_treatments=planned_treatments, max_pos_encoding=max_pos_encoding, dim_pos_embed=d_pos_embed, nheads=nheads, nhead_treat=nhead_treat, treat_attention_type=treat_attention_type)

        if self.treat_attention_type == "mlp":
            self.mlp_head = True
        else:
            self.mlp_head = False

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, B, X, M, T, times, times_forward, T_forward):

        processed_sequence = self.model(
            B=B, X=X, M=M, T=T, times=times, times_forward=times_forward, T_forward=T_forward, Y=None, E=None)

        return processed_sequence

    def training_step(self, batch, batch_idx):

        B0, X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(
            batch)

        forecast = self.forward(
            B=B0, X=X_long, M=M_long, T=A_long, times=times_X, times_forward=times_Y, T_forward=A_forward)
        return {"loss": loss}

    def compute_mse(self, pred, target, mask):

        if isinstance(pred, list):
            mse = torch.stack([((pred[i]-target[i]).pow(2)*mask[i]).sum()
                              for i in range(len(pred))]).sum() / torch.stack([m.sum() for m in mask]).sum()
            return mse
        else:
            if mask.sum() == 0:
                # This is to deal with batches with no samples in them
                return 0 * pred.sum()
            return ((pred-target).pow(2)*mask).sum()/mask.sum()

    def parse_batch(self, batch):
        X, Y, T, Y_cf, p, B, Ax, Ay, Mx, My, times_X, times_Y, Y_countdown, CE = batch
        #A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
        A_future = Ay
        return B, X.permute(0, 2, 1), Mx.permute(0, 2, 1), Ax.permute(0, 2, 1), Y, My.permute(0, 2, 1), A_future.permute(0, 2, 1), times_X, times_Y, Y_countdown, CE

    def validation_step(self, batch, batch_idx):

        B0, X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE = self.parse_batch(
            batch)

        forecast = self.forward(
            B=B0, X=X_long, M=M_long, T=A_long, times=times_X, times_forward=times_Y, T_forward=A_forward)

    def test_step(self, batch, batch_idx):

        B0, X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(
            batch)

        long_input_x = self.get_long_input_x(X_long, M_long, A_long, mask_pre)
        long_input_y = self.get_long_input_y(
            X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post)
        times_full = self.get_times_full(times_X, times_Y)

        forecast, forecast_surv = self.forward(
            B0, long_input_x, long_input_y, times_full, mask_pre, mask_post)
        loss_X, loss_Y, loss_surv, forecast_X, forecast_Y = self.compute_losses(
            forecast=forecast, forecast_surv=forecast_surv, X_long=X_long, M_long=M_long, X_forward=X_forward, M_forward=M_forward, Y_countdown=Y_countdown, CE=CE, mask_pre=mask_pre, mask_post=mask_post)
        #forecast_X = forecast[:,:,:X_long.shape[-1]]
        #forecast_Y = forecast[:,self.output_dims,X_long.shape[-1]:]

        #loss_X = self.compute_mse(forecast_X,X_long,M_long)
        #loss_Y = self.compute_mse(forecast_Y,X_forward[:,self.output_dims],M_forward[:,self.output_dims])

        loss = loss_X + loss_Y + loss_surv
        self.log("test_loss", loss, on_epoch=True)

        self.log("test_loss_X", loss_X, on_epoch=True)
        self.log("test_loss_Y", loss_Y, on_epoch=True)
        self.log("test_loss_surv", loss_surv, on_epoch=True)

        return {"loss": loss, "Y_pred": forecast_Y, "Y": X_forward, "M": M_forward}

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=48)
        parser.add_argument('--trans_layers', type=int, default=2)
        parser.add_argument('--nheads', type=int, default=8,
                            help=" total number of heads in the transformer (normal + treat heads)")
        parser.add_argument('--ff_dim', type=int, default=512,
                            help="hidden_units of the feed-forward in the transformer architecture")
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--dropout_p', type=float, default=0.)
        parser.add_argument('--planned_treatments',
                            type=str2bool, default=False)
        parser.add_argument('--nhead_treat', type=int,
                            default=0, help="number of treatment heads to use")
        parser.add_argument('--max_pos_encoding', type=int, default=100,
                            help="Maximum time (used for computing the continuous positional embeddings)")
        parser.add_argument('--d_pos_embed', type=int, default=10,
                            help="Dimension of the positional embedding")
        parser.add_argument('--treat_attention_type', type=str,
                            default=None, help="Expert or Linear")
        parser.add_argument('--linear_type', type=str,
                            default="normal", help="normal or all")
        return parser
