import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from mm79.utils.utils import str2bool
import math
import numpy as np
from mm79 import AE_ALL_DIM  # number of AE in total


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding():
    """Apply positional encoding to instances."""

    def __init__(self, min_timescale, max_timescale, n_channels):
        """PositionalEncoding.
        Args:
            min_timescale: minimal scale of values
            max_timescale: maximal scale of values
            n_channels: number of channels to use to encode position
        """
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.n_channels = n_channels

        self._num_timescales = self.n_channels // 2
        self._inv_timescales = self._compute_inv_timescales()

    def _compute_inv_timescales(self):
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale))
            / (float(self._num_timescales) - 1)
        )
        inv_timescales = (
            self.min_timescale
            * np.exp(
                np.arange(self._num_timescales)
                * -log_timescale_increment
            )
        )
        return torch.Tensor(inv_timescales)

    def __call__(self, times):
        """Apply positional encoding to instances."""
        # instance = instance.copy()  # We only want a shallow copy
        positions = times
        scaled_time = (
            positions[..., None] *
            self._inv_timescales[None, :].to(times.device)
        )
        signal = torch.cat(
            (torch.sin(scaled_time), torch.cos(scaled_time)),
            axis=-1
        )
        return signal


class TransformerEncoderLayer(nn.Module):
    """
    Args:
        d_temporal_in: dimension of features in the  temporal input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
            (default=2048).
        dropout: the dropout value (default=0.1).
        nhead_treat: Number of treatment heads to use.
        treat_attention_type : Type of attention to use (linear, mlp or None)

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_temporal_in, nhead, dim_feedforward=2048, dropout=0.,
                 nhead_treat=0, treat_attention_type=None):
        super(TransformerEncoderLayer, self).__init__()

        self.nhead = nhead
        self.nhead_treat = nhead_treat
        self.d_temporal_in = d_temporal_in

        d_model_head = d_temporal_in // (nhead + nhead_treat)
        self.d_model_head = d_model_head

        self.map_embed = nn.Linear(d_temporal_in, d_model_head*nhead)
        self.self_attn = nn.MultiheadAttention(
            d_model_head * nhead, nhead, dropout=dropout)

        if self.nhead_treat > 0:
            self.map_embed_treat = nn.Linear(
                d_temporal_in, d_model_head*nhead_treat)
            self.map_embed_treat_src = nn.Linear(
                d_temporal_in, d_model_head*nhead_treat)
            self.treat_attn = nn.MultiheadAttention(
                d_model_head * nhead_treat, nhead_treat, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_temporal_in, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_temporal_in)

        def get_residual():
            def residual(x1, x2):
                return x1 + x2
            return residual

        def get_norm():
            return nn.LayerNorm(self.d_temporal_in)

        self.norm1 = get_norm()
        self.norm2 = get_norm()
        self.residual1 = get_residual()
        self.residual2 = get_residual()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self.treat_attention_type = treat_attention_type

        if self.treat_attention_type == "linear":
            self.querry_mod = nn.Linear(d_temporal_in, d_temporal_in)

            self.key_mod = nn.Linear(2*d_temporal_in, d_temporal_in)
            self.value_mod = nn.Linear(2*d_temporal_in, d_temporal_in)

        elif self.treat_attention_type == "mlp":
            self.mlp1 = nn.Sequential(
                nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, d_temporal_in))
            self.mlp2 = nn.Sequential(
                nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, d_temporal_in))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def mlp_head(self, src_treat, src_mask_treat):
        T = src_treat.shape[0]
        N = src_treat.shape[1]
        time_diffs = (torch.arange(T, device=src_treat.device)[:, None] - torch.arange(
            T, device=src_treat.device)[None, :])[None, ...].repeat(N, 1, 1).float()

        mask = (~src_mask_treat).float()[:N]

        treat_vec1 = src_treat[..., 0].permute(1, 0)[..., None]
        treat_vec2 = src_treat[..., 1].permute(1, 0)[..., None]

        mask1 = mask * treat_vec1
        mask2 = mask * treat_vec2

        effect1 = self.mlp1(time_diffs[..., None])
        effect2 = self.mlp2(time_diffs[..., None])

        additive1 = (effect1 * mask1[..., None]).sum(-2)
        additive2 = (effect2 * mask2[..., None]).sum(-2)

        return (additive1 + additive2).permute(1, 0, 2)

    def forward(self, src, src_treat, src_mask=None, src_mask_treat=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required). # T X N X D
            src_mask: the mask for the src sequence (optional) # N X T X T
            src_treat : separate sequence for the treatment information # T X N X D
            src_mask_treat : separate mask for the treatment information # N X T X T
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        if len(src_mask.shape) == 3:
            src_mask = src_mask.repeat(self.nhead, 1, 1)
            if self.treat_attention_type == "mlp":
                src_mask_treat = src_mask_treat
            else:
                src_mask_treat = src_mask_treat.repeat(self.nhead_treat, 1, 1)

        if self.treat_attention_type == "linear":
            querry = self.querry_mod(src)

            key = self.key_mod(torch.cat((src, src_treat), -1))
            values = self.value_mod(torch.cat((src, src_treat), -1))

            src2 = self.self_attn(
                querry, key, values,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)[0]
        else:
            mapped_src = self.map_embed(src)
            src2 = self.self_attn(
                mapped_src, mapped_src, mapped_src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0]

        # src2 = torch.nan_to_num(src2)
        if self.nhead_treat > 0:
            mapped_src_ = self.map_embed_treat_src(src)
            mapped_src_treat = self.map_embed_treat(src_treat)

            src_treat2 = self.treat_attn(
                mapped_src_, mapped_src_treat, mapped_src_treat,
                attn_mask=src_mask_treat,
                key_padding_mask=src_key_padding_mask
            )[0]
            src2 = torch.cat((src2, src_treat2), -1)
        else:
            if self.treat_attention_type == "mlp":
                src_treat2 = self.mlp_head(src_treat, src_mask_treat)
                src2 = src2 + src_treat2

        src = self.residual1(src, self.dropout1(src2))
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.residual2(src, self.dropout2(src2))
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_long_size,
                 baseline_size,
                 treatment_size,
                 hidden_dim, output_dim, num_layers, t_cond,
                 max_pos_encoding, dim_pos_embed, nheads, dropout_p, ff_dim=512, treat_attention_type=None, **kwargs):
        """
        ---Treatformer Module---
        Args:
            input_long_size (_type_): _description_
            baseline_size (_type_): _description_
            treatment_size (_type_): _description_
            hidden_dim (_type_): size of the embedding for the temporal data. Will be mapped internally.
            out_dim : the dimension of the output embedding.
            num_layers (_type_): _description_
            t_cond (_type_): _description_
            t_horizon (_type_): _description_
            planned_treatments (_type_): _description_
            max_pos_encoding (_type_): _description_
            dim_pos_embed (_type_): _description_
            nheads (_type_): _description_
            ff_dim : the dimension of the hidden layer in the pointwise MLP at the output of the transformer.
            dropout_p (_type_, optional): _description_. Defaults to 0..
            nhead_treat (int, optional): _description_. Defaults to 0.
            treat_attention_type (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()

        self.PositionalEncoding = PositionalEncoding(
            1, max_pos_encoding, dim_pos_embed)

        self.treatment_heads = False

        self.t_cond = t_cond

        self.temporal_input_size = input_long_size

        self.hidden_dim = hidden_dim

        base_hidden_dim = 8
        self.embed_static = nn.Linear(baseline_size, base_hidden_dim)

        if self.treatment_heads:
            d_in = 2*input_long_size + dim_pos_embed + base_hidden_dim
        else:
            d_in = 2*input_long_size + dim_pos_embed + base_hidden_dim + treatment_size

        self.input_mapping = nn.Linear(d_in, hidden_dim)
        if self.treatment_heads:
            self.treat_mapping = nn.Linear(
                treatment_size + dim_pos_embed, hidden_dim)

        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, nheads, ff_dim, dropout_p,
                                    nhead_treat=0, treat_attention_type=treat_attention_type) for n in range(num_layers)])

        self.output_mapping = nn.Linear(hidden_dim, output_dim)

        self.mlp_head = False

    def get_long_input_x(self, X, M, T):
        return torch.cat((X, M, T), -1), None

    def get_src_mask(self, tx, device):
        """
        Computes the attention mask for the vitals and the treatment heads.
        """

        src_mask_x = torch.ones(tx, tx)  # Tx x Tx

        src_mask = ~ (src_mask_x.bool().to(device)).T

        return src_mask

    def forward(self, B, X, M, T, times,):
        """ processes a sequence and return the output sequence

        Args:
            B (N x baseline_size): Static baseline
            X (N x t_cond x input_long_size): Longitudinal data
            M (N x t_cond x input_long_size): Mask for the longitudinal data (1 if observed and 0 if not observed)
            T (N x t_cond x treatment_size): Longitudinal treatment information
            times (N x t_cond x 1): Absolute time values of the observations X, M and T
            times_forward (N x T_horizon x 1): Absolute times at which to evaluate the sequence in the future
            T_forward (N x T_horizon x treatment_size)): Future treatment values
            Y (N x 1): outcome label
            E (N,1): censorship flag

        Returns:
            _type_: _description_ TODO
        """
        X_cond, Treat_cond = self.get_long_input_x(X=X, M=M, T=T)

        # TODO Check if it can take a 3 dim tensor as input
        pos_encodings_cond = self.PositionalEncoding(times)
        pos_encodings_full = pos_encodings_cond
        sequence_full = X_cond

        x_full = torch.cat(
            (pos_encodings_full, sequence_full), -1).float()  # N x T x D

        # Adding the baseline info to the sequence.
        x_full = torch.cat(
            (x_full, self.embed_static(B).unsqueeze(1).repeat(1, x_full.shape[1], 1)), -1)

        x_embedding = self.input_mapping(
            x_full).permute(1, 0, 2)  # T x x N x D

        src_mask = self.get_src_mask(
            tx=self.t_cond, device=X.device)

        import ipdb
        ipdb.set_trace()
        for layer in self.layers:
            x_embedding = layer(x_embedding, None, src_mask=src_mask, src_mask_treat=None,
                                src_key_padding_mask=None)
        x_out = x_embedding.permute(1, 0, 2)  # N x Tx x D

        x_pred = self.output_mapping(x_out)[:, -1, :]
        return x_pred

    @ classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=48)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--nheads', type=int, default=8,
                            help=" total number of heads in the transformer (normal + treat heads)")
        parser.add_argument('--ff_dim', type=int, default=512,
                            help="hidden_units of the feed-forward in the transformer architecture")
        parser.add_argument('--nhead_treat', type=int,
                            default=0, help="number of treatment heads to use")
        parser.add_argument('--max_pos_encoding', type=int, default=100,
                            help="Maximum time (used for computing the continuous positional embeddings)")
        parser.add_argument('--dim_pos_embed', type=int, default=10,
                            help="Dimension of the positional embedding")
        parser.add_argument('--treat_attention_type', type=str,
                            default=None, help="Expert or Linear")
        parser.add_argument('--linear_type', type=str,
                            default="normal", help="normal or all")
        return parser


class FrozenTreatformer(nn.Module):
    def __init__(self,
                 pre_trained_model,
                 emission_type='non_linear',
                 include_baseline=False,
                 include_last=False,
                 emission_window=1,
                 event_type="pfs"):
        super().__init__()
        self.pre_trained_model = pre_trained_model
        self.hidden_dim = self.pre_trained_model.hidden_dim
        self.baseline_dim = self.pre_trained_model.baseline_size
        self.emission_type = emission_type
        self.include_baseline = include_baseline
        self.include_last = include_last
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        input_dim = emission_window * self.hidden_dim
        # first we add baseline and then last observation
        if self.include_baseline:
            input_dim += self.baseline_dim
        if self.include_last:
            input_dim += self.pre_trained_model.temporal_input_size

        self.event_type = event_type

        emission_dim = AE_ALL_DIM

        if self.emission_type == 'non_linear':
            self.emission_mapping_ae = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, emission_dim))
            self.emission_mapping_surv = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        elif self.emission_type == 'linear':
            self.emission_mapping_ae = nn.Linear(input_dim, emission_dim)
            self.emission_mapping_surv = nn.Linear(input_dim, 1)

        self.emission_window = emission_window

    def forward_both_heads(self, h):
        y_ae = self.emission_mapping_ae(h)

        y_surv = self.emission_mapping_surv(h)
        return torch.cat((y_surv, y_ae), -1)

    def forward(self, B, X, M, T, times, times_forward, T_forward, Y, E):

        x_out, _, h_out = self.pre_trained_model.forward(
            B, X, M, T, times, times_forward, T_forward, Y, E, return_hidden=True)

        if self.emission_window == 1:
            input = h_out
            if self.include_baseline:
                input = torch.cat(
                    (input, B[:, None, :].repeat(1, input.shape[1], 1)), -1)
            if self.include_last:
                input = torch.cat(
                    (input, X), -1)

            y_proba = self.forward_both_heads(input)

        else:
            h_out_list = []
            for i_t in range(1, X.shape[1]-self.emission_window+1):
                x = X[:, :i_t, :]
                m = M[:, :i_t, :]
                t = T[:, :i_t, :]
                times_ = times[:, :i_t]
                times_forward_ = times[:, i_t: i_t + self.emission_window]
                t_forward = T[:, i_t: i_t + self.emission_window]
                h_out = self.sample(
                    B, x, m, t, times_, times_forward_, t_forward, Y, E, return_hidden=True)
                h_out_list.append(torch.cat(h_out, -1))
            h_out_vec = torch.stack(h_out_list, 1)
            y_input = h_out_vec
            if self.include_baseline:
                y_input = torch.cat(
                    (h_out_vec, B[:, None, :].repeat(1, h_out_vec.shape[1], 1)), -1)
            if self.include_last:
                y_input = torch.cat(
                    (y_input, X[:, :-self.emission_window, :]), -1)
            y_proba = self.forward_both_heads(y_input)

        return x_out, y_proba

    def sample(self, B, X, M, T, times, times_forward, T_forward, Y, E, return_hidden=False, return_y_hidden=False):
        """Samples forecasts from the model

        Args:
            B (_type_): _description_
            X (_type_): the conditioning data (batch_size, t_cond, input_size)
            M (_type_): the conditioning mask (batch_size, t_cond, input_size)
            T (_type_): the conditioning treatment (batch_size, t_cond, treatment_size)
            times (_type_): the conditioning times (batch_size, t_cond)
            times_forward (_type_): the times for which to give a prediction (length = forecast_window)
            T_forward (_type_): the future treatments (length = forecast_window -1)
            Y (_type_): the survival times
            E (_type_): the censoring label
            return_hidden (bool, optional): Wether to return the sequence of forecasted hiddens. Defaults to False. W
        """
        forecast_window = times_forward.shape[1]
        X_ = X
        M_ = M
        T_ = T
        times_ = times
        h_out_list = []
        for i in range(forecast_window):
            pred, _, h_out = self.pre_trained_model.forward(B, X_, M_, T_, times_,
                                                            times_forward, T_forward, Y, E, return_hidden=True)
            X_ = torch.cat((X_, pred[:, -1, :][:, None, :]), 1)
            M_ = torch.cat(
                (M_, torch.ones(M[:, -1, :].shape, device=M.device)[:, None, :]), 1)
            times_ = torch.cat((times_, times_forward[:, i][:, None]), 1)
            h_out_list.append(h_out[:, -1, :])

            if i != (forecast_window - 1):
                T_ = torch.cat((T_, T_forward[:, i, :][:, None, :]), 1)

        y_input = torch.cat(h_out_list[:self.emission_window], -1)
        if self.include_baseline:
            y_input = torch.cat(
                (y_input, B), -1)
        if self.include_last:
            y_input = torch.cat(
                (y_input, X[:, -1, :]), -1)
        y_pred = self.forward_both_heads(y_input)[:, None, :]

        if return_hidden:
            return h_out_list

        if return_y_hidden:
            return X_, y_pred, None, torch.cat(h_out_list[:self.emission_window], -1)
        else:
            return X_, y_pred, None

    def sample_old(self, B, X, M, T, times, times_forward, T_forward, Y, E, return_hidden=False):

        forecast_window = T_forward.shape[1]
        X_ = X
        M_ = M
        T_ = T
        times_ = times
        y_pred_ = []  # the last prediction iteratively
        y_pred_cond_ = None  # single vector of predictions with t_cond
        h_out_list = []
        for i in range(forecast_window):
            pred, _, h_out = self.pre_trained_model.forward(B, X_, M_, T_, times_,
                                                            times_forward, T_forward, Y, E, return_hidden=True)
            X_ = torch.cat((X_, pred[:, -1, :][:, None, :]), 1)
            M_ = torch.cat(
                (M_, torch.ones(M[:, -1, :].shape, device=M.device)[:, None, :]), 1)
            T_ = torch.cat((T_, T_forward[:, i, :][:, None, :]), 1)
            times_ = torch.cat((times_, times_forward[:, i][:, None]), 1)

            if self.emission_window == 1:
                input = h_out
                if self.include_baseline:
                    input = torch.cat(
                        (h_out, B[:, None, :].repeat(1, h_out.shape[1], 1)), -1)
                if self.include_last:
                    input = torch.cat(
                        (input, X_[:, :-1, :]), -1)
                y_pred = self.forward_both_heads(input)
                if i == 0:
                    y_pred_cond_ = y_pred
                y_pred_.append(y_pred[:, -1, :][:, None, :])

        if return_hidden:
            return h_out[:, -forecast_window:]

        if self.emission_window > 1:
            X_ = X_[:, :-self.emission_window+1, :]
            h_out_future = h_out[:, -forecast_window:, :]
            for i in range(forecast_window-(self.emission_window-1)):
                h_ = h_out_future[:, i:i+self.emission_window,
                                  :].reshape((h_out.shape[0], -1))
                input = h_
                if self.include_baseline:
                    input = torch.cat((h_, B), -1)
                if self.include_last:
                    input = torch.cat((input,
                                       X_[:, i:i+self.emission_window, :].reshape((X_.shape[0], -1))), -1)
                y_pred = self.forward_both_heads(input)
                if i == 0:
                    # ypred cond may have a different shape in the case emission_window > 1
                    y_pred_cond_ = y_pred
                y_pred_.append(y_pred[:, None, :])
            y_pred_ = torch.cat(y_pred_, 1)
        else:
            y_pred_ = torch.cat(y_pred_, 1)

        return X_, y_pred_, y_pred_cond_


class Treatformer(nn.Module):
    def __init__(self,
                 input_long_size,
                 baseline_size,
                 treatment_size,
                 prediction_idx,
                 input_idx,
                 prediction_names,
                 input_names,
                 hidden_dim, num_layers, t_cond, t_horizon,
                 planned_treatments, max_pos_encoding, dim_pos_embed, nheads, dropout_p,  nhead_treat, baseline_hidden_dim=16, ff_dim=512, treat_attention_type=None, emission_proba=False, **kwargs):
        """
        ---Treatformer Module---
        Args:
            temporal_input_size (_type_): _description_
            baseline_size (_type_): _description_
            treatment_size (_type_): _description_
            hidden_dim (_type_): size of the embedding for the temporal data. Will be mapped internally.
            num_layers (_type_): _description_
            t_cond (_type_): _description_
            t_horizon (_type_): _description_
            planned_treatments (_type_): _description_
            max_pos_encoding (_type_): _description_
            dim_pos_embed (_type_): _description_
            nheads (_type_): _description_
            ff_dim : the dimension of the hidden layer in the pointwise MLP at the output of the transformer.
            dropout_p (_type_, optional): _description_. Defaults to 0..
            nhead_treat (int, optional): _description_. Defaults to 0.
            treat_attention_type (_type_, optional): _description_. Defaults to None.
            emission_probabilities : if True, the model also outputs probabilities of event (bernoulli process)
        """
        super().__init__()

        self.PositionalEncoding = PositionalEncoding(
            1, max_pos_encoding, dim_pos_embed)

        if nhead_treat > 0:
            self.treatment_heads = True
        else:
            # no real treatment heads but use treatment information
            if (treat_attention_type == "linear") or (treat_attention_type == "mlp"):
                self.treatment_heads = True
            else:
                self.treatment_heads = False

        self.t_cond = t_cond

        self.t_horizon = t_horizon

        self.planned_treatments = planned_treatments
        self.temporal_input_size = len(input_idx)  # input_long_size

        self.prediction_idx = prediction_idx
        self.prediction_names = prediction_names
        self.prediction_size = len(prediction_idx)

        self.hidden_dim = hidden_dim
        self.baseline_size = baseline_size
        # if self.treatment_heads:
        #    input_long_size = temporal_input_size - treatment_size
        # self.input_long_size = temporal_input_size

        # base_hidden_dim = baseline_hidden_dim
        # self.embed_static = nn.Linear(baseline_size, base_hidden_dim)

        if self.treatment_heads:
            d_in = 2*self.temporal_input_size + dim_pos_embed + baseline_size
        else:
            d_in = 2*self.temporal_input_size + \
                dim_pos_embed + baseline_size + treatment_size

        self.input_mapping = nn.Linear(d_in, hidden_dim)
        if self.treatment_heads:
            self.treat_mapping = nn.Linear(
                treatment_size + dim_pos_embed, hidden_dim)

        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, nheads-nhead_treat, ff_dim, dropout_p,
                                                             nhead_treat=nhead_treat, treat_attention_type=treat_attention_type) for n in range(num_layers)])

        self.output_mapping = nn.Sequential(nn.Linear(
            hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.prediction_size))

        self.emission_proba = emission_proba
        if self.emission_proba:
            self.emission_mapping = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1 + AE_ALL_DIM))

        if treat_attention_type == "mlp":
            self.mlp_head = True
        else:
            self.mlp_head = False

    def get_long_input_x(self, X, M, T):

        if self.treatment_heads:
            return torch.cat((X, M), -1), T
        else:
            return torch.cat((X, M, T), -1), None

    def get_long_input_y(self, times_forward, T_forward):

        if self.t_cond == - 1:
            return None, None

        N = times_forward.shape[0]
        if self.treatment_heads:
            long_input_y = torch.cat((torch.zeros((N, self.t_horizon, self.temporal_input_size), device=times_forward.device), torch.zeros(
                (N, self.t_horizon, self.temporal_input_size), device=times_forward.device)), -1), T_forward
        else:
            long_input_y = torch.cat((torch.zeros((N, self.t_horizon, self.temporal_input_size), device=times_forward.device), torch.zeros(
                (N, self.t_horizon, self.temporal_input_size), device=times_forward.device), T_forward), -1), None
        return long_input_y

    def get_src_mask(self, tx, ty, device):
        """
        Computes the attention mask for the vitals and the treatment heads.
        """

        if self.t_cond == -1:
            src_mask = torch.triu(torch.ones(tx, tx))
            src_mask = ~(src_mask.bool().to(device)).T
            return src_mask, src_mask

        src_mask_x = torch.triu(torch.ones(tx, tx))  # Tx x Tx
        src_mask_x = torch.cat(
            (src_mask_x, torch.zeros(ty, tx)), 0)  # (Tx + Ty) x Tx

        src_mask_y = torch.ones(tx+ty, ty)
        src_mask = torch.cat((src_mask_x, src_mask_y), 1)
        src_mask = ~ (src_mask.bool().to(device)).T

        src_mask_treat = torch.triu(torch.ones(tx+ty, tx+ty))
        src_mask_treat = ~(src_mask_treat.bool().to(device)).T

        key_padding_mask = None

        return src_mask, src_mask_treat

    def sample(self, B, X, M, T, times, times_forward, T_forward, Y, E, return_y_hidden=False):

        forecast_window = T_forward.shape[1]
        X_ = X
        M_ = M
        T_ = T
        times_ = times
        y_pred_ = []  # the last prediction iteratively
        y_pred_cond_ = None  # single vector of predictions with t_cond
        for i in range(forecast_window):
            pred, y_pred = self.forward(B, X_, M_, T_, times_,
                                        times_forward, T_forward, Y, E)
            X_ = torch.cat((X_, pred[:, -1, :][:, None, :]), 1)
            M_ = torch.cat(
                (M_, torch.ones(M[:, -1, :].shape, device=M.device)[:, None, :]), 1)
            T_ = torch.cat((T_, T_forward[:, i, :][:, None, :]), 1)
            times_ = torch.cat((times_, times_forward[:, i][:, None]), 1)

            if y_pred is not None:
                if i == 0:
                    y_pred_cond_ = y_pred
                y_pred_.append(y_pred[:, -1, :][:, None, :])

        if y_pred is not None:
            y_pred_ = torch.cat(y_pred_, 1)
        else:
            y_pred_ = None

        if return_y_hidden:
            return X_, y_pred_, y_pred_cond_, None  # no y_hidden here
        else:
            return X_, y_pred_, y_pred_cond_

    def forward(self, B, X, M, T, times, times_forward, T_forward, Y, E, return_hidden=False):
        """ processes a sequence and return the output sequence

        Args:
            B (N x baseline_size): Static baseline
            X (N x t_cond x input_long_size): Longitudinal data
            M (N x t_cond x input_long_size): Mask for the longitudinal data (1 if observed and 0 if not observed)
            T (N x t_cond x treatment_size): Longitudinal treatment information
            times (N x t_cond x 1): Absolute time values of the observations X, M and T
            times_forward (N x T_horizon x 1): Absolute times at which to evaluate the sequence in the future
            T_forward (N x T_horizon x treatment_size)): Future treatment values
            Y (N x 1): outcome label
            E (N,1): censorship flag

        Returns:
            _type_: _description_ TODO
        """
        X_cond, Treat_cond = self.get_long_input_x(X=X, M=M, T=T)

        X_forward, Treat_forward = self.get_long_input_y(
            times_forward=times_forward, T_forward=T_forward)

        # TODO Check if it can take a 3 dim tensor as input
        pos_encodings_cond = self.PositionalEncoding(times)
        pos_encodings_forward = self.PositionalEncoding(
            times_forward)

        if self.t_cond == -1:
            pos_encodings_full = pos_encodings_cond
            sequence_full = X_cond
        else:
            pos_encodings_full = torch.cat(
                (pos_encodings_cond, pos_encodings_forward), 1)

            sequence_full = torch.cat((X_cond, X_forward), 1)

        if Treat_cond is None:  # classical head
            treatment_sequence = None
            treat_full = None
        else:
            if self.t_cond == -1:
                treatment_sequence = Treat_cond
            else:
                treatment_sequence = torch.cat((Treat_cond, Treat_forward), 1)
            if self.mlp_head:
                treat_full = treatment_sequence
            else:
                treat_full = torch.cat(
                    (pos_encodings_full, treatment_sequence), -1)

        x_full = torch.cat(
            (pos_encodings_full, sequence_full), -1).float()  # N x T x D

        # Adding the baseline info to the sequence.
        x_full = torch.cat(
            (x_full, B.unsqueeze(1).repeat(1, x_full.shape[1], 1)), -1)

        x_embedding = self.input_mapping(
            x_full).permute(1, 0, 2)  # T x x N x D

        if treat_full is not None:  # treatment head
            if self.mlp_head:
                x_treat = treat_full.permute(1, 0, 2)  # Tx x N x D
            else:
                x_treat = self.treat_mapping(
                    treat_full).permute(1, 0, 2)  # Tx x N x D
        else:  # classical head
            x_treat = None

        tx = X_cond.shape[1]
        src_mask, src_mask_treat = self.get_src_mask(
            tx=tx, ty=self.t_horizon, device=X.device)

        for layer in self.layers:
            x_embedding = layer(x_embedding, x_treat, src_mask=src_mask, src_mask_treat=src_mask_treat,
                                src_key_padding_mask=None)
        x_out = x_embedding.permute(1, 0, 2)  # N x Tx x D

        x_pred = self.output_mapping(x_out)

        if self.emission_proba:
            y_proba = self.emission_mapping(x_out)
        else:
            y_proba = None

        if return_hidden:
            return x_pred, y_proba, x_out
        else:
            return x_pred, y_proba

    @ classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--nheads', type=int, default=8,
                            help=" total number of heads in the transformer (normal + treat heads)")
        parser.add_argument('--ff_dim', type=int, default=128,
                            help="hidden_units of the feed-forward in the transformer architecture")
        parser.add_argument('--nhead_treat', type=int,
                            default=0, help="number of treatment heads to use")
        parser.add_argument('--max_pos_encoding', type=int, default=100,
                            help="Maximum time (used for computing the continuous positional embeddings)")
        parser.add_argument('--dim_pos_embed', type=int, default=10,
                            help="Dimension of the positional embedding")
        parser.add_argument('--treat_attention_type', type=str,
                            default=None, help="Expert or Linear")
        parser.add_argument('--linear_type', type=str,
                            default="normal", help="normal or all")
        parser.add_argument('--baseline_hidden_dim', type=int,
                            default=16, help="dimension of the baseline embedding")
        return parser
