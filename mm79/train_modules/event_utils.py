import numpy as np
import torch
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import pandas as pd
from mm79 import EVENT_SHIFT


def cumulative_dyn_auc(et_train, et_test, risk, quantile):
    mask = et_test["t"] <= et_train["t"].max()
    try:
        roc_auc = cumulative_dynamic_auc(
            et_train, et_test[mask], risk[mask], quantile)
    except:
        print("Error in the evaluation of the event AUC")
        roc_auc = [[np.nan]]
    return roc_auc


def kp_weighted(event_durations, event_observed, groups, weights):

    #weights[groups == 0] = 1-weights[groups == 0]
    unique_times = torch.unique(event_durations)
    Nij = torch.zeros((len(event_durations), len(unique_times)),
                      device=event_durations.device)
    Oij = torch.zeros((len(event_durations), len(unique_times)),
                      device=event_durations.device)
    for i in range(len(event_durations)):
        Nij[i, torch.where(unique_times <= event_durations[i])[
            0]] = 1 * weights[i, groups[i].long()]
        Oij[i, torch.where(unique_times == event_durations[i])[0]
            ] = event_observed[i] * weights[i, groups[i].long()]

    Nj = Nij.sum(0)
    N0j = Nij[groups == 0].sum(0)
    N1j = Nij[groups == 1].sum(0)

    Oj = Oij.sum(0)
    O0j = Oij[groups == 0].sum(0)
    O1j = Oij[groups == 1].sum(0)

    S0 = torch.cumprod(1 - O0j / N0j, 0)
    S1 = torch.cumprod(1 - O1j / N1j, 0)
    S = torch.cumprod(1 - Oj / Nj, 0)

    return (-S).mean()
    # return -(S1 + S0).mean()


def torch_logrank(event_durations, event_observed, groups, weights, true_variance=False):
    """
    Compute logrank test statistics
    event_durations : time of events
    event_observed : indicator of event (1 if event is observed)
    groups : group indicator (0 or 1) - only supports 2 groups now.
    weights : weights for each event
    """
    assert len(torch.unique(groups)) == 2

    #weights_ = weights[np.arange(len(weights)), groups]
    weights_ = weights[:, 1]

    unique_times = torch.unique(event_durations)
    Nij_w = torch.zeros((len(event_durations), len(unique_times)),
                        device=event_durations.device)
    Oij_w = torch.zeros((len(event_durations), len(unique_times)),
                        device=event_durations.device)

    Nij_ww = torch.zeros((len(event_durations), len(unique_times)),
                         device=event_durations.device)
    Oij_ww = torch.zeros((len(event_durations), len(unique_times)),
                         device=event_durations.device)

    Nij = torch.zeros((len(event_durations), len(unique_times)),
                      device=event_durations.device)
    Oij = torch.zeros((len(event_durations), len(unique_times)),
                      device=event_durations.device)

    for i in range(len(event_durations)):
        Nij_w[i, torch.where(unique_times <= event_durations[i])[
            0]] = 1 * weights_[i]
        Oij_w[i, torch.where(unique_times == event_durations[i])[0]
              ] = event_observed[i] * weights_[i]

        Nij_ww[i, torch.where(unique_times <= event_durations[i])[
            0]] = 1 * weights_[i] * weights_[i]
        Oij_ww[i, torch.where(unique_times == event_durations[i])[0]
               ] = event_observed[i] * weights_[i] * weights_[i]

        Nij[i, torch.where(unique_times <= event_durations[i])[
            0]] = 1
        Oij[i, torch.where(unique_times == event_durations[i])[0]
            ] = event_observed[i]

    # if len(torch.unique(weights)) == 2:
    #    import ipdb
    #    ipdb.set_trace()
    N0j_w = Nij_w[groups == 0].sum(0)
    N1j_w = Nij_w[groups == 1].sum(0)

    N0j_ww = Nij_ww[groups == 0].sum(0)
    N1j_ww = Nij_ww[groups == 1].sum(0)

    Nj_w = Nij_w.sum(0)
    Nj = Nij.sum(0)
    Nj_ww = Nij_ww.sum(0)

    O0j_w = Oij_w[groups == 0].sum(0)
    O1j_w = Oij_w[groups == 1].sum(0)
    Oj_w = Oij_w.sum(0)
    Oj = Oij.sum(0)

    E0j = N0j_w * Oj_w / Nj_w
    E1j = N1j_w * Oj_w / Nj_w

    E1j[(torch.isnan(E1j) | torch.isinf(E1j))] = 0
    E0j[(torch.isnan(E0j) | torch.isinf(E0j))] = 0

    Z0 = (O0j_w - E0j).sum()
    Z1 = (O1j_w - E1j).sum()

    if true_variance:
        ratio = (Oj*(Nj - Oj)) / ((Nj - 1)*Nj)
        ratio[(torch.isnan(ratio) | torch.isinf(ratio))] = 0

        Var = (ratio * ((N0j_w * N0j_w) * N1j_ww +
                        (N1j_w * N1j_w) * N0j_ww) / (Nj_w * Nj_w)).sum()
    else:
        Var_ = ((N1j_w / Nj_w) * (1-N1j_w / Nj_w) *
                ((Nj_w - Oj_w) / (Nj_w - 1)) * Oj_w)
        Var_[(torch.isnan(Var_) | torch.isinf(Var_))] = 0
        Var = Var_.sum()

    #U0 = torch.sqrt(Z0**2)
    U0 = torch.sqrt(Z1**2 / Var)
    if not torch.isfinite(U0).all():
        import ipdb
        ipdb.set_trace()

    return U0


def vectorized_logrank(event_durations, event_observed, groups, weights):
    """
    Compute logrank test statistics
    event_durations : time of events
    event_observed : indicator of event (1 if event is observed)
    groups : group indicator (0 or 1) - only supports 2 groups now.
    weights : weights for each event
    """

    assert len(np.unique(groups)) == 2

    unique_times = np.unique(event_durations)
    Nij = np.zeros((len(event_durations), len(unique_times)))
    Oij = np.zeros((len(event_durations), len(unique_times)))
    for i in range(len(event_durations)):
        Nij[i, np.where(unique_times <= event_durations[i])] = 1 * weights[i]
        Oij[i, np.where(unique_times == event_durations[i])
            ] = event_observed[i] * weights[i]

    N0j = Nij[groups == 0].sum(0)
    N1j = Nij[groups == 1].sum(0)
    Nj = Nij.sum(0)
    O0j = Oij[groups == 0].sum(0)
    O1j = Oij[groups == 1].sum(0)
    Oj = Oij.sum(0)

    E0j = N0j * Oj / Nj
    E1j = N1j * Oj / Nj

    Z0 = (O0j - E0j).sum()
    Z1 = (O1j - E1j).sum()

    V0j = E0j * (Nj - Oj) * (Nj - N0j) / (Nj * (Nj - 1))
    V1j = E1j * (Nj - Oj) * (Nj - N1j) / (Nj * (Nj - 1))

    U0 = Z0**2 / V0j.sum()
    return U0


def time_censoring_proc(E, Y):
    et_hp = np.array([(E[i].numpy(), Y[i].numpy()) for i in
                      range(len(Y))], dtype=[('e', bool), ('t', float)])

    return et_hp


def time_censoring_processing(data, event_type):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, Eae, Yae, times_cond, times_future, Treat_flag, _ = data

    Y, E = select_event_type(Y, E, Yae, Eae, event_type=event_type)

    et_hp = np.array([(E[i].numpy(), Y[i].numpy()) for i in
                      range(len(Y))], dtype=[('e', bool), ('t', float)])

    return et_hp


def extract_quantiles(dataset, t_cond, event_type):
    """Extracting time quantiles from the dataset

    Args:
        dataset (_type_): the dataset (pl module)
        t_cond (_type_): the conditioning window

    Returns:
        _type_: Extract the 25, 50 and 75% quantiles from the dataset using the training data
    """

    train_data = extract_dataset(
        dataset.train_dataloader(shuffle=False), t_cond=t_cond, use_traj=False, shift_baseline=True, event_type=event_type)
    pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, Eae, Yae, times_cond, times_future, Treat_flag, _ = train_data

    et_train = time_censoring_processing(train_data, event_type)
    times = []
    quantiles = [0.25, 0.5, 0.75]
    for quant in quantiles:
        if event_type == "pfs":
            times.append(np.quantile(Y.squeeze(), quant))
        else:
            times.append(np.quantile(Yae[:, int(event_type)], quant))
    return times, et_train


def get_idxs_from_F(pids, subgroup):
    F = pd.read_csv('/dbfs/FileStore/takeda-mm/data/F_df')
    subgroup_idxs = []
    _, _, val_list = subgroup
    for idx, pid in enumerate(pids):
        v = F[F["USUBJID"].apply(lambda x: "".join(
            x.split("-")[1:])).astype(float) == pid.numpy()]['AVAL'].values[0]
        if v in val_list:
            subgroup_idxs.append(idx)
    return subgroup_idxs


def extract_dataset(dl, t_cond, use_traj, shift_baseline, subgroup=None, subgroup_strat="all", datamodule=None, randomize_idx=None, pretrained_model=None, event_type="pfs", iss_only=False):
    if isinstance(t_cond, list):
        return [extract_dataset_(dl=dl,
                                 t_cond=t_cond_,
                                 use_traj=use_traj,
                                 shift_baseline=shift_baseline,
                                 subgroup=subgroup,
                                 subgroup_strat=subgroup_strat,
                                 datamodule=datamodule,
                                 randomize_idx=randomize_idx,
                                 pretrained_model=pretrained_model,
                                 event_type=event_type,
                                 iss_only=iss_only) for t_cond_ in t_cond]
    else:
        return extract_dataset_(dl=dl,
                                t_cond=t_cond,
                                use_traj=use_traj,
                                shift_baseline=shift_baseline,
                                subgroup=subgroup,
                                subgroup_strat=subgroup_strat,
                                datamodule=datamodule,
                                randomize_idx=randomize_idx,
                                pretrained_model=pretrained_model,
                                event_type=event_type,
                                iss_only=iss_only)


def select_event_type(Y, E, Yae, Eae, event_type="pfs"):
    if event_type == "pfs":
        Y = Y
        E = E
    else:
        Y = Yae[:, int(event_type)][:, None]
        E = Eae[:, int(event_type)][:, None]
    return Y, E


def extract_dataset_(dl, t_cond, use_traj, shift_baseline, subgroup=None, subgroup_strat='all', datamodule=None, randomize_idx=None, pretrained_model=None, event_type="pfs", iss_only=False):
    '''
        Args: 
            subgroup: expects a tuple (feature index, comparator [e.g. ==, <, >], value)

        Return: 
            TODO
    '''
    batches = []
    for i, batch in enumerate(dl):
        batches.append(batch)
    concats = []
    for i in range(len(batches[0])):
        concats.append(torch.cat([batch[i] for batch in batches]))

    pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, \
        T_future, E, Y, Eae, Yae, times_cond, times_future, Treat_flag = concats

    if randomize_idx is not None:
        X_cond[:, :, randomize_idx] = X_cond[torch.randperm(
            X_cond.shape[0]), :, randomize_idx]

    if t_cond >= 0:
        if ((use_traj) or (shift_baseline)):
            if event_type == "pfs":
                retained_idx = (Y > (t_cond - EVENT_SHIFT))[:, 0]
            else:
                retained_idx = (Yae[:, int(event_type)]
                                > (t_cond - EVENT_SHIFT))

            pids = pids[retained_idx]
            B = B[retained_idx]
            R = R[retained_idx]
            R_mask = R_mask[retained_idx]
            X_cond = X_cond[retained_idx, :t_cond]
            X_future = X_future[retained_idx]
            M_cond = M_cond[retained_idx, :t_cond]
            M_future = M_future[retained_idx]
            T_cond = T_cond[retained_idx, :t_cond]
            T_future = T_future[retained_idx]
            E = E[retained_idx]
            Y = Y[retained_idx] - (t_cond-EVENT_SHIFT)
            Eae = Eae[retained_idx]
            Yae = Yae[retained_idx] - (t_cond-EVENT_SHIFT)
            times_cond = times_cond[retained_idx, :t_cond]
            times_future = times_future[retained_idx]
            Treat_flag = Treat_flag[retained_idx]

    if subgroup is not None:
        # use datamodule here
        B_orig = datamodule.unnormalize_B(B)
        idx, comp, val = subgroup
        if subgroup_strat == 'treatment-response':
            subgroup_idxs = get_idxs_from_F(pids, subgroup)
        elif comp == '==':
            subgroup_idxs = np.where(np.abs(B_orig[:, idx]-val) < 1e-2)
        elif comp == '<':
            subgroup_idxs = np.where(B_orig[:, idx] < val)
        elif comp == '>':
            subgroup_idxs = np.where(B_orig[:, idx] > val)
        elif comp == '?':
            subgroup_idxs = np.arange(len(B_orig))
        pids = pids[subgroup_idxs]
        B = B[subgroup_idxs]
        R = R[subgroup_idxs]
        R_mask = R_mask[subgroup_idxs]
        X_cond = X_cond[subgroup_idxs]
        X_future = X_future[subgroup_idxs]
        M_cond = M_cond[subgroup_idxs]
        M_future = M_future[subgroup_idxs]
        T_cond = T_cond[subgroup_idxs]
        T_future = T_future[subgroup_idxs]
        E = E[subgroup_idxs]
        Y = Y[subgroup_idxs]
        Eae = Eae[subgroup_idxs]
        Yae = Yae[subgroup_idxs]
        times_cond = times_cond[subgroup_idxs]
        times_future = times_future[subgroup_idxs]
        Treat_flag = Treat_flag[subgroup_idxs]

    h_out_final = None
    if pretrained_model is not None:
        x_out, _, h_out = pretrained_model.model(
            B, X_cond, M_cond, T_cond, times_cond, times_future, T_future, Y, E, return_hidden=True)

        h_out_final = h_out[:, -1, :].detach().numpy()

    if iss_only:
        iss_idx = np.where(datamodule.feats['B_feat_names'] == 'ISSENT')[0][0]
        trt_idx = np.where(datamodule.feats['B_feat_names'] == 'TRT01AN')[0][0]
        B = B[:, [iss_idx, trt_idx]]

    return pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, Eae, Yae, times_cond, times_future, Treat_flag, h_out_final
