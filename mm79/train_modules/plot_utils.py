import matplotlib.pyplot as plt
import os
from mm79.train_modules.train_module import TrainModule
from mm79.train_modules.utils import model_type_class, data_type_class, get_predictions, get_var_labels
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from mm79 import EXPERIMENT_DIR, TYPE_VAR_DICT
import pickle


def plot_wandb_predictions(run=None, run_name=None, pat_idx=42, t_cond=None, t_horizon=None):
    """
    plotting utils for the predicted trajectories
    pat_idx is the index of the patient in the test to plot.
    """
    if run_name is not None:
        api = wandb.Api()
        run = api.run(f"edebrouwer/mm-long/{run_name}")

    pred_dict = evaluate_wandb_run(run, t_cond=t_cond, t_horizon=t_horizon)
    prediction_idx = pred_dict["prediction_idx"]
    times_cond = pred_dict["times_cond"]
    M_cond = pred_dict["M_cond"]
    X_cond = pred_dict["X_cond"]
    times_future = pred_dict["times_future"]
    forecast_future = pred_dict["forecast_future"]
    target_future = pred_dict["target_future"]
    mask_future = pred_dict["mask_future"]
    prediction_names = pred_dict["prediction_names"]

    # plot
    pat_idx = pat_idx
    f, ax = plt.subplots(figsize=(14, 10))
    for i, pred_idx in enumerate(prediction_idx):
        p0 = ax.plot(times_cond[pat_idx][M_cond[pat_idx, :, pred_idx] == 1],
                     X_cond[pat_idx, M_cond[pat_idx, :, pred_idx] == 1, pred_idx].detach().numpy())
        ax.scatter(times_cond[pat_idx][M_cond[pat_idx, :, pred_idx] == 1],
                   X_cond[pat_idx, M_cond[pat_idx, :, pred_idx]
                          == 1, pred_idx].detach().numpy(),
                   label=f"Cond dim-{prediction_names[i]}",
                   color=p0[0].get_color())
        ax.plot(times_future[pat_idx],
                forecast_future[pat_idx, :, i].detach().numpy(),
                label=f"Preds dim-{prediction_names[i]}",
                color=p0[0].get_color(),
                linestyle="--")
        ax.scatter(times_future[pat_idx][mask_future[pat_idx, :, pred_idx] == 1],
                   target_future[pat_idx, mask_future[pat_idx,
                                                      :, pred_idx] == 1, pred_idx],
                   label=f"Obs dim-{prediction_names[i]}",
                   color=p0[0].get_color(),
                   marker="*")
    ax.legend()
    ax.set_title("Predictions")
    ax.set_xlabel("Time")
    return ax, pred_dict


def plot_predictions(experiment_name,
                     version_name=None,
                     pat_idx=2,
                     t_cond=None,
                     t_horizon=None):
    """_summary_

    Args:
        experiment_name (st): 
            Name of the experiment folder
        version_name (str, optional): 
            Version to use for the plot. 
            Should be "version_XX" where XX is the version number. 
            Defaults to None. If None, the latest version will be used.
    """
    experiment_dir = os.path.join(EXPERIMENT_DIR, "logs", experiment_name)
    max_version = 0

    if version_name is None:
        for version in os.listdir(experiment_dir):
            version_num = int(version.split("_")[-1])
            if version_num >= max_version:
                version_max_name = version
                max_version = version_num
    else:
        version_max_name = version_name
    experiment_dir = os.path.join(experiment_dir, version_max_name)

    pred_dict = get_predictions(
        experiment_dir=experiment_dir,
        t_cond=t_cond,
        t_horizon=t_horizon,
        test_fold=True
    )

    prediction_idx = pred_dict["prediction_idx"]
    prediction_names = pred_dict["prediction_names"]
    times_cond = pred_dict["times_cond"]
    times_future = pred_dict["times_future"]
    M_cond = pred_dict["M_cond"]
    X_cond = pred_dict["X_cond"]
    mask_future = pred_dict["mask_future"]
    target_future = pred_dict["target_future"]
    forecast_future = pred_dict["forecast_future"]

    # plot
    #pat_idx = 2
    f, ax = plt.subplots(figsize=(14, 10))
    for i, pred_idx in enumerate(prediction_idx):
        p0 = ax.plot(times_cond[pat_idx][M_cond[pat_idx, :, pred_idx] == 1],
                     X_cond[pat_idx, M_cond[pat_idx, :, pred_idx] == 1, pred_idx].detach().numpy())
        ax.scatter(times_cond[pat_idx][M_cond[pat_idx, :, pred_idx] == 1],
                   X_cond[pat_idx, M_cond[pat_idx, :, pred_idx]
                          == 1, pred_idx].detach().numpy(),
                   label=f"Cond dim-{prediction_names[i]}",
                   color=p0[0].get_color())
        ax.plot(times_future[pat_idx],
                forecast_future[pat_idx, :, i].detach().numpy(),
                label=f"Preds dim-{prediction_names[i]}",
                color=p0[0].get_color(),
                linestyle="--")
        ax.scatter(times_future[pat_idx][mask_future[pat_idx, :, pred_idx] == 1],
                   target_future[pat_idx, mask_future[pat_idx,
                                                      :, pred_idx] == 1, pred_idx],
                   label=f"Obs dim-{prediction_names[i]}",
                   color=p0[0].get_color(),
                   marker="*")
    ax.legend()
    ax.set_title("Predictions")
    ax.set_xlabel("Time")
    plt.savefig(os.path.join(experiment_dir, "predictions_plot.png"))
    return ax, pred_dict


def plot_metrics(experiment_name, version_name=None):
    experiment_dir = os.path.join(EXPERIMENT_DIR, "logs", experiment_name)
    max_version = 0

    if version_name is None:
        for version in os.listdir(experiment_dir):
            version_num = int(version.split("_")[-1])
            if version_num >= max_version:
                version_max_name = version
                max_version = version_num
    else:
        version_max_name = version_name
    experiment_dir = os.path.join(experiment_dir, version_max_name)

    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    metrics = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(metrics.loc[~metrics.val_loss.isna()].step.values,
            metrics.loc[~metrics.val_loss.isna()].val_loss.values,
            label="val loss")
    ax.plot(metrics.loc[~metrics.train_loss_epoch.isna()].step.values,
            metrics.loc[~metrics.train_loss_epoch.isna()
                        ].train_loss_epoch.values,
            label="training loss")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss at different steps")
    return ax


def plot_mm2_trajectories(split='train', fold=0):
    data_cls = data_type_class("MM2")
    dl = data_cls(process_data=False, t_cond=82, t_horizon=0, fold=fold)
    dl.prepare_data()
    batches = []
    if split == 'train':
        data_loader = dl.train_dataloader
    elif split == 'val':
        data_loader = dl.val_dataloader
    elif split == 'test':
        data_loader = dl.test_dataloader
    for i, batch in enumerate(data_loader()):
        batches.append(batch)
    concats = []
    for i in range(len(batches[0])):
        concats.append(torch.cat([batch[i] for batch in batches]))

    return dl, concats, dl.prediction_idx, dl.prediction_names


def get_mse(exp_path, subgroup, t_cond, t_horizon, var_type, split='val', preds=None, dataset_name="MM2"):
    """_summary_

    Args:
        exp_path (str): path to the experiment folder
        subgroup (str): subgroup for which to compute the mse
        t_cond (int): observation window
        t_horizon (int): length of prediction window
        var_type (str): variables to use for evaluation
        split (str): Whether to compute the val or test. Defaults to 'val'.
        preds predictions for each subgroup. Defaults to None. If not provided, will be computed from the exp_path. 
        If provided, overrides the exp_path, t_cond and t_horizon and split.

    Returns:
        mse, var_names: mse for this subgroup and var strategy + names of variables used in the evaluation
    """
    type_var_dict = TYPE_VAR_DICT

    if split == 'test':
        test_fold = True
    else:
        test_fold = False

    if preds is None:

        # check if the predictions have been pre-computed
        if os.path.exists(os.path.join(exp_path, f'test_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl')):
            print("Loading pre-computed predictions")
            with open(os.path.join(exp_path, f"{split}_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl"), 'rb') as f:
                pred_dict_forecast = pickle.load(f)

        else:
            if dataset_name == 'MM2':
                MM1 = False
            else:
                MM1 = True
            pred_dict_forecast = get_predictions(exp_path,
                                                 t_cond=t_cond,
                                                 t_horizon=t_horizon,
                                                 test_fold=test_fold,
                                                 subgroup_strat='myeloma-type',
                                                 MM1=MM1)[subgroup]
    else:
        pred_dict_forecast = preds[subgroup]

    future_variance = np.var(
        pred_dict_forecast["target_future"].numpy(), axis=1).mean(1)
    var_labels = get_var_labels(future_variance)

    feat_names = pred_dict_forecast['prediction_names']
    if var_type == 'var-all':
        var_names = feat_names
    elif var_type == 'var-serum' or var_type == 'var-chem':
        var_names = type_var_dict[var_type]

    var_idxs = np.array([list(feat_names).index(x) for x in var_names])
    forecast_ = pred_dict_forecast["forecast_future"][..., var_idxs]
    target_ = pred_dict_forecast["target_future"][..., var_idxs]
    mask_ = pred_dict_forecast["mask_future"][..., var_idxs]
    assert len(var_idxs) == forecast_.shape[-1]
    total_mse = ((forecast_-target_)*mask_).pow(2).sum() / mask_.sum()

    var01_mse = ((forecast_[var_labels == 0]-target_[var_labels == 0]) *
                 mask_[var_labels == 0]).pow(2).sum() / mask_[var_labels == 0].sum()
    var0102_mse = ((forecast_[var_labels == 1]-target_[var_labels == 1]) *
                   mask_[var_labels == 1]).pow(2).sum() / mask_[var_labels == 1].sum()
    var0203_mse = ((forecast_[var_labels == 2]-target_[var_labels == 2]) *
                   mask_[var_labels == 2]).pow(2).sum() / mask_[var_labels == 2].sum()
    var03_mse = ((forecast_[var_labels == 3]-target_[var_labels == 3]) *
                 mask_[var_labels == 3]).pow(2).sum() / mask_[var_labels == 3].sum()

    mse = {
        'total': total_mse,
        'var01': var01_mse,
        'var0102': var0102_mse,
        'var0203': var0203_mse,
        'var03': var03_mse
    }

    return mse, var_names


def get_forecasting_results_df(sweep_name,
                               best_runs,
                               folds=[0],
                               subgroups=['all'],
                               t_cond_horizon_pairs=[(6, 12)],
                               var_types=['var-all'],
                               save=False,
                               dataset_name="MM2"):

    best_runs_dict = {x[0]: {'version_name': x[1], 'pred_dict_forecast': {}}
                      for x in list(zip(best_runs["fold"].values, best_runs["run_name"]))}
    res_list = []

    i = 0
    for subgroup in subgroups:
        for fold in folds:
            version = best_runs_dict[fold]["version_name"]
            exp_path = os.path.join(
                EXPERIMENT_DIR, "logs", sweep_name, version)
            for t_cond, t_horizon in t_cond_horizon_pairs:
                for var_type in var_types:
                    print(
                        f'[compiling results for fold {fold}, subgroup {subgroup}, t_cond {t_cond}, t_horizon {t_horizon}, var_type {var_type}]')

                    test_mse, var_names = get_mse(
                        exp_path, subgroup, t_cond, t_horizon, var_type, split='test', dataset_name=dataset_name)
                    val_mse, _ = get_mse(
                        exp_path, subgroup, t_cond, t_horizon, var_type, split='val', dataset_name=dataset_name)
                    var_names = ','.join([x for x in var_names])

                    res = {'subgroup': subgroup,
                           'fold': fold,
                           '(t_cond,t_horizon)': (t_cond, t_horizon),
                           'var_type': var_type,
                           'var_names': var_names,
                           'val_mse': val_mse['total'].item(),
                           'val_var01_mse': val_mse['var01'].item(),
                           'val_var0102_mse': val_mse['var0102'].item(),
                           'val_var0203_mse': val_mse['var0203'].item(),
                           'val_var03_mse': val_mse['var03'].item(),
                           'test_mse': test_mse['total'].item(),
                           'test_var01_mse': test_mse['var01'].item(),
                           'test_var0102_mse': test_mse['var0102'].item(),
                           'test_var0203_mse': test_mse['var0203'].item(),
                           'test_var03_mse': test_mse['var03'].item(),
                           }
                    res_list.append(res)
                    i += 1

                    if i % 10 == 0 and save:
                        R = pd.DataFrame(res_list)
                        R.to_csv(os.path.join(EXPERIMENT_DIR, "logs",
                                 sweep_name, "forecasting_results_all_part2.csv"))

    R = pd.DataFrame(res_list)
    if save:
        R.to_csv(os.path.join(EXPERIMENT_DIR, "logs",
                 sweep_name, "forecasting_results_all_part2.csv"))
    return R


if __name__ == "__main__":
    experiment_name = "Seq2Seq_transformer_Random"
    version_name = None

    best_runs = pd.DataFrame([
        {'fold': 0, 'run_name': 'version_90'},
        {'fold': 1, 'run_name': 'version_96'},
        {'fold': 2, 'run_name': 'version_102'},
        {'fold': 3, 'run_name': 'version_78'},
        {'fold': 4, 'run_name': 'version_114'}
    ])

    sweep_name = '4sw35m8cnw_Seq2Seq_transformer_MM2'
    folds = [0, 1, 2, 3, 4]
    group_select = ["all", "IGG", "IGA"]
    t_cond_horizon_pairs = [(6, 12), (12, 6), (12, 12)]
    var_types = ["var-all", "var-serum", "var-chem"]

    df = get_forecasting_results_df(sweep_name,
                                    best_runs,
                                    folds,
                                    subgroups=group_select,
                                    t_cond_horizon_pairs=t_cond_horizon_pairs,
                                    var_types=var_types,
                                    save=True)
    print(df)
