from re import L
from mm79.data.TMMDataModule import TMMDataModule
from mm79.data.TMM1DataModule import TMM1DataModule
from mm79.data.randomDataModule import RandomDataModule
from mm79.data.syntheticDataModule import SyntheticDataModule
from mm79.data.mmsyntheticDataModule import SyntheticMM2DataModule, SyntheticMM3DataModule
from mm79.data.compasDataModule import CompasDataModule
from mm79.data.syntheticSurvival import SyntheticSurvivalDataModule
from mm79.data.CustomDataModule import CustomDataModule
from mm79.models.transformer import Treatformer, FrozenTreatformer
from mm79.models.rnn import RNN
from mm79.models.dummy_model import LastObs, Prophet
from mm79.models.survival_cluster_models import SurvivalClassifier, VariationalMod
from mm79.models.dmm.dmm import DMM
from mm79.train_modules.train_module import TrainModule, FineTuneModule
from mm79.train_modules.train_module_dmm import DMMTrainModule
from mm79.train_modules.event_utils import extract_quantiles, extract_dataset, time_censoring_processing
from mm79.train_modules.event_utils import cumulative_dyn_auc, time_censoring_proc, select_event_type
from mm79.models.baseline_models import event_model_conversion, predict_scores
from sksurv.metrics import concordance_index_ipcw, brier_score, concordance_index_censored
import yaml
import pandas as pd
import os
import itertools
import collections.abc
import pytorch_lightning as pl
import torch
import numpy as np
from mm79 import EXPERIMENT_DIR, TYPE_VAR_DICT
import random
import pickle
from mm79 import AE_ALL_DIM  # number of AE in total
from mm79 import EVENT_SHIFT

def typical_result_dict():
    d = {"val_loss": [],
         "val_loss_mse": [],
         "val_loss_event": [],
         "val_roc_auc_25": [],
         "val_roc_auc_50": [],
         "val_roc_auc_75": [],
         "val_concordance_index_event": [],
         "test_loss": [],
         "test_loss_mse": [],
         "test_loss_event": [],
         "test_roc_auc_25": [],
         "test_roc_auc_50": [],
         "test_roc_auc_75": [],
         "test_concordance_index_event": [],
         "overall_val_loss": [],
         "overall_test_loss": [],
         }

    for ae_idx in range(AE_ALL_DIM):
        d[f"val_concordance_index_ae_{ae_idx}"] = []
        d[f"test_concordance_index_ae_{ae_idx}"] = []
    return d


def model_type_class(model_type):
    if model_type == 'transformer':
        return Treatformer
    elif model_type == 'frozen_transformer':
        return FrozenTreatformer
    elif model_type == 'dummy':
        return LastObs
    elif model_type == 'rnn':
        return RNN
    elif model_type == 'dmm':
        return DMM
    else:
        raise ValueError(f'Unknown model type {model_type}')


def train_module_type_class(model_type):
    if model_type == 'dmm':
        return DMMTrainModule
    elif model_type == "transformer":
        return TrainModule
    elif model_type == "frozen_transformer":
        return FineTuneModule
    elif model_type == "rnn":
        return TrainModule


def cluster_model_type_class(model_type):
    if model_type == 'simple':
        return SurvivalClassifier
    if model_type == "variational":
        return VariationalMod
    else:
        raise ValueError(f'Unknown model type {model_type}')


def data_type_class(data_type):
    if data_type == 'Random':
        return RandomDataModule
    elif data_type == 'Synthetic':
        return SyntheticDataModule
    elif data_type == "SyntheticMM2":
        return SyntheticMM2DataModule
    elif data_type == 'MM2':
        return TMMDataModule
    elif data_type == 'MM1':
        return TMM1DataModule
    elif data_type == "Compas":
        return CompasDataModule
    elif data_type == "SyntheticSurv":
        return SyntheticSurvivalDataModule
    elif data_type == "Custom":
        return CustomDataModule
    else:
        raise ValueError(f'Unknown data type {data_type}')


def get_var_labels(var_array):
    return (var_array < 0.1).astype(float) * 0 \
        + ((var_array >= 0.1) * (var_array < 0.2)).astype(float) * 1 \
        + ((var_array >= 0.2) * (var_array < 0.3)).astype(float) * 2 \
        + (var_array >= 0.3).astype(float) * 3


def get_wandb_run(run_name, name="edebrouwer", project="mm-long"):
    import wandb
    api = wandb.Api()
    run = api.run(f"{name}/{project}/{run_name}")
    return run


def evaluate_dummy_pred(fold, t_cond, t_horizon, data_type, restricted_pred_features_set=True, subgroup_strat="myeloma-type"):
    data_cls = data_type_class(data_type)
    data_args = data_cls.defaults()

    model_cls = LastObs

    module_args = TrainModule.defaults()
    full_args = {**data_args, **module_args}
    full_args["t_cond"] = t_cond
    full_args["t_horizon"] = t_horizon
    full_args["fold"] = fold
    full_args["data_type"] = data_type

    dataset = data_cls(
        **full_args, restricted_pred_features_set=restricted_pred_features_set)
    dataset.prepare_data()

    subgroup_dict = get_subgroup_dict(dataset, subgroup_strat='myeloma-type')

    dummy_model = TrainModule(
        model_cls=model_cls, baseline_size=dataset.baseline_size,
        input_long_size=dataset.input_long_size,
        treatment_size=dataset.treatment_size,
        prediction_idx=dataset.prediction_idx, prediction_names=dataset.prediction_names,
        input_idx=dataset.input_idx, emission_proba=False,
        quantiles_event=None,
        et_train_event=None,
        quantiles_ae=None,
        et_train_ae=None,
        **full_args)

    dummy_trainer = pl.Trainer(logger=False, gpus=0)

    dummy_preds = dummy_trainer.predict(
        dummy_model, dataset.test_dataloader())

    dummy_forecast_future_all = torch.cat(
        [pred["forecast_future"] for pred in dummy_preds])
    dummy_target_future_all = torch.cat(
        [pred["target_future"] for pred in dummy_preds])
    dummy_mask_future_all = torch.cat(
        [pred["mask_future"] for pred in dummy_preds])
    B = torch.cat(
        [pred["B"] for pred in dummy_preds])

    B_orig = dataset.unnormalize_B(B)

    res_dict = {}
    for subgroup_name, subgroup in subgroup_dict.items():
        # extract datasets
        idx, comp, val = subgroup
        if subgroup_strat == 'treatment-response':
            subgroup_idxs = get_idxs_from_F(pids_, subgroup)
        elif comp == '==':
            subgroup_idxs = np.where(np.abs(B_orig[:, idx]-val) < 1e-2)
        elif comp == '<':
            subgroup_idxs = np.where(B_orig[:, idx] < val)
        elif comp == '>':
            subgroup_idxs = np.where(B_orig[:, idx] > val)
        elif comp == "?":
            subgroup_idxs = np.ones(B_orig.shape[0], dtype=bool)

        dummy_forecast_future = dummy_forecast_future_all[
            subgroup_idxs]
        dummy_target_future = dummy_target_future_all[subgroup_idxs]
        dummy_mask_future = dummy_mask_future_all[subgroup_idxs]

        pids = dataset.pids
        test_idx = dataset.test_idx

        prediction_idx = dataset.prediction_idx
        prediction_names = dataset.prediction_names

        mse = dummy_model.compute_mse(
            dummy_forecast_future, dummy_target_future[:, :, prediction_idx], dummy_mask_future[:, :, prediction_idx])

        res_dict[subgroup_name] = {"mse": mse,
                                   "forecast_future": dummy_forecast_future,
                                   "target_future": dummy_target_future,
                                   "mask_future": dummy_mask_future,
                                   "B": B[subgroup_idxs],
                                   "prediction_names": prediction_names}

    return res_dict


def get_results_wandb_sweep(sweep_name, constraints, user="edebrouwer", project="mm-long", evaluate_dummy=False):
    """
    Does same thing as get_results_sweep but with wandb/runs
    """
    import wandb
    api = wandb.Api()
    sweep = api.sweep(f"{user}/{project}/{sweep_name}")
    runs = sweep.runs

    overall_res_list = []

    keys, values = zip(*constraints.items())
    permutations_dicts = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    for constraint in permutations_dicts:

        res_dict = {"val_loss": [], "test_loss": []}
        runs_fold = [r for r in runs if check_wandb_constraints(r, constraint)]

        result_dict = constraint.copy()
        result_dict["val_loss"] = []
        result_dict["test_loss"] = []
        result_dict["run_name"] = []

        if evaluate_dummy:
            result_dict["dummy_test_loss"] = []

        for run in runs_fold:
            result_dict["val_loss"].append(
                run.summary["restored_val_loss"])
            result_dict["test_loss"].append(
                run.summary["restored_test_loss"])
            result_dict["run_name"].append(run.id)
            if evaluate_dummy:
                dummy_preds = evaluate_dummy_wandb_run(run)
                result_dict["dummy_test_loss"].append(
                    dummy_preds["dummy_test_loss"].item())

        overall_res_list.append(pd.DataFrame(result_dict))

    overall_res = pd.concat(overall_res_list)
    return overall_res


def populate_ensemble_dict(ensemble_dict, preds_dict, subgroup):
    '''
        TODO: comment this function
    '''
    forecast = preds_dict[subgroup]["forecast_future"]
    target = preds_dict[subgroup]["target_future"]
    mask = preds_dict[subgroup]["mask_future"]
    y_pred = preds_dict[subgroup]["y_pred"]
    E = preds_dict[subgroup]["E"]
    Y = preds_dict[subgroup]["Y"]
    pred_idx = preds_dict[subgroup]["prediction_idx"]

    if subgroup not in ensemble_dict.keys():
        ensemble_dict[subgroup] = {
            "Y": Y,
            "E": E,
            "M": mask,
            "Y_pred": [y_pred],
            "prediction_idx": pred_idx,
            "forecast": [forecast],
            "target": target,
            "overall_loss": [],
            "model": preds_dict[subgroup]["model"],
            "t_cond": preds_dict[subgroup]["t_cond"]
        }
    else:
        ensemble_dict[subgroup]["forecast"].append(forecast)
        ensemble_dict[subgroup]["Y_pred"].append(y_pred)
        assert (ensemble_dict[subgroup]["target"] == target).all()
        assert (ensemble_dict[subgroup]["Y"] == Y).all()

    return ensemble_dict


def process_ensemble_dict(test_ensemble_dict, val_ensemble_dict, reference_dict):
    reference_dict["subgroup"] = []
    reference_dict["mse"] = []
    reference_dict["roc_auc_25"] = []
    reference_dict["roc_auc_50"] = []
    reference_dict["roc_auc_75"] = []
    reference_dict["roc_auc_mean"] = []
    reference_dict["split"] = []

    dicts = {'test': test_ensemble_dict, 'val': val_ensemble_dict}

    for prefix, di in dicts.items():
        for subgroup, subdict in di.items():
            overall_loss = np.array(subdict["overall_loss"])
            top_20_idxs = np.argsort(overall_loss)[:20]
            weights = torch.softmax(torch.Tensor(overall_loss[top_20_idxs]), 0)
            t_cond = subdict["t_cond"]
            model = subdict["model"]

            # Y_pred
            if subdict["Y_pred"][0] is not None:
                top_20_Y_pred = torch.stack(subdict["Y_pred"])[top_20_idxs]
                ensemble_Y_pred = torch.mean(
                    top_20_Y_pred*weights[:, None, None, None], dim=0)
                roc_auc_event = model.evaluate_event_auc(
                    event_risk=ensemble_Y_pred[:, 0],
                    Y=subdict["Y"],
                    E=subdict["E"],
                    t_cond=t_cond
                )
            else:
                roc_auc_event = [np.nan, np.nan, np.nan]

            # forecast
            top_20_forecast = torch.stack(subdict["forecast"])[top_20_idxs]
            ensemble_forecast = torch.mean(
                top_20_forecast*weights[:, None, None, None], dim=0)
            prediction_idx = subdict["prediction_idx"]
            mse = model.compute_mse(
                ensemble_forecast,
                subdict["target"][:, :, prediction_idx],
                subdict["M"][:, :, prediction_idx]
            )

            reference_dict["subgroup"].append(subgroup)
            reference_dict[f"mse"].append(mse.item())
            reference_dict[f"roc_auc_25"].append(roc_auc_event[0])
            reference_dict[f"roc_auc_50"].append(roc_auc_event[1])
            reference_dict[f"roc_auc_75"].append(roc_auc_event[2])
            reference_dict[f"roc_auc_mean"].append(
                np.mean(np.array(roc_auc_event)))
            reference_dict["split"].append(prefix)

    return reference_dict


def get_ensemble_results(sweep_name, constraints, evaluation_params=None):
    """_summary_

    Args:
        sweep_name (_type_): _description_
        constraints (_type_): _description_
        evaluation_params (_type_, optional): _description_. Defaults to None.
            var_bin (int, optional): _description_. The variance bin to use in the evaluation. If -1, uses all patients.

    Returns:
        _type_: _description_
    """

    dataset_name = evaluation_params.get("dataset_name", "MM2")
    print(f"Evaluating on {dataset_name} dataset...")

    log_dir = os.path.join(EXPERIMENT_DIR, "logs")
    exp_dirs = [os.path.join(log_dir, d)
                for d in os.listdir(log_dir) if sweep_name in d]
    assert(len(exp_dirs) == 1)

    run_dirs = []
    for exp_dir in exp_dirs:
        run_dirs += [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]

    import yaml
    with open(os.path.join(EXPERIMENT_DIR, "configs", "records", sweep_name + ".yml")) as f:
        sweep_hparams = yaml.load(
            f, Loader=yaml.Loader)

    for k, v in sweep_hparams.items():
        if k in constraints:
            sweep_hparams[k] = constraints[k]

    for k, v in constraints.items():
        if k not in sweep_hparams:
            sweep_hparams[k] = v

    # How to get the number of bootstraps from the fine tune sweep ?
    bootstraps = sweep_hparams.pop("bootstrap_seed", None)
    if bootstraps is None:  # fine tune sweep
        bootstraps = sweep_hparams.pop("n_booststraps", None)
        if bootstraps is None:
            bootstraps = [0, 1, 2, 3, 4]
        else:
            bootstraps = list(range(bootstraps))

    _ = sweep_hparams.pop("pretrained_sweep_name", None)
    _ = sweep_hparams.pop("pretrained_version_name", None)
    _ = sweep_hparams.pop("program", None)
    _ = sweep_hparams.pop("n_bootstraps", None)

    if len(sweep_hparams) > 0:
        keys, values = zip(*sweep_hparams.items())
        permutations_dicts = [dict(zip(keys, v))
                              for v in itertools.product(*values)]
    else:
        permutations_dicts = [{}]

    subgroups = get_subgroup_list(evaluation_params["subgroup_strat"])

    t_cond = evaluation_params["t_cond"]
    t_horizon = evaluation_params["t_horizon"]

    df = []
    for constraint in permutations_dicts:

        y_pred_test = {subgroup: [] for subgroup in subgroups}
        y_pred_val = {subgroup: [] for subgroup in subgroups}
        forecast_test = {subgroup: [] for subgroup in subgroups}
        forecast_val = {subgroup: [] for subgroup in subgroups}
        target_test = {subgroup: None for subgroup in subgroups}
        target_val = {subgroup: None for subgroup in subgroups}
        mask_test = {subgroup: None for subgroup in subgroups}
        mask_val = {subgroup: None for subgroup in subgroups}
        Y_test = {subgroup: None for subgroup in subgroups}
        Y_val = {subgroup: None for subgroup in subgroups}
        E_test = {subgroup: None for subgroup in subgroups}
        E_val = {subgroup: None for subgroup in subgroups}
        Y_ae_test = {subgroup: None for subgroup in subgroups}
        Y_ae_val = {subgroup: None for subgroup in subgroups}
        E_ae_test = {subgroup: None for subgroup in subgroups}
        E_ae_val = {subgroup: None for subgroup in subgroups}
        quantiles_event = {subgroup: [] for subgroup in subgroups}
        et_train_event_all = {subgroup: [] for subgroup in subgroups}
        quantiles_ae = {subgroup: [] for subgroup in subgroups}
        et_train_ae_all = {subgroup: [] for subgroup in subgroups}

        df_list = []
        for i_b, bootstrap in enumerate(bootstraps):
            if bootstrap == -1:  # -1 is for no bootstrap
                continue
            constraint_ = {**constraint, **{"bootstrap_seed": bootstrap}}

            print(f"Constraint")
            print(constraint_)
            runs_ok = [
                r for r in run_dirs if check_constraints(r, constraint_)]
            # runs_ok = [r for r in runs_ok if check_constraints(r,constraints)]
            assert(len(runs_ok) == 1)
            run = runs_ok[0]

            print(f"Processing run {run}")

            for subgroup in subgroups:
                with open(os.path.join(run, f"test_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl"), 'rb') as f:
                    test_dict = pickle.load(f)
                with open(os.path.join(run, f"val_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl"), 'rb') as f:
                    val_dict = pickle.load(f)

                # Y_pred
                if test_dict["y_pred"] is not None:
                    y_pred_test[subgroup].append(test_dict["y_pred"])
                    y_pred_val[subgroup].append(val_dict["y_pred"])

                forecast_test[subgroup].append(test_dict["forecast_future"])
                forecast_val[subgroup].append(val_dict["forecast_future"])
                quantiles_event[subgroup].append(test_dict["quantiles_event"])
                et_train_event_all[subgroup].append(
                    test_dict["et_train_event"])
                quantiles_ae[subgroup].append(test_dict["quantiles_ae"])
                et_train_ae_all[subgroup].append(test_dict["et_train_ae"])

                if i_b == 0:
                    target_test[subgroup] = test_dict["target_future"]
                    mask_test[subgroup] = test_dict["mask_future"]

                    target_val[subgroup] = val_dict["target_future"]
                    mask_val[subgroup] = val_dict["mask_future"]

                    Y_test[subgroup] = test_dict["Y"]
                    E_test[subgroup] = test_dict["E"]

                    Y_val[subgroup] = val_dict["Y"]
                    E_val[subgroup] = val_dict["E"]

                    Y_ae_test[subgroup] = test_dict["Y_ae"]
                    E_ae_test[subgroup] = test_dict["E_ae"]

                    Y_ae_val[subgroup] = val_dict["Y_ae"]
                    E_ae_val[subgroup] = val_dict["E_ae"]

                else:
                    assert((test_dict["target_future"] ==
                           target_test[subgroup]).all())
                    assert((test_dict["mask_future"] ==
                           mask_test[subgroup]).all())
                    assert((val_dict["target_future"] ==
                           target_val[subgroup]).all())
                    assert((val_dict["mask_future"] ==
                           mask_val[subgroup]).all())
                    assert((test_dict["Y"] == Y_test[subgroup]).all())
                    assert((test_dict["E"] == E_test[subgroup]).all())
                    assert((val_dict["Y"] == Y_val[subgroup]).all())
                    assert((val_dict["E"] == E_val[subgroup]).all())
                    assert((test_dict["Y_ae"] == Y_ae_test[subgroup]).all())
                    assert((test_dict["E_ae"] == E_ae_test[subgroup]).all())
                    assert((val_dict["Y_ae"] == Y_ae_val[subgroup]).all())
                    assert((val_dict["E_ae"] == E_ae_val[subgroup]).all())

        forecast_test_mean = {subgroup: np.mean(
            np.stack(forecast_test[subgroup]), axis=0) for subgroup in subgroups}
        forecast_val_mean = {subgroup: np.mean(
            np.stack(forecast_val[subgroup]), axis=0) for subgroup in subgroups}

        if len(y_pred_test[subgroup]) != 0:
            y_pred_test_mean = {subgroup: np.mean(
                np.stack(y_pred_test[subgroup]), axis=0) for subgroup in subgroups}
            y_pred_val_mean = {subgroup: np.mean(
                np.stack(y_pred_val[subgroup]), axis=0) for subgroup in subgroups}

        val_mses = {}
        test_mses = {}
        val_mses_serum = {}
        test_mses_serum = {}
        val_mses_chem = {}
        test_mses_chem = {}

        test_aucs = {}
        val_aucs = {}
        for subgroup in subgroups:

            if len(y_pred_test[subgroup]) > 0:  # survival prediction

                quant_event = quantiles_event[subgroup][0]
                et_train_event = et_train_event_all[subgroup][0]

                quant_ae = quantiles_ae[subgroup][0]
                et_train_ae = et_train_ae_all[subgroup][0]

                t_cond_event = t_cond - EVENT_SHIFT

                et_test_event = time_censoring_proc(
                    E_test[subgroup][Y_test[subgroup] > t_cond_event], Y_test[subgroup][Y_test[subgroup] > t_cond_event] - t_cond_event)
                et_val_event = time_censoring_proc(
                    E_val[subgroup][Y_val[subgroup] > t_cond_event], Y_val[subgroup][Y_val[subgroup] > t_cond_event] - t_cond_event)

                et_test_ae = [time_censoring_proc(
                    E_ae_test[subgroup][Y_ae_test[subgroup]
                                        [:, idx_ae] > t_cond_event, idx_ae][:, None],
                    Y_ae_test[subgroup][Y_ae_test[subgroup][:, idx_ae] > t_cond_event, idx_ae][:, None] - t_cond_event) for idx_ae in range(AE_ALL_DIM)]
                et_val_ae = [time_censoring_proc(
                    E_ae_val[subgroup][Y_ae_val[subgroup]
                                       [:, idx_ae] > t_cond_event, idx_ae][:, None],
                    Y_ae_val[subgroup][Y_ae_val[subgroup][:, idx_ae] > t_cond_event, idx_ae][:, None] - t_cond_event) for idx_ae in range(AE_ALL_DIM)]

                test_risk_event = y_pred_test_mean[subgroup][:,
                                                             0, 0][Y_test[subgroup][:, 0] > t_cond_event]
                val_risk_event = y_pred_val_mean[subgroup][:,
                                                           0, 0][Y_val[subgroup][:, 0] > t_cond_event]

                test_risk_ae = [y_pred_test_mean[subgroup][:, 0, idx_ae+1][Y_ae_test[subgroup]
                                                                           [:, idx_ae] > t_cond_event] for idx_ae in range(AE_ALL_DIM)]
                val_risk_ae = [y_pred_val_mean[subgroup][:, 0, idx_ae+1][Y_ae_val[subgroup]
                                                                         [:, idx_ae] > t_cond_event] for idx_ae in range(AE_ALL_DIM)]

                test_roc_auc_25 = cumulative_dyn_auc(
                    et_train_event, et_test_event, test_risk_event, quant_event[0])[0][0]
                test_roc_auc_50 = cumulative_dyn_auc(
                    et_train_event, et_test_event, test_risk_event, quant_event[1])[0][0]
                test_roc_auc_75 = cumulative_dyn_auc(
                    et_train_event, et_test_event, test_risk_event, quant_event[2])[0][0]

                def concordance_index_(et_train_, et_test_, test_risk_):
                    try:
                        return concordance_index_ipcw(et_train_, et_test_, test_risk_)[0]
                    except:
                        return np.nan

                test_ccidx_event = concordance_index_(et_train_event, et_test_event, test_risk_event)

                test_ccidx_ae = [concordance_index_(
                    et_train_ae[idx_ae], et_test_ae[idx_ae], test_risk_ae[idx_ae]) if et_test_ae[idx_ae]["e"].sum() > 0 else np.nan for idx_ae in range(AE_ALL_DIM)]

                test_aucs[subgroup] = [test_roc_auc_25,
                                       test_roc_auc_50, test_roc_auc_75]

                val_roc_auc_25 = cumulative_dyn_auc(
                    et_train_event, et_val_event, val_risk_event, quant_event[0])[0][0]
                val_roc_auc_50 = cumulative_dyn_auc(
                    et_train_event, et_val_event, val_risk_event, quant_event[1])[0][0]
                val_roc_auc_75 = cumulative_dyn_auc(
                    et_train_event, et_val_event, val_risk_event, quant_event[2])[0][0]

                val_ccidx_event = concordance_index_(
                    et_train_event, et_val_event, val_risk_event)

                val_ccidx_ae = [concordance_index_(
                    et_train_ae[idx_ae], et_val_ae[idx_ae], val_risk_ae[idx_ae]) if et_val_ae[idx_ae]["e"].sum() > 0 else np.nan for idx_ae in range(AE_ALL_DIM)]

                val_aucs[subgroup] = [val_roc_auc_25,
                                      val_roc_auc_50, val_roc_auc_75]
            else:
                val_ccidx_event = np.nan
                test_ccidx_event = np.nan
                val_ccidx_ae = [np.nan for idx_ae in range(AE_ALL_DIM)]
                test_ccidx_ae = [np.nan for idx_ae in range(AE_ALL_DIM)]
                val_aucs[subgroup] = [np.nan, np.nan, np.nan]
                test_aucs[subgroup] = [np.nan, np.nan, np.nan]

            # forecasting

            # Evaluating for different variables subsets
            if "prediction_names" not in test_dict:
                print(
                    "PREDICTION NAMES NOT FOUND ! - Temporarily hardcoding them with all variables")
                feat_names = ['Albumin (g/L)', 'Alkaline Phosphatase (U/L)',
                              'Alanine Aminotransferase (U/L)',
                              'Aspartate Aminotransferase (U/L)', 'Bilirubin (umol/L)',
                              'Blood Urea Nitrogen (mmol/L)', 'Calcium (mmol/L)',
                              'Chloride (mmol/L)', 'Carbon Dioxide (mmol/L)',
                              'Corrected Calcium (mmol/L)', 'Creatinine (umol/L)',
                              'Glomerular Filtration Rate Adj for BSA via CKD-EPI (mL/min/1.73m2)',
                              'Glucose (mmol/L)', 'Hematocrit', 'Hemoglobin (g/L)',
                              'Potassium (mmol/L)', 'Lactate Dehydrogenase (U/L)',
                              'Lymphocytes (10^9/L)', 'Magnesium (mmol/L)', 'Monocytes (10^9/L)',
                              'Neutrophils (10^9/L)', 'Phosphate (mmol/L)', 'Platelets (10^9/L)',
                              'Protein (g/L)', 'Serum Globulin (g/L)', 'Sodium (mmol/L)',
                              'SPEP Gamma Globulin (g/L)', 'SPEP Kappa Light Chain, Free (mg/L)',
                              'SPEP Kappa Lt Chain,Free/Lambda Lt Chain,Free',
                              'SPEP Lambda Light Chain, Free (mg/L)',
                              'SPEP Monoclonal Protein (g/L)', 'Serum TM Albumin/Globulin',
                              'Immunoglobulin A (g/L)', 'Immunoglobulin G (g/L)',
                              'Immunoglobulin M (g/L)', 'Urine Albumin (%)',
                              'UPEP Monoclonal Protein (mg/day)', 'Urate (umol/L)',
                              'Leukocytes (10^9/L)']
            else:
                feat_names = test_dict['prediction_names']

            all_var_idx = np.array([list(feat_names).index(x)
                                   for x in feat_names])
            serum_var_idx = np.array([list(feat_names).index(x)
                                     for x in TYPE_VAR_DICT["var-serum"]])
            chem_var_idx = np.array([list(feat_names).index(x)
                                    for x in TYPE_VAR_DICT["var-chem"]])

            # Evaluating for the desired variance bin

            if evaluation_params is None:
                var_bin = None
            else:
                var_bin = evaluation_params.get("var_bin", None)

            future_variance_test = np.var(
                target_test[subgroup].numpy(), axis=1).mean(1)
            future_variance_val = np.var(
                target_val[subgroup].numpy(), axis=1).mean(1)
            var_labels_test = get_var_labels(future_variance_test)
            var_labels_val = get_var_labels(future_variance_val)
            if var_bin is None:
                var_idx_val = np.array(
                    [True for _ in var_labels_val])
                var_idx_test = np.array(
                    [True for _ in var_labels_test])
            else:
                var_idx_val = np.array(var_labels_val == var_bin)
                var_idx_test = np.array(var_labels_test == var_bin)

            forecast_test_mean_ = forecast_test_mean[subgroup][var_idx_test]
            forecast_val_mean_ = forecast_val_mean[subgroup][var_idx_val]
            target_test_ = target_test[subgroup][var_idx_test]
            target_val_ = target_val[subgroup][var_idx_val]
            mask_test_ = mask_test[subgroup][var_idx_test]
            mask_val_ = mask_val[subgroup][var_idx_val]

            test_mses[subgroup] = (((forecast_test_mean_-target_test_.numpy()) **
                                    2)*mask_test_.numpy()).sum() / mask_test_.numpy().sum()
            val_mses[subgroup] = (((forecast_val_mean_-target_val_.numpy()) **
                                   2)*mask_val_.numpy()).sum() / mask_val_.numpy().sum()

            test_mses_serum[subgroup] = (((forecast_test_mean_[..., serum_var_idx]-target_test_.numpy()[..., serum_var_idx]) **
                                          2)*mask_test_[..., serum_var_idx].numpy()).sum() / mask_test_[..., serum_var_idx].numpy().sum()
            val_mses_serum[subgroup] = (((forecast_val_mean_[..., serum_var_idx]-target_val_.numpy()[..., serum_var_idx]) **
                                         2)*mask_val_.numpy()[..., serum_var_idx]).sum() / mask_val_.numpy()[..., serum_var_idx].sum()

            test_mses_chem[subgroup] = (((forecast_test_mean_[..., chem_var_idx]-target_test_.numpy()[..., chem_var_idx]) **
                                         2)*mask_test_.numpy()[..., chem_var_idx]).sum() / mask_test_.numpy()[..., chem_var_idx].sum()
            val_mses_chem[subgroup] = (((forecast_val_mean_[..., chem_var_idx]-target_val_.numpy()[..., chem_var_idx]) **
                                        2)*mask_val_.numpy()[..., chem_var_idx]).sum() / mask_val_.numpy()[..., chem_var_idx].sum()

            df_dict = {"test_mse": test_mses[subgroup],
                       "val_mse": val_mses[subgroup],
                       "test_mse_serum": test_mses_serum[subgroup],
                       "val_mse_serum": val_mses_serum[subgroup],
                       "test_mse_chem": test_mses_chem[subgroup],
                       "val_mse_chem": val_mses_chem[subgroup],
                       "val_auc": np.mean(val_aucs[subgroup]),
                       "test_auc": np.mean(test_aucs[subgroup]),
                       "val_concordance_event": val_ccidx_event,
                       "test_concordance_event": test_ccidx_event}
            for idx_ae in range(AE_ALL_DIM):
                df_dict[f"val_concordance_ae_{idx_ae}"] = val_ccidx_ae[idx_ae]
                df_dict[f"test_concordance_ae_{idx_ae}"] = test_ccidx_ae[idx_ae]

            df_subgroup = pd.DataFrame({**constraint, **df_dict}, index=[0])
            df_subgroup["subgroup"] = subgroup
            df_list.append(df_subgroup)
        df.append(pd.concat(df_list))
    return pd.concat(df)


def get_results_sweep(sweep_name,
                      constraints,
                      evaluation_params=None,
                      force_recompute=False,
                      hyper_params=None,
                      MM1=False):
    """ Collects the results of a sweep, structured according to the constraints dictionary

    Args:
        sweep_name (str): the unique identifier of the sweep

        constraints (dict): dictionary of the constraints to check for the run 
            to be classified as true
                Example : {"fold":[0,1,2,3,4]}
            or {"fold":[0,1,2,3,4], "model_type":["transformer", "lstm"]} to structure both 
            by fold and by another hyperparameter

        evaluation_params (dict): dictionary of the parameters to evaluate the model 
            on. If left as None, will just retrieve the losses of the run. In auto-regressive 
            mode, can be used to set t_cond and t_horizon for the evaluation. 
            Example : {"t_cond": 6, "t_horizon": 10, "evaluate_event":True}. If evaluate_event is 
            set to True, it will also evaluate the performance on the event prediction task.

        hyper_params (list): list of hyperparameters to retrieve from the run. If None, no hyper params is returned.

        MM1  : wether to evaluate the model on MM1 or MM2. If True, will evaluate on MM1. If False, will evaluate on MM2.

    Returns:
        pd.DataFrame: DataFrame with all the results
    """

    log_dir = os.path.join(EXPERIMENT_DIR, "logs")
    exp_dirs = [os.path.join(log_dir, d)
                for d in os.listdir(log_dir) if sweep_name in d]

    run_dirs = []
    for exp_dir in exp_dirs:
        run_dirs += [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]

    overall_res_list = []
    overall_ensemble_list = []

    keys, values = zip(*constraints.items())
    permutations_dicts = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    random.shuffle(permutations_dicts)
    for ci, constraint in enumerate(permutations_dicts):

        res_dict = {"val_loss": [], "test_loss": []}
        runs_fold = [r for r in run_dirs if check_constraints(r, constraint)]

        result_dict = typical_result_dict()
        result_dict.update(constraint.copy())
        result_dict["run_name"] = []
        result_dict["subgroup"] = []

        if hyper_params is not None:
            for hp in hyper_params:
                result_dict[hp] = []

        test_ensemble_dict = {}
        val_ensemble_dict = {}

        for run in runs_fold:
            print(f"EXTRACTING RESULTS FROM {run}")
            if evaluation_params is not None:
                t_cond = evaluation_params["t_cond"]
                t_horizon = evaluation_params["t_horizon"]
                subgroup_strat = evaluation_params["subgroup_strat"]

                results_path = os.path.join(run, "recovered_results.csv")
                if os.path.exists(results_path):
                    run_results = pd.read_csv(results_path)
                    if hyper_params is not None:
                        class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
                            def ignore_unknown(self, node):
                                return None

                        SafeLoaderIgnoreUnknown.add_constructor(
                            None, SafeLoaderIgnoreUnknown.ignore_unknown)

                        with open(os.path.join(run, "hparams.yaml")) as f:
                            hparams = yaml.load(
                                f, Loader=SafeLoaderIgnoreUnknown)
                else:
                    print(f'WARNING: no recovered results found for {run}')
                    continue
                if "t_cond" not in run_results.columns:
                    run_results["t_cond"] = np.nan
                    run_results["t_horizon"] = np.nan
                    run_results["subgroup_strat"] = np.nan
                    run_results["subgroup"] = np.nan

                if "dataset" not in run_results.columns:
                    run_results["dataset"] = np.nan
                if MM1:
                    dataset_name = "MM1"
                else:
                    dataset_name = "MM2"

                if force_recompute:
                    run_results = run_results.loc[run_results["t_cond"].isna()].copy(
                    )
                    run_results["subgroup_strat"] = np.nan

                if run_results.loc[(run_results["t_cond"] == t_cond) & (run_results["t_horizon"] == t_horizon) & (run_results["subgroup_strat"] == subgroup_strat) & (run_results["dataset"] == dataset_name)].shape[0] > 0:
                    run_result_ = run_results.loc[(run_results["t_cond"] == t_cond) & (
                        run_results["t_horizon"] == t_horizon) & (run_results["subgroup_strat"] == subgroup_strat) & (run_results["dataset"] == dataset_name)]

                    for subgroup in run_result_["subgroup"].unique():
                        run_result_subgroup = run_result_.loc[run_result_[
                            "subgroup"] == subgroup]

                        for metric in run_result_subgroup["metric"].values:
                            result_dict[metric].append(
                                run_result_subgroup.loc[run_result_["metric"] == metric, "value"].values[0])
                        result_dict["run_name"].append(run.split("/")[-1])

                        if "val_loss_epoch" in run_results.metric.values:
                            result_dict["overall_val_loss"].append(
                                run_results.loc[run_results.metric == "val_loss_epoch", "value"].item())
                            result_dict["overall_test_loss"].append(
                                run_results.loc[run_results.metric == "test_loss_epoch", "value"].item())
                        else:
                            result_dict["overall_val_loss"].append(
                                run_results.loc[run_results.metric == "val_loss", "value"].item())
                            result_dict["overall_test_loss"].append(
                                run_results.loc[run_results.metric == "test_loss", "value"].item())
                            result_dict["val_loss_event"].append(
                                run_results.loc[run_results.metric == "test_loss_event", "value"].item())
                            result_dict["test_loss_event"].append(
                                run_results.loc[run_results.metric == "test_loss_event", "value"].item())

                        result_dict["subgroup"].append(
                            run_result_subgroup["subgroup"].values[0])

                        if hyper_params is not None:
                            for hyper_param_name in hyper_params:
                                result_dict[hyper_param_name].append(
                                    hparams[hyper_param_name])

                    # val_result = run_results.loc[(run_results["t_cond"] == t_cond) & (
                    #    run_results["t_horizon"] == t_horizon) & (run_results["metric"] == "val_loss")]["value"].item()
                    # test_result = run_results.loc[(run_results["t_cond"] == t_cond) & (
                    #    run_results["t_horizon"] == t_horizon) & (run_results["metric"] == "test_loss")]["value"].item()

                else:

                    test_preds_dict = get_predictions(
                        experiment_dir=run,
                        t_cond=t_cond,
                        t_horizon=t_horizon,
                        test_fold=True,
                        subgroup_strat=subgroup_strat,
                        MM1=MM1
                    )

                    if MM1:
                        val_preds_dict = test_preds_dict
                    else:
                        val_preds_dict = get_predictions(
                            experiment_dir=run,
                            t_cond=t_cond,
                            t_horizon=t_horizon,
                            test_fold=False,
                            subgroup_strat=subgroup_strat,
                        )

                    for subgroup in test_preds_dict.keys():
                        test_mse = test_preds_dict[subgroup]["mse"].item()
                        val_mse = val_preds_dict[subgroup]["mse"].item()

                        subset_test_dict = dict([(key, value) for key, value in test_preds_dict[subgroup].items()
                                                 if key in ["forecast_future", "target_future", "mask_future", "Y", "E",
                                                            "Y_ae", "E_ae", "y_pred", "prediction_idx", "prediction_names",
                                                            "t_cond", "quantiles_event", "et_train_event", "quantiles_ae", "et_train_ae"]])
                        subset_val_dict = dict([(key, value) for key, value in val_preds_dict[subgroup].items()
                                                if key in ["forecast_future", "target_future", "mask_future", "Y", "E",
                                                           "Y_ae", "E_ae", "y_pred", "prediction_idx", "prediction_names",
                                                           "t_cond", "quantiles_event", "et_train_event", "quantiles_ae", "et_train_ae"]])

                        with open(os.path.join(run, f'test_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl'), 'wb') as f:
                            pickle.dump(subset_test_dict, f)
                        with open(os.path.join(run, f'val_preds_{dataset_name}_tcond_{t_cond}_thorizon{t_horizon}_{subgroup}.pkl'), 'wb') as f:
                            pickle.dump(subset_val_dict, f)

                        run_results = run_results.append(
                            {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "val_loss_mse", "value": val_mse, "dataset": dataset_name}, ignore_index=True)

                        run_results = run_results.append(
                            {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "test_loss_mse", "value": test_mse, "dataset": dataset_name}, ignore_index=True)

                        test_roc_auc_event = test_preds_dict[subgroup]["roc_auc_event"]
                        val_roc_auc_event = val_preds_dict[subgroup]["roc_auc_event"]
                        test_concordance_index_event = test_preds_dict[subgroup]["concordance_index_event"]
                        val_concordance_index_event = val_preds_dict[subgroup]["concordance_index_event"]

                        if test_roc_auc_event is not None:
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "test_roc_auc_25", "value": test_roc_auc_event[0], "dataset": dataset_name}, ignore_index=True)
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "test_roc_auc_50", "value": test_roc_auc_event[1], "dataset": dataset_name}, ignore_index=True)
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "test_roc_auc_75", "value": test_roc_auc_event[2], "dataset": dataset_name}, ignore_index=True)

                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "val_roc_auc_25", "value": val_roc_auc_event[0], "dataset": dataset_name}, ignore_index=True)
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "val_roc_auc_50", "value": val_roc_auc_event[1], "dataset": dataset_name}, ignore_index=True)
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "val_roc_auc_75", "value": val_roc_auc_event[2], "dataset": dataset_name}, ignore_index=True)

                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "test_concordance_index_event", "value": test_concordance_index_event[0], "dataset": dataset_name}, ignore_index=True)
                            run_results = run_results.append(
                                {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": "val_concordance_index_event", "value": val_concordance_index_event[0], "dataset": dataset_name}, ignore_index=True)

                            for i_ae in range(len(test_concordance_index_event[1])):
                                run_results = run_results.append(
                                    {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": f"test_concordance_index_ae_{i_ae}", "value": test_concordance_index_event[1][i_ae], "dataset": dataset_name}, ignore_index=True)
                                run_results = run_results.append(
                                    {"t_cond": t_cond, "t_horizon": t_horizon, "subgroup": subgroup, "subgroup_strat": subgroup_strat, "metric": f"val_concordance_index_ae_{i_ae}", "value": val_concordance_index_event[1][i_ae], "dataset": dataset_name}, ignore_index=True)

                    run_results.to_csv(os.path.join(
                        run, "recovered_results.csv"), index=False)

                    run_result_ = run_results.loc[(run_results["t_cond"] == t_cond) & (
                        run_results["t_horizon"] == t_horizon) & (run_results["subgroup_strat"] == subgroup_strat) & (run_results["dataset"] == dataset_name)]

                    for subgroup in run_result_["subgroup"].unique():
                        run_result_subgroup = run_result_.loc[run_result_[
                            "subgroup"] == subgroup]

                        for metric in run_result_subgroup["metric"].values:
                            result_dict[metric].append(
                                run_result_subgroup.loc[run_result_["metric"] == metric, "value"].values[0])
                        result_dict["run_name"].append(run.split("/")[-1])
                        if "val_loss_epoch" in run_results.metric.values:
                            try:
                                result_dict["overall_val_loss"].append(
                                    run_results.loc[run_results.metric == "val_loss_epoch", "value"].item())
                                result_dict["overall_test_loss"].append(
                                    run_results.loc[run_results.metric == "test_loss_epoch", "value"].item())
                            except:
                                import pdb
                                pdb.set_trace()
                        else:
                            result_dict["overall_val_loss"].append(
                                run_results.loc[run_results.metric == "val_loss", "value"].item())
                            result_dict["overall_test_loss"].append(
                                run_results.loc[run_results.metric == "test_loss", "value"].item())

                            result_dict["val_loss_event"].append(
                                run_results.loc[run_results.metric == "test_loss_event", "value"].item())
                            result_dict["test_loss_event"].append(
                                run_results.loc[run_results.metric == "test_loss_event", "value"].item())

                        result_dict["subgroup"].append(
                            run_result_subgroup["subgroup"].values[0])

            else:
                # TODO: do we ever use this?
                if os.path.exists(os.path.join(run, "recovered_results.csv")):
                    run_results = pd.read_csv(
                        os.path.join(run, "recovered_results.csv"))
                    for metric in run_results["metric"].values:
                        result_dict[metric].append(
                            run_results.loc[run_results.metric == metric, "value"].item())
                else:
                    print("Warning : No results found for run {}".format(run))
                    print(constraint)

        reference_dict = {
            **constraint.copy(), **evaluation_params.copy()}.copy()

        # Setting to nan the non assigned metrics
        for k, v in result_dict.items():
            if isinstance(v, list):
                if len(v) == 0:
                    result_dict[k] = np.nan

        try:
            overall_res_list.append(pd.DataFrame(result_dict))
        except:
            if np.array([not isinstance(result_dict[k], collections.abc.Sequence) for k in result_dict.keys()]).all():
                overall_res_list.append(pd.DataFrame(result_dict, index=[0]))
            else:
                lens = [(k, len(result_dict[k])) for k in result_dict.keys(
                ) if isinstance(result_dict[k], collections.abc.Sequence)]
                import pdb
                pdb.set_trace()

    overall_res = pd.concat(overall_res_list)
    return overall_res


def convert_results_to_latex(mu_df, std_df,  protected_cols):
    """ Creates a latex table with standard deviations based on a dataframe containing the means and standard deviations of the results.
    protected_col is the column that is not getting concatenated. Only a single column can be protected for now.

    Args:
        mu_df (_type_): data frame containing the means
        std_df (_type_): data frame containing the standard deviations
        protected_col (_type_): the name of the column to protect

    Returns:
        _type_: dataframe with entries as strings with format $mu \pm std$.
    """
    aggregate_cols = [c for c in mu_df.columns if c not in protected_cols]

    df = pd.DataFrame()

    for i in range(len(mu_df)):
        constraint = mu_df.iloc[i][protected_cols].to_dict()
        mu_df_ = mu_df[mu_df[list(constraint.keys())].eq(
            constraint).all(axis=1)]
        std_df_ = std_df[std_df[list(constraint.keys())].eq(
            constraint).all(axis=1)]
        df_ = {
            c: f"${mu_df_[c].item()} \pm {std_df_[c].item()}$" for c in aggregate_cols}
        for k, v in constraint.items():
            df_[k] = v
        df = df.append(pd.DataFrame(df_, index=[0]))
    df = df[protected_cols+aggregate_cols]
    return df


def get_results_event_sweep(sweep_name,
                            constraints,
                            eval_subgroups=False,
                            subgroup_strat='',
                            run_names=None,
                            MM='MM2'):
    """ Collects the results of an event sweep, structured according to the constraints dictionary

    Args:
        sweep_name (str): the unique identifier of the sweep
        constraints (dict): dictionary of the constraints to check for the run to be classified as true
            Example : {"fold":[0,1,2,3,4]}
            or {"fold":[0,1,2,3,4], "model_type":["transformer", "lstm"]} to structure both by fold and by another hyperparameter
    Returns:
        pd.DataFrame: DataFrame with all the results
    """

    log_dir = os.path.join(EXPERIMENT_DIR, "logs")
    exp_dirs = [os.path.join(log_dir, d)
                for d in os.listdir(log_dir) if sweep_name in d]
    if subgroup_strat == 'all':
        eval_subgroups = False

    run_dirs = []
    for exp_dir in exp_dirs:
        run_dirs += [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]

    overall_res_list = []

    keys, values = zip(*constraints.items())
    permutations_dicts = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    for constraint in permutations_dicts:

        res_dict = {"val_loss": [], "test_loss": []}
        runs_fold = [r for r in run_dirs if check_constraints(
            r, constraint, run_names)]
        
        results_dict = get_results_dict_helper(constraint,
                                               runs_fold,
                                               eval_subgroups,
                                               subgroup_strat,
                                               MM=MM)
        overall_res_list.append(pd.DataFrame(results_dict))

    overall_res = pd.concat(overall_res_list)
    return overall_res


def get_times(Y):
    times = []
    quantiles = [0.25, 0.5, 0.75]
    for quant in quantiles:
        times.append(np.quantile(Y.squeeze(), quant))
    return times


def get_subgroup_list(subgroup_strat):
    if subgroup_strat == 'myeloma-type':
        return ['IGA', 'IGG', 'all']
    else:
        raise("Not Implemented")


def get_subgroup_dict(dataset, subgroup_strat='myeloma-type'):
    b_feat_names = dataset.ddata[2]['B_feat_names']
    if subgroup_strat == 'myeloma-type':
        meltype_idx = list(b_feat_names).index('BMELTYPE')
        # not enough support for biclonal, igd, ige, and igm
        subgroup_dict = {
            'IGA': (meltype_idx, '==', 2),
            'IGG': (meltype_idx, '==', 4),
            'all': (meltype_idx, '?', 0),
        }
    elif subgroup_strat == 'all':
        subgroup_dict = {
            'all': (0, '?', 0),
        }
    elif subgroup_strat == 'treatment-response':
        subgroup_dict = {
            'sCR+CR+SD+PD': (0, '==', [1, 2, 5, 6]),
            'VGPR+PR': (0, '==', [3, 4])
            # 'SD+PD': (0, '==', [5,6])
        }
    else:
        raise ValueError('Invalid subgroup strategy given.')

    return subgroup_dict


def compute_event_metrics(model,
                          et_train,
                          et_eval,
                          data_sub,
                          times,
                          results_dict,
                          eval_type='test',
                          **run_yaml):
    eval_score = predict_scores(model, data_sub, None, **run_yaml)
    try:
        cis_eval = concordance_index_ipcw(et_train,et_eval, 1-eval_score)[0]
    except:
        cis_eval = np.nan

    for i, time in enumerate(times):
        eval_score = predict_scores(model, data_sub, time, **run_yaml)
        results_dict[f"roc_auc_{eval_type}"].append(cumulative_dyn_auc(et_train,
                                                                       et_eval, 1-eval_score, time)[0][0])
        try:
            results_dict[f"brs_{eval_type}"].append(float(brier_score(et_train,
                                                                      et_eval, eval_score, time)[1]))
        except:
            results_dict[f"brs_{eval_type}"].append(np.nan)
        results_dict[f"cis_{eval_type}"].append(cis_eval)

    results_dict[f"roc_auc_{eval_type}"].append(
        np.mean(results_dict[f"roc_auc_{eval_type}"]))
    results_dict[f"cis_{eval_type}"].append(cis_eval)
    results_dict[f"brs_{eval_type}"].append(
        np.mean(results_dict[f"brs_{eval_type}"]))
    return results_dict


def compute_results_subgroups(run,
                              run_yaml,
                              model,
                              dataset,
                              results_dict,
                              subgroup_strat,
                              randomize_idx=None,
                              pretrained_model=None,
                              other_mm_dataset=None,
                              iss_only=False):

    train_data = extract_dataset(
        dataset.train_dataloader(),
        t_cond=run_yaml["t_cond"],
        use_traj=run_yaml["use_traj"],
        shift_baseline=run_yaml["shift_baseline"],
        subgroup=None,
        pretrained_model=pretrained_model,
        event_type=run_yaml["event_type"],
        iss_only=iss_only,
        datamodule=dataset
    )

    _, _, _, _, _, _, _, _, _,  _, E, Y, Eae, Yae, _, _, _, _ = train_data
    Y, E = select_event_type(Y, E, Yae, Eae, event_type=run_yaml["event_type"])
    times = get_times(Y)

    # key: subgroup name; val: list of indices
    subgroup_dict = get_subgroup_dict(dataset, subgroup_strat)
    for subgroup_name in subgroup_dict.keys():
        # extract datasets
        subgroup = subgroup_dict[subgroup_name]

        train_data_sub = extract_dataset(
            dataset.train_dataloader(shuffle=False),
            t_cond=run_yaml["t_cond"],
            use_traj=run_yaml["use_traj"],
            shift_baseline=run_yaml["shift_baseline"],
            subgroup=subgroup,
            subgroup_strat=subgroup_strat,
            datamodule=dataset,
            randomize_idx=randomize_idx,
            pretrained_model=pretrained_model,
            event_type=run_yaml["event_type"],
            iss_only=iss_only
        )
        val_data_sub = extract_dataset(
            dataset.val_dataloader(),
            t_cond=run_yaml["t_cond"],
            use_traj=run_yaml["use_traj"],
            shift_baseline=run_yaml["shift_baseline"],
            subgroup=subgroup,
            subgroup_strat=subgroup_strat,
            datamodule=dataset,
            randomize_idx=randomize_idx,
            pretrained_model=pretrained_model,
            event_type=run_yaml["event_type"],
            iss_only=iss_only
        )
        test_data_sub = extract_dataset(
            dataset.test_dataloader(),
            t_cond=run_yaml["t_cond"],
            use_traj=run_yaml["use_traj"],
            shift_baseline=run_yaml["shift_baseline"],
            subgroup=subgroup,
            subgroup_strat=subgroup_strat,
            datamodule=dataset,
            randomize_idx=randomize_idx,
            pretrained_model=pretrained_model,
            event_type=run_yaml["event_type"],
            iss_only=iss_only
        )

        # this is the case when we are evaluating on other than MM2
        if other_mm_dataset is not None:
            other_test_data_sub = extract_dataset(
                other_mm_dataset.test_dataloader(),
                t_cond=run_yaml["t_cond"],
                use_traj=run_yaml["use_traj"],
                shift_baseline=run_yaml["shift_baseline"],
                subgroup=subgroup,
                subgroup_strat=subgroup_strat,
                datamodule=dataset,
                randomize_idx=randomize_idx,
                pretrained_model=pretrained_model,
                event_type=run_yaml["event_type"],
                iss_only=iss_only
            )
            other_test_sub_size = other_test_data_sub[0].shape[0]
        else:
            other_test_sub_size = 0.

        train_sub_size = train_data_sub[0].shape[0]
        val_sub_size = val_data_sub[0].shape[0]
        test_sub_size = test_data_sub[0].shape[0]

        et_test = time_censoring_processing(
            test_data_sub, event_type=run_yaml["event_type"])
        et_val = time_censoring_processing(
            val_data_sub, event_type=run_yaml["event_type"])

        et_train = time_censoring_processing(
            train_data_sub, event_type=run_yaml["event_type"])

        results_dict = compute_event_metrics(model, et_train, et_val, val_data_sub,
                                             times, results_dict, eval_type='val', **run_yaml)
        results_dict = compute_event_metrics(model, et_train, et_test, test_data_sub,
                                             times, results_dict, eval_type='test', **run_yaml)
        if other_mm_dataset is not None:
            et_test_other = time_censoring_processing(
                other_test_data_sub, event_type=run_yaml["event_type"])
            results_dict = compute_event_metrics(model, et_train, et_test_other, other_test_data_sub,
                                                 times, results_dict, eval_type='test_mm1', **run_yaml)
        else:
            results_dict = compute_event_metrics(model, et_train, et_test, test_data_sub,
                                                 times, results_dict, eval_type='test_mm1', **run_yaml)

        for i, time in enumerate(times):
            results_dict["quantile"].append(i)
            results_dict["quantile_value"].append(time)
            results_dict["subgroup_name"].append(subgroup_name)
            results_dict["subgroup_sizes"].append(
                f'train: {train_sub_size}, val: {val_sub_size}, test: {test_sub_size}, test_mm1: {other_test_sub_size}')
            results_dict["run_name"].append(run.split("/")[-1])
            results_dict["t_cond"].append(run_yaml["t_cond"])
        results_dict["quantile"].append("mean")
        results_dict["quantile_value"].append("mean")
        results_dict["subgroup_name"].append(subgroup_name)
        results_dict["subgroup_sizes"].append(
            f'train: {train_sub_size}, val: {val_sub_size}, test: {test_sub_size}, test_mm1: {other_test_sub_size}')
        results_dict["run_name"].append(run.split("/")[-1])
        results_dict["t_cond"].append(run_yaml["t_cond"])

    return results_dict


def evaluate_event_model(run, subgroup_strat, randomize_indices=None):
    with open(os.path.join(run, "hparams.yaml")) as f:
        run_yaml = yaml.load(f, Loader=yaml.Loader)
    run_model_path = os.path.join(run, "model.ckpt")
    model_cls = event_model_conversion(run_yaml["model_type"])
    model = model_cls(**run_yaml)
    model.load(path=run_model_path)

    # get dataset
    data_cls = data_type_class(run_yaml["data_type"])
    dataset = data_cls(**run_yaml)
    dataset.prepare_data()

    results_dict = {}
    results_dict["roc_auc_val"] = []
    results_dict["roc_auc_test"] = []
    results_dict["cis_val"] = []
    results_dict["cis_test"] = []
    results_dict["brs_val"] = []
    results_dict["brs_test"] = []
    results_dict["quantile"] = []
    results_dict["quantile_value"] = []
    results_dict["run_name"] = []
    results_dict["subgroup_name"] = []
    results_dict["t_cond_eval"] = []
    results_dict["subgroup_name"] = []
    results_dict["subgroup_sizes"] = []
    results_dict["t_cond"] = []

    if not isinstance(randomize_indices, list):
        results_dict = compute_results_subgroups(run,
                                                 run_yaml,
                                                 model,
                                                 dataset,
                                                 results_dict,
                                                 subgroup_strat, randomize_idx=randomize_indices)
        return results_dict
    else:
        results_list = []
        for randomize_idx in randomize_indices:
            results_dict = evaluate_event_model(
                run, subgroup_strat, randomize_indices=randomize_idx)
            results_list.append(results_dict)
        return results_list


def get_results_dict_helper(constraint,
                            runs_fold,
                            eval_subgroups=False,
                            subgroup_strat='myeloma-type',
                            MM='MM2'):
    results_dict_keys = [
        "roc_auc_val",
        "roc_auc_test",
        "cis_val",
        "cis_test",
        "brs_val",
        "brs_test",
        "quantile",
        "quantile_value",
        "run_name",
        "subgroup_name",
        "t_cond_eval",
        "pretrained_version"
    ]
    if MM != 'MM2':
        results_dict_keys.append("roc_auc_test_mm1")
        results_dict_keys.append("cis_test_mm1")
        results_dict_keys.append("brs_test_mm1")

    if eval_subgroups:
        results_dict_keys.append("subgroup_name")
        results_dict_keys.append("subgroup_sizes")
    results_dict = constraint.copy()
    for key in results_dict_keys:
        results_dict[key] = []

    for run in runs_fold:
        print(f"Getting results from {run}")
        run_results = pd.read_csv(
            os.path.join(run, "results.csv"))
        if eval_subgroups:
            with open(os.path.join(run, "hparams.yaml")) as f:
                run_yaml = yaml.load(f, Loader=yaml.Loader)
            run_model_path = os.path.join(run, "model.ckpt")
            model_cls = event_model_conversion(run_yaml["model_type"])
            model = model_cls(**run_yaml)
            model.load(path=run_model_path)

            # get dataset
            assert run_yaml["data_type"] == "MM2"
            data_cls = data_type_class(run_yaml["data_type"])
            dataset = data_cls(**run_yaml)
            dataset.prepare_data()

            other_mm_dataset = None
            if MM != "MM2":
                data_cls = data_type_class(MM)
                other_mm_dataset = data_cls(**run_yaml)
                other_mm_dataset.prepare_data()

            results_dict = compute_results_subgroups(run,
                                                     run_yaml,
                                                     model,
                                                     dataset,
                                                     results_dict,
                                                     subgroup_strat,
                                                     other_mm_dataset=other_mm_dataset)
        else:
            if MM != "MM2":
                include_eval_other = True
            else:
                include_eval_other = False
            if "t_cond" in run_results.columns:
                for t_cond in run_results["t_cond"].unique():
                    run_results_ = run_results.loc[run_results["t_cond"] == t_cond]
                    results_dict = extract_results(
                        results_dict, run_results_, run, t_cond=t_cond, include_eval_other=include_eval_other)
            else:
                results_dict = extract_results(
                    results_dict, run_results, run, include_eval_other=include_eval_other)

    return results_dict


def extract_results(results_dict, run_results, run, t_cond=None, include_eval_other=False):
    with open(os.path.join(run, "hparams.yaml")) as f:
        run_yaml = yaml.load(f, Loader=yaml.Loader)
    if "pretrained_version_name" in run_yaml:
        pretrained_version = run_yaml["pretrained_version_name"]
    else:
        pretrained_version = ""
    eval_types = ['val', 'test']
    if include_eval_other:
        eval_types.append('test_mm1')
    for subgroup_name in run_results["subgroup_name"].unique():
        for quantile_idx in run_results["quantile"].unique():
            run_results_df_ = run_results.loc[(run_results["quantile"] ==
                                               quantile_idx) & (run_results["subgroup_name"] == subgroup_name)]
            for eval_type in eval_types:
                results_dict[f"roc_auc_{eval_type}"].append(
                    run_results_df_[f"roc_auc_{eval_type}"].item())
                results_dict[f"cis_{eval_type}"].append(
                    run_results_df_[f"cis_{eval_type}"].item())
                results_dict[f"brs_{eval_type}"].append(
                    run_results_df_[f"brs_{eval_type}"].item())
            results_dict["quantile"].append(quantile_idx)
            results_dict["quantile_value"].append(
                run_results_df_["quantile"].item())
            results_dict["run_name"].append(run.split("/")[-1])
            results_dict["pretrained_version"].append(pretrained_version)
            results_dict["subgroup_name"].append(subgroup_name)

            results_dict["t_cond_eval"].append(t_cond)
        # Avg results
        run_results_avg = run_results.loc[run_results["subgroup_name"]
                                          == subgroup_name]
        for eval_type in eval_types:
            results_dict[f"roc_auc_{eval_type}"].append(
                run_results_avg[f"roc_auc_{eval_type}"].mean())
            results_dict[f"cis_{eval_type}"].append(
                run_results_avg[f"cis_{eval_type}"].mean())
            results_dict[f"brs_{eval_type}"].append(
                run_results_avg[f"brs_{eval_type}"].mean())
        results_dict["quantile"].append("mean")
        results_dict["quantile_value"].append("mean")
        results_dict["run_name"].append(run.split("/")[-1])
        results_dict["pretrained_version"].append(pretrained_version)
        results_dict["subgroup_name"].append(subgroup_name)
        results_dict["t_cond_eval"].append(t_cond)
    return results_dict


def check_wandb_constraints(r, constraints):
    """ Function to check if a specific run satisfies the constraints

    Args:
            wandb run: wandb run to check
            constraints (dict): dictionary of the constraints to check for the run to be classified as true 

    Returns:
            Boolean : wether the particular run respects the constraints or not.
    """

    for k, v in constraints.items():
        if r.config[k] != v:
            return False

    return True


def check_constraints(r, constraints, run_names=None):
    """ Function to check if a specific run satisfies the constraints

    Args:
            r (path): path of the folder of the run to check
            constraints (dict): dictionary of the constraints to check for the run to be classified as true 

    Returns:
            Boolean : wether the particular run respects the constraints or not.
    """
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown)
    # data = yaml.load(os.path.join(r,"hparams.yaml"),Loader= SafeLoaderIgnoreUnknown)
    # import ipdb;  ipdb.set_trace()
    if run_names is not None:
        if not (r.split("/")[-1] in run_names):
            return False

    if not os.path.isdir(r):
        return False

    with open(os.path.join(r, "hparams.yaml")) as f:
        data = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
    for k, v in constraints.items():
        if isinstance(v, list):
            if data[k] not in v:
                return False
        elif data[k] != v:
            return False

    return True


def get_params_run(r):
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(os.path.join(r, "hparams.yaml")) as f:
        data = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
    return data


def get_predictions(experiment_dir, t_cond=None, t_horizon=None, test_fold=True, subgroup_strat='all', treatment_forcing=None, MM1=False, all_data=False):
    """_summary_

    Args:
        experiment_dir (st): Name of the experiment folder
        t_cond (int, optional): Number of time steps to condition on. Defaults to None. - only used in autoregressive mode
        t_horizon : Number of time steps to predict. Defaults to None. - only used in autoregressive mode
        test_fold : if true, uses the test split, if False uses the validation split.
        subgroup_strat : the strategy for subgroup evaluation. If "all", evaluates on all patients. Other possibilities are "myeloma-type".
        all_data : wether to predict on the whole dataset or only on the test set. Defaults to False.
    """

    model_path_name = [path for path in os.listdir(
        experiment_dir) if "ckpt" in path][0]
    checkpoint_path = os.path.join(experiment_dir, model_path_name)

    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(os.path.join(experiment_dir, "hparams.yaml")) as f:
        run_yaml = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)  # yaml.Loader)

    if "finetune_model_type" in run_yaml:  # finetuned model.
        pretrained_model_type = run_yaml['model_type']
        pretrained_checkpoint_path = run_yaml["pretrained_checkpoint_path"]
        pretrained_module_cls = train_module_type_class(pretrained_model_type)
        pretrained_model = pretrained_module_cls.load_from_checkpoint(
            pretrained_checkpoint_path)

        _ = pretrained_model.hparams.pop("outcome", None)
        outcome = run_yaml.get("outcome", "pfs")
        data_cls = data_type_class(pretrained_model.hparams.data_type)
        dataset = data_cls(**pretrained_model.hparams, outcome=outcome)
        dataset.prepare_data()

        # finetune_model_cls = train_module_type_class(
        #    run_yaml["finetune_model_type"]
        finetune_model_cls = model_type_class(run_yaml["finetune_model_type"])
        finetune_module_cls = train_module_type_class(
            run_yaml["finetune_model_type"])
        model = finetune_module_cls.load_from_checkpoint(checkpoint_path,
                                                         finetune_model_cls=finetune_model_cls, pretrained_model=pretrained_model.model)

    else:  # classical model
        model_type = run_yaml['model_type']
        train_module_cls = train_module_type_class(model_type)
        model = train_module_cls.load_from_checkpoint(checkpoint_path)

        data_cls = data_type_class(model.hparams.data_type)
        dataset = data_cls(**model.hparams)
        dataset.prepare_data()

    trainer = pl.Trainer(logger=False, gpus= model.hparams.gpus)
    if t_cond is not None:

        quantiles_event, et_train_event = extract_quantiles(
            dataset, t_cond=t_cond, event_type="pfs")

        quantiles_ae = []
        et_train_ae = []
        for i in range(AE_ALL_DIM):
            quantiles_ae_, et_train_ae_ = extract_quantiles(
                dataset, t_cond=t_cond, event_type=i)
            quantiles_ae.append(quantiles_ae_)
            et_train_ae.append(et_train_ae_)

        model.set_predict_cond_horizon(
            t_cond, t_horizon, quantiles_event, et_train_event, quantiles_ae, et_train_ae)

    if treatment_forcing is not None:
        model.set_treatment_forcing(treatment_forcing)

    if MM1:  # setting dataset after the extract quantiles, etc have been computed
        dataset = TMM1DataModule(**model.hparams)
        dataset.prepare_data()

    if all_data:
        preds = trainer.predict(model, dataset.all_dataloader())
        pids_ = dataset.pids
    else:
        if test_fold:
            preds = trainer.predict(model, dataset.test_dataloader())
            pids_ = dataset.pids[dataset.test_idx]
        else:
            preds = trainer.predict(model, dataset.val_dataloader())
            pids_ = dataset.pids[dataset.val_idx]

    subgroup_dict = get_subgroup_dict(dataset, subgroup_strat=subgroup_strat)
    B = torch.cat([p["B"] for p in preds])
    B_orig = dataset.unnormalize_B(B)

    res_dict = {}
    for subgroup_name, subgroup in subgroup_dict.items():
        # extract datasets
        idx, comp, val = subgroup
        if subgroup_strat == 'treatment-response':
            subgroup_idxs = get_idxs_from_F(pids_, subgroup)
        elif comp == '==':
            subgroup_idxs = np.where(np.abs(B_orig[:, idx]-val) < 1e-2)
        elif comp == '<':
            subgroup_idxs = np.where(B_orig[:, idx] < val)
        elif comp == '>':
            subgroup_idxs = np.where(B_orig[:, idx] > val)
        elif comp == "?":
            subgroup_idxs = np.ones(B_orig.shape[0], dtype=bool)

        forecast_future = torch.cat([pred["forecast_future"] for pred in preds])[
            subgroup_idxs]
        target_future = torch.cat([pred["target_future"]
                                  for pred in preds])[subgroup_idxs]
        mask_future = torch.cat([pred["mask_future"]
                                for pred in preds])[subgroup_idxs]
        X_cond = torch.cat([pred["X_cond"] for pred in preds])[subgroup_idxs]
        M_cond = torch.cat([pred["M_cond"] for pred in preds])[subgroup_idxs]
        times_cond = torch.cat([pred["times_cond"]
                               for pred in preds])[subgroup_idxs]
        times_future = torch.cat([pred["times_future"]
                                 for pred in preds])[subgroup_idxs]
        mse = torch.stack([pred["loss"] for pred in preds]).mean()
        pids = torch.cat([pred["pids"] for pred in preds])[subgroup_idxs]
        treat_flag = torch.cat([pred["treat_flag"]
                               for pred in preds])[subgroup_idxs]
        test_idx = dataset.test_idx

        if preds[0]["y_pred"] is not None:
            y_pred = torch.cat([pred["y_pred"]
                               for pred in preds])[subgroup_idxs]
            y_pred_cond = None
            # y_pred_cond = torch.cat([pred["y_pred_cond"]
            #                         for pred in preds])[subgroup_idxs]
        else:
            y_pred = None
            y_pred_cond = None

        if preds[0]["y_hidden"] is not None:
            y_hidden = torch.cat([pred["y_hidden"]
                                  for pred in preds])[subgroup_idxs]
        else:
            y_hidden = None

        Y = torch.cat([pred["Y"] for pred in preds])[subgroup_idxs]
        E = torch.cat([pred["E"] for pred in preds])[subgroup_idxs]

        Y_ae = torch.cat([pred["Y_ae"] for pred in preds])[subgroup_idxs]
        E_ae = torch.cat([pred["E_ae"] for pred in preds])[subgroup_idxs]

        prediction_idx = dataset.prediction_idx
        prediction_names = dataset.prediction_names

        if y_pred is not None:
            roc_auc_event = model.evaluate_event_auc(
                event_risk=y_pred[:, 0, 0][..., None], Y=Y, E=E, t_cond=t_cond)

            concordance_index_event = model.evaluate_concordance_index(
                event_risk=y_pred[:, 0], Y=Y, E=E, Y_ae=Y_ae, E_ae=E_ae, t_cond=t_cond)  # returns (cis_event, [cis_ae])

            #preds_train = trainer.predict(model, dataset.train_dataloader())
            y_pred_cond_train = None
            # y_pred_cond_train = torch.cat([pred["y_pred_cond"]
            #                               for pred in preds_train])

        else:
            roc_auc_event = None
            y_pred_cond_train = None
            concordance_index_event = None

        mse = model.compute_mse(
            forecast_future, target_future[:, :, prediction_idx], mask_future[:, :, prediction_idx])

        res_dict[subgroup_name] = {
            "mse": mse,
            "forecast_future": forecast_future,
            "target_future": target_future,
            "mask_future": mask_future,
            "X_cond": X_cond,
            "M_cond": M_cond,
            "Y": Y,
            "E": E,
            "Y_ae": Y_ae,
            "E_ae": E_ae,
            "times_cond": times_cond,
            "times_future": times_future,
            "prediction_idx": prediction_idx,
            "prediction_names": prediction_names,
            "pids": pids,
            "test_idx": test_idx,
            "roc_auc_event": roc_auc_event,
            "y_pred": y_pred,
            "y_pred_cond": y_pred_cond,
            "y_pred_cond_train": y_pred_cond_train,
            "t_cond": t_cond,
            "model": model,
            "quantiles_event": quantiles_event,
            "et_train_event": et_train_event,
            "quantiles_ae": quantiles_ae,
            "et_train_ae": et_train_ae,
            "concordance_index_event": concordance_index_event,
            "y_hidden": y_hidden,
            "treat_flag": treat_flag
        }

    return res_dict


def get_idxs_from_F(pids, subgroup):
    F = pd.read_csv('/dbfs/FileStore/takeda-mm/data/F_df')
    subgroup_idxs = []
    _, _, val_list = subgroup
    for idx, pid in enumerate(pids):
        v = F[F['USUBJID'] == pid]['AVAL'].values[0]
        if v in val_list:
            subgroup_idxs.append(idx)
    return subgroup_idxs


if __name__ == "__main__":
    # res_df = get_results_wandb_sweep("z9gwhrwa", constraints={
    #                                 "fold": [0, 1]}, evaluate_dummy=True)
    # print('run 1')
    # df = get_results_sweep(sweep_name="fklbx3pwpo", constraints={"fold": [3]}, evaluation_params={
    #    "t_cond": 5, "t_horizon": 7, "subgroup_strat": "myeloma-type"}, force_recompute=True)

    df = get_ensemble_results(sweep_name="ha8crk9tk2", constraints={"fold": [0, 1, 2]}, evaluation_params={
        "t_cond": 6, "t_horizon": 6, "subgroup_strat": "myeloma-type", "var_bin": None})

    import ipdb
    ipdb.set_trace()

    get_predictions(experiment_dir=os.path.join(EXPERIMENT_DIR, "logs",
                                                "_Seq2Seq_transformer_MM2", "version_26"), t_cond=6, t_horizon=6, test_fold=True, subgroup_strat='all')

    print('run 3')
    res_df = get_results_event_sweep("debug_rsf",
                                     constraints={"fold": [0]},
                                     eval_subgroups=True,
                                     subgroup_strat='treatment-response')
    import ipdb
    ipdb.set_trace()
