from mm79.train_modules.event_module import EventModule
from mm79.train_modules.utils import data_type_class
import os
import pytorch_lightning as pl
import torch


def evaluate_wandb_run(run):
    """_summary_

    Args:
        run (_type_): the run name (wandb tag)
        evaluate_dummy (bool, optional): whether to evaluate the dummy model as well. Defaults to False.
        t_cond (_type_, optional): Only used when in autoregressive mode. The conditioning window 
        t_horizon (_type_, optional): Only used when in autoregressive mode. The forecasting window

    Returns:
        _type_: prediction dictionary
    """

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = EventModule.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = dict(model.hparams)

    data_cls = data_type_class(model.hparams.data_type)
    dataset = data_cls(**hparams)
    dataset.prepare_data()
    trainer = pl.Trainer(logger=False, gpus=1)

    preds = trainer.predict(model, dataset.test_dataloader())

    class_preds = torch.cat([p["class_preds"] for p in preds])
    group = torch.cat([p["group"] for p in preds])
    Y = torch.cat([p["Y"] for p in preds])
    E = torch.cat([p["E"] for p in preds])
    T_flag = torch.cat([p["T_flag"] for p in preds])
    B = torch.cat([p["B"] for p in preds])

    pred_dict = {"class_preds": class_preds,
                 "group": group, "Y": Y, "E": E, "T_flag": T_flag, "B": B}

    return pred_dict
