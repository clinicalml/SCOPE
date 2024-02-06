import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from argparse import ArgumentParser
from mm79.data.TMMDataModule import TMMDataModule
from mm79.data.randomDataModule import RandomDataModule
from mm79.data.syntheticDataModule import SyntheticDataModule
from mm79.models.transformer import Treatformer
from mm79.utils.utils import str2bool
import pandas as pd
import os
from mm79.train_modules.train_module import TrainModule
from mm79.train_modules.train_module_dmm import DMMTrainModule
from mm79.train_modules.utils import model_type_class, data_type_class, train_module_type_class
from mm79.train_modules.plot_utils import plot_predictions
from mm79.train_modules.event_utils import extract_quantiles
from mm79.utils.utils import logging_folder
from mm79 import AE_ALL_DIM


def get_logger(data_cls, sweep_name="", force_wandb=False):
    if ((data_cls == SyntheticDataModule) or (force_wandb)):
        # set up wandb
        exp_name = f'{sweep_name}_Seq2Seq_{args.model_type}_{args.data_type}'
        logger = WandbLogger(
            name=exp_name,
            project='mm-long',
            entity=args.wandb_user,
            log_model=False
        )
        log_dir = logger.experiment.dir
    else:
        exp_name = f'{sweep_name}_Seq2Seq_{args.model_type}_{args.data_type}'
        logger = CSVLogger(
            logging_folder(), name=exp_name)
        log_dir = logger.log_dir
    return logger, log_dir, exp_name


def main(model_cls, data_cls, train_module_cls, args):

    # prepare dataset and load in dimensions
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    quantiles_event, et_train_event = extract_quantiles(
        dataset, t_cond=args.t_cond, event_type="pfs")

    quantiles_ae = []
    et_train_ae = []
    for i in range(AE_ALL_DIM):
        quantiles_ae_, et_train_ae_ = extract_quantiles(
            dataset, t_cond=args.t_cond, event_type=i)
        quantiles_ae.append(quantiles_ae_)
        et_train_ae.append(et_train_ae_)

    # init model
    model = train_module_cls(model_cls=model_cls, baseline_size=dataset.baseline_size,
                             input_long_size=dataset.input_long_size,
                             treatment_size=dataset.treatment_size,
                             prediction_idx=dataset.prediction_idx, prediction_names=dataset.prediction_names,
                             input_idx=dataset.input_idx, input_names=dataset.input_names, quantiles_event=quantiles_event,
                             et_train_event=et_train_event, quantiles_ae=quantiles_ae, et_train_ae=et_train_ae,
                             **vars(args))

    logger, log_dir, exp_name = get_logger(
        data_cls, sweep_name=args.sweep_name, force_wandb=args.force_wandb)
    # checkpointing
    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', patience=args.early_stopping)

    # model training
    trainer = pl.Trainer(gpus=args.gpus,
                         logger=logger,
                         callbacks=[checkpoint_cb, early_stopping_cb],
                         max_epochs=args.max_epochs)
    trainer.fit(model, datamodule=dataset)

    # inference
    checkpoint_path = checkpoint_cb.best_model_path
    test_trainer = pl.Trainer(logger=False)
    model = train_module_cls.load_from_checkpoint(checkpoint_path)
    val_results = test_trainer.test(
        model, dataloaders=dataset.val_dataloader())[0]
    val_results = {
        name.replace('test', 'val'): value for name, value in val_results.items()
    }

    test_results = test_trainer.test(
        model, dataloaders=dataset.test_dataloader())[0]

    # logging
    if isinstance(logger, WandbLogger):
        for name, value in {**test_results}.items():
            logger.experiment.summary['restored_' + name] = value
        for name, value in {**val_results}.items():
            logger.experiment.summary['restored_' + name] = value
    else:
        res = pd.Series(test_results).append(pd.Series(val_results))
        df = res.to_frame(name="value").reset_index(names=["metric"])
        df.to_csv(os.path.join(log_dir, 'recovered_results.csv'), index=False)
        if args.t_cond == -1:
            pass
        else:
            pass
            # plot_predictions(exp_name)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--fold', default=0, type=int,
                        help=' fold number to use')
    parser.add_argument('--wandb_user', default='edebrouwer',
                        type=str, help='wandb username')
    parser.add_argument('--gpus', default=1, type=int,
                        help='the number of gpus to use to train the model')
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--model_type', type=str, default='transformer')
    parser.add_argument('--data_type', type=str, default="Synthetic")
    parser.add_argument('--sweep_name', type=str, default="",
                        help="this is just a prefix that will be added for saving")
    parser.add_argument('--early_stopping', type=int, default=50,
                        help="Number of epochs without improvements before stopping the run")
    parser.add_argument('--restricted_input_features_set',
                        type=str2bool, default=False, help="whether to use a restricted set of input features")
    parser.add_argument('--restricted_pred_features_set',
                        type=str2bool, default=True, help="whether to predict a resticted number of features")
    parser.add_argument('--planned_treatments',
                        type=str2bool, default=True, help="wether future treatments are available for prediction")
    parser.add_argument('--emission_proba',
                        type=str2bool, default=False, help = "if True, the model also outputs probabilities of event (bernoulli process)")
    parser.add_argument('--force_wandb',
                        type=str2bool, default=False)
    parser.add_argument('--event_type', type=str, default="pfs",
                        help="Event to use for model. pfs or a digit for the index of the AE to use")

    partial_args, _ = parser.parse_known_args()

    # get model type, data type, and train_module type
    model_cls = model_type_class(partial_args.model_type)
    data_cls = data_type_class(partial_args.data_type)
    train_module_cls = train_module_type_class(partial_args.model_type)

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    parser = train_module_cls.add_module_specific_args(parser)
    args = parser.parse_args()
    #   if args.T_mask == -1:
    #       args.T_mask = args.T_cond
    main(model_cls, data_cls, train_module_cls, args)
