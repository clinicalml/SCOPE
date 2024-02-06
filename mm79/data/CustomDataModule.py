import numpy as np
import pandas as pd
import pytorch_lightning as pl
import itertools
import os
import sys
import torch

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from mm79.data.process_trial_data import *
from mm79.data.data_utils import pids_to_numeric
from mm79 import EXPERIMENT_DIR
from mm79.utils.utils import str2bool


class CustomDataModule(pl.LightningDataModule):
    """
    Custom Data Module for SCOPE
    """

    def __init__(self,
                 outcome='pfs',
                 t_cond=0,
                 t_horizon=0,
                 num_workers=4,
                 batch_size=64,
                 fold=0,
                 use_rna= False,
                 **kwargs):
        
        """
        use_rna : wethere to use a complementary static variable tensors (eg. RNA seq data)
        """
        super().__init__()

        self.t_cond = t_cond
        self.t_horizon = t_horizon
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.bootstrap_seed = kwargs['bootstrap_seed']

        self.seed = 421 + fold
        self.verbose = False
        
        self.outcome = outcome

        self.use_rna = use_rna

        self.ddata = self.load_data()
        

    def prepare_data(self):
        (tensors, pids, feats, metadata) = self.ddata
        B = torch.Tensor(tensors['B'])
        R = torch.Tensor(tensors["R"])
        R_mask = torch.Tensor(tensors["R_mask"])
        R_processed = torch.Tensor(tensors["R_processed"])
        X = tensors['X']
        M = tensors['M']
        T = tensors['T']
        Y = torch.Tensor(tensors['Y'][..., None])
        E = torch.Tensor(tensors['E'][..., None])

        Yae = torch.Tensor(tensors["Yae"])
        Eae = torch.Tensor(tensors["Eae"])

        self.B_strat = tensors['B_strat']
        self.TRT01P = tensors['TRT01P']

        self.Y = Y
        self.E = E
        self.pids = pids


        self.feats = feats
        self.prediction_idx = tensors["prediction_idx"]
        self.prediction_names = feats["prediction_names"]

        self.input_idx = tensors["input_idx"]
        self.input_names = feats["input_names"]

        if self.use_rna:
            B = torch.cat([B, R_processed], dim=1)
            rna_feats = ["RNA_flag"] + \
                [f"PC_{i}" for i in range(R_processed.shape[-1]-1)]
            self.feats["B_feat_names"] = np.concatenate(
                [self.feats["B_feat_names"], rna_feats])

        t_cond = self.t_cond
        t_horizon = self.t_horizon

        N = B.shape[0]

        if t_cond == -1:  # returning the whole trajectory in X_cond and X_future
            X_cond = torch.Tensor(X[:, :81]).float()
            X_future = torch.Tensor(X[:, :81]).float()
            M_cond = torch.Tensor(M[:, :81]).float()
            M_future = torch.Tensor(M[:, :81]).float()
            T_cond = torch.Tensor(T[:, :81]).float()
            T_future = torch.Tensor(T[:, :81]).float()

            times_cond = torch.Tensor(np.arange(T.shape[1])[
                None, :].repeat(N, 0))
            times_future = torch.Tensor(
                np.arange(T.shape[1])[None, :].repeat(N, 0))
        else:
            X_cond = torch.Tensor(X[:, :t_cond, :]).float()
            X_future = torch.Tensor(X[:, t_cond:t_cond+t_horizon, :]).float()
            M_cond = torch.Tensor(M[:, :t_cond, :]).float()
            M_future = torch.Tensor(M[:, t_cond:t_cond+t_horizon, :]).float()
            T_cond = torch.Tensor(T[:, :t_cond, :]).float()
            T_future = torch.Tensor(T[:, t_cond:t_cond+t_horizon, :]).float()

            times_cond = torch.Tensor(np.arange(t_cond)[None, :].repeat(N, 0))
            times_future = torch.Tensor(
                t_cond + np.arange(t_horizon)[None, :].repeat(N, 0))

        Treat_flag = torch.Tensor((T[:, :, -1] > 0).any(1)[:, None]).float()
        self.Treat_flag = Treat_flag

        pids_numeric = torch.arange(len(B))
        dataset = TensorDataset(pids_numeric, B, R, R_mask, X_cond, X_future, M_cond, M_future,
                                T_cond, T_future, E, Y, Eae, Yae, times_cond, times_future, Treat_flag)

        train_idx, val_idx, test_idx = self.split_dataset(pids_numeric)

        if self.bootstrap_seed != -1:
            rng = np.random.default_rng(self.bootstrap_seed + 420)
            train_idx = rng.choice(
                train_idx, size=int(1.5*len(train_idx)))

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.test_dataset = Subset(dataset, test_idx)
        self.dataset = dataset

        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        self.baseline_size = B.shape[-1]
        self.input_long_size = X.shape[-1]
        self.treatment_size = T.shape[-1]

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.pids = pids_numeric

    def split_dataset(self, idx):
        
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=421)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.25, random_state=self.seed)
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def load_data(self):

        """
        Load the data from disk. In this example, we just generate random data.
        """

        np.random.seed(self.seed)

        #Baseline variables
        B = np.random.randn(703, 42)

        # RNA seq data
        R = np.random.randn(703, 22214)

        #RNA seq mak
        R_mask = np.random.randint(low=0, high=2, size=(703, 22214))

        #Processed RNA seq data
        R_processed = np.random.randn(703, 21)

        #Longitudinal variables (Npatients, Time, Ndim)
        X = np.random.randn(703, 81, 39)
        #Longitudinal mask (Npatients, Time, Ndim)
        M = np.random.randint(low=0, high=2,  size=(703, 81, 39))
        # Treatment data (Npatients, Time, Ndim)
        T = np.random.randn(703, 1, 3).repeat(81, axis=1)
        # Binary treatment indicator
        T_ove = np.random.randn(703, 99)

        # Event data 
        Y = 2*np.abs(B[:, 0]) + 20
        E = np.random.randint(low=0, high=2, size=(703,))

        # Event data for adverse events
        Yae = (2*np.abs(B[:, 0]) + 20)[:, None].repeat(12, axis=1)
        Eae = np.random.randint(low=0, high=2, size=(703, 12))


        # Names of the longitudinal features
        prediction_idx = np.arange(X.shape[-1])
        feats = {"prediction_names": ['Albumin (g/L)', 'Alkaline Phosphatase (U/L)',
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
                                          'Leukocytes (10^9/L)']}

        
        input_idx = np.arange(X.shape[-1])
        feats["input_names"] = np.arange(X.shape[-1])

        #Names of different features
        feats['B_feat_names'] = np.array(['AGE', 'DIAGFXMO', 'BASPPLAS', 'BFLCINV', 'BCREAT', 'BCRCL',
                                          'BALB', 'BMPLSCE', 'POLYMPHN', 'POLYCCN', 'POLYCGN', 'POLYGGN',
                                          'POLYCGGN', 'RACE', 'BCRLTM', 'BPLASMCY', 'HISBONE', 'EXMEDBS',
                                          'MEADISFL', 'LIVERALL', 'RISSENT', 'BMELTYPE', 'BLCHAIN',
                                          'DURSALH', 'LYTICBH', 'EXTRAMDH', 'ISSINIT', 'ISSENT', 'ISSENT2',
                                          'BECOG', 'BB2MGCAT', 'BFLCRCAT', 'BCYABCAT', 'BSKERSLT', 'BLYTICB',
                                          'ERFL1', 'ERFL2', 'ERFL3', 'ERFL4', 'BCYABCA2', 'SEXN', 'TRT01AN'])

        feats["X_feat_names"] = np.array(['Albumin (g/L)', 'Alkaline Phosphatase (U/L)',
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
                                          'Leukocytes (10^9/L)'])

        feats["T_feat_names"] = np.array(['DEX', 'LEN', 'MLN'])

        feats["AE_feat_names"] = np.array(['ACUTE RENAL FAILURE', 'CARDIAC ARRHYTHMIAS', 'DIARRHOEA',
                                           'ENCEPHALOPATHY', 'ENCEPHALOPATHY; LIVER IMPAIRMENT',
                                           'HEART FAILURE', 'HYPOTENSION', 'LIVER IMPAIRMENT',
                                           'MYOCARDIAL INFARCTION', 'NAUSEA', 'NEUTROPENIA',
                                           'PERIPHERAL NEUROPATHIES', 'RASH', 'THROMBOCYTOPENIA', 'VOMITING'])

        B[:, feats['B_feat_names'] == "BMELTYPE"] = np.random.choice(
            [0, 2, 4], B.shape[0])[:, None]
        
        B_strat = None
        TRT01P = None

        tensors = {
            'B': B, 'R': R, 'R_mask': R_mask, 'R_processed': R_processed, 'X': X, 'M': M, 'T': T, 'T_ove': T_ove, 'Y': Y, 'E': E, "prediction_idx": prediction_idx, "input_idx": input_idx, "Yae": Yae, "Eae": Eae,
            'TRT01P': TRT01P, 'B_strat': B_strat
        }

        #meta_data = pd.read_csv(os.path.join(
        #    EXPERIMENT_DIR, "..", "mm79", "data", "fake_metadata.csv"), sep=";")
        #pids = meta_data.SUBJID.values

        self.B_norm_stds = np.ones((1, B.shape[1]))
        self.B_norm_means = np.zeros((1, B.shape[1]))

        return (tensors, None, feats, None)


    def get_datasets(self):
        return self.ddata
    
    def unnormalize_B(self,B):
        return B

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False
        )

    def all_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False
        )

    @classmethod
    def defaults(cls):
        return {"batch_size": 64,
                "t_cond": 6,
                "t_horizon": 6,
                "num_workers": 4,
                "bootstrap_seed": -1,
                "use_rna": False,
                "outcome": "pfs",
                }

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int,
                            default=cls.defaults()["batch_size"])
        parser.add_argument('--t_cond', type=int,
                            default=cls.defaults()["t_cond"])
        parser.add_argument('--t_horizon', type=int,
                            default=cls.defaults()["t_horizon"])
        parser.add_argument('--num_workers', type=int,
                            default=cls.defaults()["num_workers"])
        parser.add_argument('--bootstrap_seed', type=int,
                            default=cls.defaults()["bootstrap_seed"])
        parser.add_argument('--use_rna', type=str2bool,
                            default=cls.defaults()["use_rna"])
        parser.add_argument('--outcome', type=str,
                            default=cls.defaults()["outcome"])
        return parser

if __name__=="__main__":
    dataset = CustomDataModule(t_cond=-1, fold=0, bootstrap_seed= -1, use_rna=False)
    dataset.prepare_data()