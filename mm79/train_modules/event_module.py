import pytorch_lightning as pl
from mm79.train_modules.event_utils import torch_logrank, kp_weighted
import torch
from sklearn.metrics import confusion_matrix
import wandb
from mm79.utils.utils import str2bool
from sklearn.metrics import roc_auc_score


class EventModule(pl.LightningModule):
    def __init__(self, model_cls,  baseline_size, lr, weight_decay,  treatment_strat, g_opt_mode=False, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.g_opt_mode = g_opt_mode

        self.treatment_strat = True
        self.model = model_cls(baseline_size=baseline_size, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(
                    param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(
                f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def forward(self, B, Y, E, T):

        loglik, kl, class_preds = self.model(B, Y, E, T)

        return loglik, kl, class_preds

    def training_step(self, batch, batch_idx):
        B, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, _ = self.parse_batch(
            batch)

        loglik, kl, class_preds = self.forward(
            B=B, Y=Y, E=E, T=T_flag)

        if torch.isnan(class_preds).any():
            import ipdb
            ipdb.set_trace()

        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)

        self.log("train_loss", loss, on_epoch=True)

        # self.log("treatment_effect",
        #         self.model.EventPred.treatment_effect, on_epoch=True)
        return loss

    def get_T_flag(self, T_cond, T_future):
        return (T_cond[..., -1] > 0).any(1)

    def parse_batch(self, batch):
        if len(batch) == 16:
            pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, group = batch
        else:
            pids, B, R, R_mask, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag = batch
            group = None
        return B, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, group

    def compute_loss(self, class_preds, Y, E, T_flag, loglik=None, kl=None):
        if loglik is not None:  # variational approach
            return -(loglik + 0*kl).mean()

        if self.g_opt_mode:
            idx = E[:, 0] == 1
            C = (Y[idx, 0]) * T_flag[idx].float() - \
                Y[idx, 0] * (1-T_flag[idx].float())
            loss = - (class_preds[idx, 1] * C).mean()
            return loss

        if self.treatment_strat:
            # return self.compute_weighted_km_loss(class_preds, Y, E, T_flag)
            return self.compute_loss_treat(class_preds, Y, E, T_flag)
        else:
            return self.compute_group_loss(class_preds, Y, E, T_flag)

    def compute_weighted_km_loss(self, class_preds, Y, E, T_flag):
        return kp_weighted(Y[:, 0], E[:, 0], T_flag, class_preds)

    def compute_group_loss(self, class_preds, Y, E, T_flag):
        """Compute loss for log-rank between two groups.

        Args:
            class_preds (_type_): _description_
            Y (_type_): _description_
            E (_type_): _description_
            T_flag (_type_): _description_

        Returns:
            _type_: _description_
        """
        event_times = torch.cat((Y, Y), dim=0)[:, 0]
        event_observed = torch.cat((E, E), dim=0)[:, 0]
        groups = torch.cat(
            (torch.zeros(Y.shape[0]), torch.ones(Y.shape[0])), dim=0)
        weights = torch.cat((class_preds[:, 0], class_preds[:, 1]), 0)
        logrank_stat = torch_logrank(event_durations=event_times,
                                     event_observed=event_observed, groups=groups, weights=weights)
        loss = -logrank_stat
        return loss

    def compute_loss_treat(self, class_preds, Y, E, T_flag):
        """Computes loss for log-rank between treated and non-treated in group 1.

        Args:
            class_preds (_type_): _description_
            Y (_type_): _description_
            E (_type_): _description_
            T_flag (_type_): _description_
        """
        event_times = Y[:, 0]
        event_observed = E[:, 0]
        #import ipdb
        # ipdb.set_trace()
        groups = T_flag.long()
        weights = class_preds
        logrank_stat = torch_logrank(event_durations=event_times,
                                     event_observed=event_observed, groups=groups, weights=weights)
        loss = -logrank_stat
        return loss

    def validation_step(self, batch, batch_idx):

        B, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, group = self.parse_batch(
            batch)

        loglik, kl, class_preds = self.forward(
            B=B, Y=Y, E=E, T=T_flag)

        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)

        return {"class_preds": class_preds, "group": group, "Y": Y, "E": E, "T_flag": T_flag, "loglik": loglik, "kl": kl}

    def validation_epoch_end(self, outputs):
        class_preds = torch.cat([x["class_preds"] for x in outputs], dim=0)
        if outputs[0]["group"] is not None:
            group = torch.cat([x["group"] for x in outputs], dim=0)
        else:
            group = None
        Y = torch.cat([x["Y"] for x in outputs], dim=0)
        E = torch.cat([x["E"] for x in outputs], dim=0)
        T_flag = torch.cat([x["T_flag"] for x in outputs], dim=0)
        if outputs[0]["loglik"] is not None:
            loglik = torch.cat([x["loglik"] for x in outputs], dim=0)
            kl = torch.cat([x["kl"] for x in outputs], dim=0)
        else:
            loglik = None
            kl = None

        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)
        #neg_group = (~group[:, 0].bool()).long()
        optimal_loss = self.compute_loss(
            torch.nn.functional.one_hot(group[:, 0].long()).float(), Y, E, T_flag, loglik, kl)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_optimal_loss", optimal_loss, on_epoch=True)

        if group is not None:
            roc = roc_auc_score(group[:, 0].cpu(), class_preds[:, 1].cpu())
            self.log("AUC_val", roc, on_epoch=True)
            self.logger.experiment.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                                        y_true=group[:, 0].long().cpu().numpy(), preds=class_preds.max(1)[1].cpu().numpy(),
                                                                                        class_names=None)})
        return

    def test_step(self, batch, batch_idx):

        B, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, group = self.parse_batch(
            batch)

        loglik, kl, class_preds = self.forward(
            B=B, Y=Y, E=E, T=T_flag)
        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)

        return {"class_preds": class_preds, "group": group, "Y": Y, "E": E, "T_flag": T_flag, "loglik": loglik, "kl": kl}

    def test_epoch_end(self, outputs):
        class_preds = torch.cat([x["class_preds"] for x in outputs], dim=0)
        if outputs[0]["group"] is not None:
            group = torch.cat([x["group"] for x in outputs], dim=0)
        else:
            group = None
        Y = torch.cat([x["Y"] for x in outputs], dim=0)
        E = torch.cat([x["E"] for x in outputs], dim=0)
        T_flag = torch.cat([x["T_flag"] for x in outputs], dim=0)

        if outputs[0]["loglik"] is not None:
            loglik = torch.cat([x["loglik"] for x in outputs], dim=0)
            kl = torch.cat([x["kl"] for x in outputs], dim=0)
        else:
            loglik = None
            kl = None

        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)
        self.log("test_loss", loss, on_epoch=True)

        if group is not None:
            roc = roc_auc_score(group[:, 0].cpu(), class_preds[:, 1].cpu())
            self.log("AUC_test", roc, on_epoch=True)

        return

    def predict_step(self, batch, batch_idx):
        B, X_cond, X_future, M_cond, M_future, T_cond, T_future, E, Y, times_cond, times_future, T_flag, group = self.parse_batch(
            batch)
        loglik, kl, class_preds = self.forward(
            B=B, Y=Y, E=E, T=T_flag)
        loss = self.compute_loss(class_preds, Y, E, T_flag, loglik, kl)
        return {"class_preds": class_preds, "group": group, "Y": Y, "E": E, "T_flag": T_flag, "B": B, "loss": loss}

    @classmethod
    def defaults(cls):
        return {
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout_p": 0.0,
            "treatment_strat": False,
        }

    @classmethod
    def add_module_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--lr', type=float, default=cls.defaults()["lr"])
        parser.add_argument('--weight_decay', type=float,
                            default=cls.defaults()["weight_decay"])
        parser.add_argument('--dropout_p', type=float,
                            default=cls.defaults()["dropout_p"])
        return parser
