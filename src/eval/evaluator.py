from cmath import nan
import os
import pickle
from typing import List
import numpy as np
import pandas
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,  roc_curve
import matplotlib.pyplot as plt
from datetime import datetime

from ysdc_dataset_api.evaluation.metrics import compute_all_aggregator_metrics
from sdc.analyze_metadata import filter_top_d_plans, calc_uncertainty_regection_curve
from src.eval.plot_retention_curves import get_sparsification_factor, plot_retention_curve_with_baselines
from src.metrics.metrics import log_likelihood


class Evaluator(object):

    def __init__(self) -> None:

        self.eval_path = None

        self.D = 5

        self.trajectory_prediction_metrics = dict()
        self.trajectory_prediction_metrics["minADE"] = []
        self.trajectory_prediction_metrics["top1ADE"] = []
        self.trajectory_prediction_metrics["avgADE"] = []
        self.trajectory_prediction_metrics["weightedADE"] = []
        self.trajectory_prediction_metrics["minFDE"] = []
        self.trajectory_prediction_metrics["top1FDE"] = []
        self.trajectory_prediction_metrics["avgFDE"] = []
        self.trajectory_prediction_metrics["weightedFDE"] = []
        self.trajectory_prediction_metrics["NLL"] = []
        
        # trajectory prediction metrics
        self.trajectory_metrics_full = dict()
        self.trajectory_metrics_id = dict()
        self.trajectory_metrics_ood = dict()

        # ood detection metrics
        self.alpha = []  # OOD label
        self.alpha_hat = []  # OOD score
        self.ood_detection_metrics = dict()

        # uncertainty estimation metrics
        self.e_hat = [] # uncertainty
        self.uncertainty_metrics_full = dict()
        self.uncertainty_metrics_id = dict()
        self.uncertainty_metrics_ood = dict()
        self.retention_data_full = dict()
        self.retention_data_id = dict()
        self.retention_data_ood = dict()     

        self.black_list = ["NLL"]

        # eval summary
        self.eval_summary = ""

        # paths
        self.eval_path = None


    def set_eval_path_from(self, exp_path, kwargs=None):
        eval_dir = "eval_"
        eval_dir += datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
        if kwargs:
            for k, v in kwargs.items():
                eval_dir += f"_{k}_{v}"

        self.eval_path = os.path.join(exp_path, eval_dir)

        os.makedirs(self.eval_path, exist_ok=True)

        print("Saving evaluation results in {}...".format(self.eval_path))


    def log_metrics(self, metrics_dict) -> None:
        for mk, ms in metrics_dict.items():
            for m in ms:
                self.trajectory_prediction_metrics[mk].append(m)


    def log_ood(self, alpha: np.ndarray) -> None:
        self.alpha.extend(alpha)


    def log_ood_score(self, alpha_hat: np.ndarray) -> None:
        self.alpha_hat.extend(alpha_hat)


    def log_e_hat(self, e_hat: np.ndarray) -> None:
        self.e_hat.extend(e_hat)


    def compute_trajectory_metrics(self) -> None:
        """
        Compute the trajectoy prediction metrics for the
        FULL, ID and OOD (sub)sets.
        
        """

        ood = np.array(self.alpha)
        
        # full
        for k, m in self.trajectory_prediction_metrics.items():
            self.trajectory_metrics_full[k] = np.mean(m)   

        # id
        for k, m in self.trajectory_prediction_metrics.items():
            self.trajectory_metrics_id[k] = np.mean(np.array(m)[np.invert(ood)])

        # ood
        for k, m in self.trajectory_prediction_metrics.items():
            self.trajectory_metrics_ood[k] = np.mean(np.array(m)[ood])


    def compute_uncertainty_metrics(self, export=False) -> None:

        ood = np.array(self.alpha)

        # full
        for k, m in self.trajectory_prediction_metrics.items():
            if not np.isnan(m).any() and k not in self.black_list:
                retention_curve = calc_uncertainty_regection_curve(np.array(m), uncertainty=np.array(self.e_hat))
                auc = retention_curve.mean()
                self.uncertainty_metrics_full[f"{k} R-AUC"] = auc
                self.retention_data_full[f"{k}_retention"] = retention_curve

        if not np.isnan(m).any():

            # id
            for k, m in self.trajectory_prediction_metrics.items():
                if k not in self.black_list:
                    retention_values = calc_uncertainty_regection_curve(np.array(m)[np.invert(ood)], uncertainty=np.array(self.e_hat)[np.invert(ood)])
                    auc = retention_values.mean()
                    self.uncertainty_metrics_id[f"{k} R-AUC"] = auc
                    # retention data
                    sparsification_factor = get_sparsification_factor(retention_values.shape[0])
                    retention_values = retention_values[::sparsification_factor][::-1]
                    retention_thresholds = np.arange(len(retention_values)) / len(retention_values)
                    self.retention_data_id[f"{k} R-AUC"] = auc
                    self.retention_data_id[f"{k}_retention"] = retention_values
                    self.retention_data_id[f"{k}_retention_thresholds"] = retention_thresholds
                    # oracle
                    retention_values = calc_uncertainty_regection_curve(errors=np.array(m), uncertainty=np.array(m))
                    auc = retention_values.mean()
                    sparsification_factor = get_sparsification_factor(retention_values.shape[0])
                    retention_values = retention_values[::sparsification_factor][::-1]
                    retention_thresholds = np.arange(len(retention_values)) / len(retention_values)
                    self.retention_data_full[f"Oracle {k} R-AUC"] = auc
                    self.retention_data_full[f"Oracle {k}_retention"] = retention_values
                    self.retention_data_full[f"Oracle {k}_retention_thresholds"] = retention_thresholds
                else:
                    self.uncertainty_metrics_id[f"{k} R-AUC"] = nan
                    self.retention_data_id[f"{k} R-AUC"] = auc
                    self.retention_data_id[f"{k}_retention"] = nan
                    self.retention_data_id[f"{k}_retention_thresholds"] = nan
                    self.retention_data_full[f"Oracle {k} R-AUC"] = nan
                    self.retention_data_full[f"Oracle {k}_retention"] = nan
                    self.retention_data_full[f"Oracle {k}_retention_thresholds"] = nan

            # ood
            for k, m in self.trajectory_prediction_metrics.items():
                if sum(ood) > 0 and k not in self.black_list:
                    retention_values = calc_uncertainty_regection_curve(np.array(m)[ood], uncertainty=np.array(self.e_hat)[ood])
                    auc = retention_values.mean()
                    self.uncertainty_metrics_ood[f"{k} R-AUC"] = auc
                    # retention data
                    sparsification_factor = get_sparsification_factor(retention_values.shape[0])
                    retention_values = retention_values[::sparsification_factor][::-1]
                    retention_thresholds = np.arange(len(retention_values)) / len(retention_values)
                    self.retention_data_ood[f"{k} R-AUC"] = auc
                    self.retention_data_ood[f"{k}_retention"] = retention_values
                    self.retention_data_ood[f"{k}_retention_thresholds"] = retention_thresholds
                    # oracle
                    retention_values = calc_uncertainty_regection_curve(errors=np.array(m), uncertainty=np.array(m))
                    auc = retention_values.mean()
                    sparsification_factor = get_sparsification_factor(retention_values.shape[0])
                    retention_values = retention_values[::sparsification_factor][::-1]
                    retention_thresholds = np.arange(len(retention_values)) / len(retention_values)
                    self.retention_data_full[f"Oracle {k} R-AUC"] = auc
                    self.retention_data_full[f"Oracle {k}_retention"] = retention_values
                    self.retention_data_full[f"Oracle {k}_retention_thresholds"] = retention_thresholds
                else:
                    self.uncertainty_metrics_ood[f"{k} R-AUC"] = nan
                    self.retention_data_ood[f"{k} R-AUC"] = auc
                    self.retention_data_ood[f"{k}_retention"] = nan
                    self.retention_data_ood[f"{k}_retention_thresholds"] = nan
                    self.retention_data_full[f"Oracle {k} R-AUC"] = nan
                    self.retention_data_full[f"Oracle {k}_retention"] = nan
                    self.retention_data_full[f"Oracle {k}_retention_thresholds"] = nan
        
        # save retention results
        retention_results = dict()
        retention_results["full"] = self.retention_data_full
        retention_results["id"] = self.retention_data_id
        retention_results["ood"] = self.retention_data_ood
        if export:
            with open(os.path.join(self.eval_path, "retention_data.pkl"), 'wb') as f:
                pickle.dump(retention_results, f)


    def compute_ood_metrics(self) -> None:
        """
        Compute the OOD detection metrics and plot the ROC curve.

        """
        if sum(self.alpha) > 0:
            self.ood_detection_metrics["AUROC"] = roc_auc_score(y_true=np.array(self.alpha), y_score=np.array(self.alpha_hat)) *100
            fpr, tpr, thresholds = roc_curve(y_true=np.array(self.alpha), y_score=np.array(self.alpha_hat), pos_label=1, drop_intermediate=False)
            # roc curve
            fig, ax = plt.subplots(1)
            plt.plot(fpr, tpr, label="AUC = "+"{:6.2f}".format(self.ood_detection_metrics["AUROC"]), color="#36337a")
            plt.legend()
            plt.grid()
            plt.title("ROC curve")
            plt.xlabel("False positive rate [-]")
            plt.ylabel("True positive rate [-]")
            plt.savefig(os.path.join(self.eval_path, "roc_curve.png"))
            plt.close()
            roc_dict = dict()
            roc_dict["fpr"] = fpr
            roc_dict["tpr"] = tpr
            roc_dict["thresholds"] = thresholds
            df = pandas.DataFrame(roc_dict)
            df.to_csv(os.path.join(self.eval_path, "roc_data.csv"))
        else:
            self.ood_detection_metrics["AUROC"] = nan


    def plot_retention_curves(self) -> None:

        print("Plotting retention curves...")

        # plot retention cuves with baselines
        e_hat = np.array(self.e_hat)
        assert os.path.isdir(self.eval_path)

        for mk, mv in self.trajectory_prediction_metrics.items():

            if mk not in self.black_list:

                if not np.isnan(mv).any():

                    mv = np.array(mv)

                    fig = plot_retention_curve_with_baselines(e_hat, mv, metric_name=mk, group_by_uncertainty=True)
                    
                    plt.savefig(os.path.join(self.eval_path, f"retention_curve_{mk}.png"), dpi=300)     
                    plt.close(fig)   


    def select_top_d_trajectories(self, predictions: np.ndarray, confidences: np.ndarray):
        """
        Select the top D trajectories based on the predicted confidcenes
        Args:
            predictions: Predicted trajectories [N, K, T, 2]
            confidences: Predicted mode confidences [N, K]
        Return:
            top_predictions: Top D predictions.
            top_confidences: Top D confidences.
        """
        top_predictions, top_confidences = filter_top_d_plans(predictions, confidences, d=self.D)
        return top_predictions, top_confidences


    def compute_nll(
            self, 
            y: np.ndarray, 
            y_hat: np.ndarray, 
            pi_prob: np.ndarray, 
            sigma: np.ndarray,
            ) -> List[float]:
        """
        Compute the negative log-likelihood metric.
        """
        nll = []
        for i in range(len(y_hat)):
            nll.append(-log_likelihood(y[i], y_hat[i], pi_prob[i], sigma[i]))

        return nll

    
    def compute_batch_metrics(
            self, y: np.ndarray, 
            y_hat: np.ndarray, 
            pi: np.ndarray, 
            sigma: np.ndarray, 
            alpha: np.ndarray, 
            alpha_hat: np.ndarray, 
            e_hat: np.ndarray,
            ) -> None:
        """
        Compute the trajectory prediction metrics and 
        log the out-of-distribution score and the uncertainty.
        
        Args:
            y: Ground truth trajectory [N, T, 2]
            y_hat: Predicted mean trajectory [N, K, T, 2]
            pi: Confidence scores [N, K]
            sigma: Standard deviation of the symmetric bi-variate Gaussian [N, K, T]
            alpha: OOD ground truth [N]
            alpha_hat: OOD score [N]
            e_hat: Uncertainty [N]
        """
        # comput trajectory prediction metrics ([ADE, FDE] x [min, avg, top1, weighted])
        metrics = compute_all_aggregator_metrics(pi, y_hat, y)

        # compute the negative-log-likelihood metric
        pi_prob = softmax(pi, axis=1) # convert raw confidences to probabilities
        metrics["NLL"] = self.compute_nll(y, y_hat, pi_prob, sigma)

        # log metrics
        self.log_metrics(metrics)

        # log ood ground truth
        self.log_ood(alpha)

        # log ood score
        self.log_ood_score(alpha_hat)
  
        # log uncertainty
        self.log_e_hat(e_hat)


    def print_evaluation_summary(self, write_to_file=True):

        # trajectory prediction metrics
        mfull = self.trajectory_metrics_full
        mid = self.trajectory_metrics_id
        mood = self.trajectory_metrics_ood

        tpm_str = "\n"
        tpm_str += "--- Trajectory Prediction Metrics ---\n"
        tpm_str += "\n{:>20} \t{:10}\t{:10}\t{:10}".format("", "- ID -", "- OOD -", "- FULL -")

        for k, v in mfull.items():
            if not np.isnan(v).any():
                tpm_str += "\n{:>20}:\t{:10.5f}\t{:10.5f}\t{:10.5f}".format(k, mid[k], mood[k], mfull[k])
            else:
                tpm_str += "\n{:>20}:\t{:10}\t{:10}\t{:10}".format(k, "nan", "nan", "nan")

        # ood detection metrics
        oodm_str = "\n"
        oodm_str += "--- OOD Detection Metrics ---\n"
        for k, v in self.ood_detection_metrics.items():
            oodm_str += "\n{:>20}:\t{:10}\t{:10}\t{:10.5f}".format(k, "", "", v)

        # uncertainty estimation metrics
        mfull = self.uncertainty_metrics_full 
        mid = self.uncertainty_metrics_id
        mood = self.uncertainty_metrics_ood

        um_str = "\n"
        um_str += "--- Uncertainty Estimation Metrics ---\n"
        um_str += "\n{:>20} \t{:10}\t{:10}\t{:10}".format("", "- ID -", "- OOD -", "- FULL -")
        for k, v in mfull.items():
            if not np.isnan(v).any():
                um_str += "\n{:>20}:\t{:10.5f}\t{:10.5f}\t{:10.5f}".format(k, mid[k], mood[k], mfull[k])
            else:
                um_str += "\n{:>20}:\t{:10}\t{:10}\t{:10}".format(k, "nan", "nan", "nan")

        # print all results
        print(tpm_str)
        print(oodm_str)
        print(um_str)
        
        # write to file
        if write_to_file:

            self.export_txt_path = os.path.join(self.eval_path, "results.txt")

            f = open(self.export_txt_path, 'w')
            f.write(tpm_str + "\n")
            f.write(oodm_str + "\n")
            f.write(um_str + "\n")
            f.close()


    def full_eval(self):
        """
        Perform full evaluation on the logged data and metrics.
        Plot the results and write the metrics to files. 
        """

        self.compute_trajectory_metrics()
        self.compute_ood_metrics()
        self.compute_uncertainty_metrics(export=True)
        self.plot_retention_curves()
        self.print_evaluation_summary()
