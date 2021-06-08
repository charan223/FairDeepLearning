from argparse import Namespace
import numpy as np

from metrics.Accuracy import Accuracy
from metrics.BCR import BCR
from metrics.CalibrationNeg import CalibrationNeg
from metrics.CalibrationPos import CalibrationPos
from metrics.CV import CV
from metrics.DIAvgAll import DIAvgAll
from metrics.DIBinary import DIBinary
from metrics.EqOppo_fn_diff import EqOppo_fn_diff
from metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from metrics.EqOppo_fp_diff import EqOppo_fp_diff
from metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from metrics.FNR import FNR
from metrics.FPR import FPR
from metrics.MCC import MCC
from metrics.SensitiveMetric import SensitiveMetric
from metrics.TNR import TNR
from metrics.TPR import TPR

ACC_METRICS = [
    Accuracy(),
    TPR(),
    TNR(),
    BCR(),
    MCC(),
]

FAIRNESS_METRICS = [
    DIBinary(),
    DIAvgAll(),
    CV(),
    SensitiveMetric(Accuracy),
    SensitiveMetric(TPR),
    SensitiveMetric(TNR),
    SensitiveMetric(FPR),
    SensitiveMetric(FNR),
    SensitiveMetric(CalibrationPos),
    SensitiveMetric(CalibrationNeg),
    EqOppo_fn_diff(),  # TODO: ~4.9k?
    EqOppo_fn_ratio(),
    EqOppo_fp_diff(),  # TODO: ~40?
    EqOppo_fp_ratio(),
]

METRICS = ACC_METRICS + FAIRNESS_METRICS


def add_metric(metric):
    METRICS.append(metric)


def get_metrics(privileged_vals, sensitive_dict):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(privileged_vals, sensitive_dict)
    return metrics


def calculate_metrics(input_dict):
    """
    example: ricci dataset
    Race  Class  other_attrs
       W      1            1
       B      1            0
       H      0            1
    single_sensitive = 'Race'
    all_sensitive_attributes = ['Race']
    sensitive_dict = {'Race': [0, 1]} from dataset -> can define new metrics
    sensitive_dict = {'Race': ['H', 'B', 'W']}
    dict_sensitive_lists = {'Race': [1, 0, 0, 1, 0, 1, 0, 1,...]}
    privileged_vals or unprotected_vals = ['W'] or [1]
    """
    inps = Namespace(**input_dict)
    results = dict()
    for mtr in get_metrics(inps.privileged_vals, inps.sensitive_dict):
        mtr_value = mtr.calc(
            inps.label,
            inps.predicted,
            {"sensitiveattr": inps.sensitiveattr},
            "sensitiveattr",
            [inps.privileged_vals["sensitiveattr"]],  # TODO: this is messy! Fix please.
            inps.positive_pred,
        )
        results[mtr.get_name()] = mtr_value
    return results


class CostTemplate:
    def is_better_than(self, val1, val2):
        return val1 < val2
