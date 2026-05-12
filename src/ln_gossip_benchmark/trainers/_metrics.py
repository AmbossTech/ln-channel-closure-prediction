import logging
from typing import Optional, Sequence

import numpy as np
import wandb
from sklearn.metrics import classification_report

from ln_gossip_benchmark.utils.wandb import report_to_wandb_log

log = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    metric_prefix: str,
    target_names: Optional[Sequence[str]] = None,
    log_confusion_matrix: bool = False,
) -> dict:
    """Build a flat metrics dict from sklearn's classification_report.

    Optionally appends a wandb confusion-matrix chart under `<prefix>/confusion_matrix`.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = report_to_wandb_log(report, metric_prefix)

    if log_confusion_matrix and target_names:
        try:
            metrics[f"{metric_prefix}/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=list(target_names),
            )
        except Exception as e:
            log.warning(f"Could not log confusion matrix: {e}")

    return metrics


def format_step_log(
    step_metrics: dict,
    global_step: int,
    epoch: int,
    extra_metrics: Optional[dict] = None,
) -> dict:
    """Build the per-step log payload that trainers send to wandb."""
    payload = {
        "train_step/loss": step_metrics["loss"],
        "train_step/date": step_metrics["date"],
        "train_step/time_ms": step_metrics["time_ms"],
        "train_step/num_open_edges": step_metrics["num_open_edges"],
        "train_step/num_active_nodes": step_metrics["num_active_nodes"],
        "train_step/num_opening_batch": step_metrics["num_opening_batch"],
        "train_step/num_closing_batch": step_metrics["num_closing_batch"],
        "epoch": epoch,
        "global_step": global_step,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    return payload
