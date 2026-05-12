def report_to_wandb_log(report: dict, prefix: str) -> dict:
    """Flatten an `sklearn.classification_report(output_dict=True)` for wandb.

    Skips the `support` field; emits `<prefix>/accuracy` and
    `<prefix>/<label>/<metric>` keys for every other entry.
    """
    metrics = {}
    for label, scores in report.items():
        if label == "accuracy":
            metrics[f"{prefix}/accuracy"] = scores
        elif isinstance(scores, dict):
            for metric, score in scores.items():
                if metric != "support":
                    metrics[f"{prefix}/{label}/{metric}"] = score
    return metrics
