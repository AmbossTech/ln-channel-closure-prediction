import logging

import hydra
import omegaconf
import pandas as pd
import torch
import wandb

from ln_gossip_benchmark import PROJECT_ROOT
from ln_gossip_benchmark.data import (
    apply_first_day_filter,
    apply_warm_start_mask,
    build_train_end_snapshot,
)
from ln_gossip_benchmark.modules import TemporalModel
from ln_gossip_benchmark.utils.seed import set_random_seed

log = logging.getLogger(__name__)


def _enable_determinism() -> None:
    # Each setter is wrapped because they fail on some platforms (e.g. MPS
    # rejects certain deterministic algos); we want best-effort, not crash.
    for setter in (
        lambda: torch.use_deterministic_algorithms(True),
        lambda: torch.set_num_threads(1),
        lambda: torch.set_num_interop_threads(1),
    ):
        try:
            setter()
        except Exception:
            pass


def _normalize_edge_features(data, train_mask) -> None:
    """log1p + train-set min-max normalise `data.msg` in place."""
    data.msg = torch.log1p(data.msg)
    train_msg = data.msg[train_mask]
    feat_min = train_msg.min(dim=0, keepdim=True)[0]
    feat_max = train_msg.max(dim=0, keepdim=True)[0]
    feat_range = feat_max - feat_min
    feat_range = torch.where(feat_range > 1e-8, feat_range, torch.ones_like(feat_range))
    data.msg = (data.msg - feat_min) / feat_range


def _build_oracle_df(full_data: dict, all_chan_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src": full_data["sources"],
            "dst": full_data["destinations"],
            "gossip_ts": full_data["timestamps"] * 1000,
            "edge_label": full_data["edge_label"],
            "channel_status": full_data["edge_status"],
            "chan_id": all_chan_ids,
        }
    )


def _build_predictor_kwargs(cfg, data, num_classes: int, device) -> dict:
    edge_features_dim = data.msg.size(-1)
    enrich_node_features = cfg.model.enrich_node_features
    node_dim = num_classes + 4 if enrich_node_features else num_classes

    log.info(
        f"Model input dims: {edge_features_dim} edge feats, {node_dim} node feats "
        f"(enriched={enrich_node_features})"
    )

    predictor = hydra.utils.instantiate(
        cfg.model.predictor,
        edge_dim=edge_features_dim,
        node_dim=node_dim,
        out_channels=num_classes,
    ).to(device)

    model = TemporalModel(
        predictor=predictor,
        num_nodes=data.num_nodes,
        num_classes=num_classes,
        enrich_node_features=enrich_node_features,
    ).to(device)

    optimizer = hydra.utils.instantiate(cfg.model.optim, model.parameters())
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": hydra.utils.instantiate(cfg.model.scheduler, optimizer=optimizer),
        "criterion": hydra.utils.instantiate(cfg.model.criterion).to(device),
        "assoc": torch.empty(data.num_nodes, dtype=torch.long, device=device),
    }


@hydra.main(
    version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default"
)
def main(cfg: omegaconf.DictConfig):
    set_random_seed(cfg.model.seed)
    if hasattr(cfg.model, "optim") and cfg.model.optim is not None:
        log.info(f"Learning rate: {cfg.model.optim.lr}")
    _enable_determinism()

    run = hydra.utils.call(cfg.wandb)
    wandb.config.update(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    )
    wandb.run.log_code(root=str(PROJECT_ROOT / "src"), name="source_code")

    device = cfg.device

    dataset = hydra.utils.instantiate(cfg.data.dataset)
    num_classes = dataset.edge_label.unique().int().max().item() + 1
    data = dataset.get_TemporalData().to(device)

    if cfg.data.warm_start and cfg.data.exclude_first_day:
        raise ValueError("warm_start and exclude_first_day are mutually exclusive.")
    first_day_mask = None
    if cfg.data.exclude_first_day:
        data = apply_first_day_filter(data, dataset)
    elif cfg.data.warm_start:
        first_day_mask = apply_warm_start_mask(data, dataset)

    data = data.to(device)
    data.global_id = torch.arange(len(data.src), device=device)

    all_chan_ids = dataset.dataset._full_data["chan_id"]
    if all_chan_ids is None or len(all_chan_ids) != len(data.src):
        raise ValueError("chan_id missing or out-of-sync with edge tensors.")

    data.raw_t = data.t.clone()
    t_min = data.t.min()
    data.t = data.t - t_min
    log.info("Normalizing edge features...")
    _normalize_edge_features(data, dataset.train_mask)

    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]
    train_loader = hydra.utils.instantiate(cfg.data.loader, data=train_data)
    val_loader = hydra.utils.instantiate(cfg.data.loader, data=val_data)
    test_loader = hydra.utils.instantiate(cfg.data.loader, data=test_data)

    log.info(
        f"Dataset: total {len(data.src):,}, train {len(train_data.src):,}, "
        f"val {len(val_data.src):,}, test {len(test_data.src):,}"
    )

    neighbor_loader = hydra.utils.instantiate(
        cfg.model.neighbor_loader,
        num_nodes=data.num_nodes,
        device=device,
    )
    ground_truth_oracle = hydra.utils.instantiate(
        cfg.model.ground_truth_oracle,
        data_df=_build_oracle_df(dataset.dataset._full_data, all_chan_ids),
    )

    warm_start_snapshot = None
    if first_day_mask is not None:
        log.info("Building warm-start snapshot from first-day events...")
        first_day_loader = hydra.utils.instantiate(
            cfg.data.loader,
            data=data[first_day_mask.to(device)],
        )
        warm_start_snapshot = build_train_end_snapshot(
            first_day_loader,
            cfg.model.neighbor_loader,
            num_nodes=data.num_nodes,
            num_classes=num_classes,
            device=device,
        )

    log.info("Building training-end snapshot...")
    train_end_snapshot = build_train_end_snapshot(
        hydra.utils.instantiate(cfg.data.loader, data=train_data),
        cfg.model.neighbor_loader,
        num_nodes=data.num_nodes,
        num_classes=num_classes,
        device=device,
        initial_snapshot=warm_start_snapshot,
    )

    trainer_args = {
        "neighbor_loader": neighbor_loader,
        "ground_truth_oracle": ground_truth_oracle,
        "data": data,
        "device": device,
        "num_classes": num_classes,
        "cfg": cfg,
        "run": run,
        "train_end_snapshot": train_end_snapshot,
        "warm_start_snapshot": warm_start_snapshot,
        "train_loader": train_loader,
        "all_chan_ids": all_chan_ids,
        "t_min": t_min,
    }

    log.info(f"Model: {cfg.model.name}")
    # Random and tree baselines don't have a `predictor` block in their YAML.
    if hasattr(cfg.model, "predictor") and cfg.model.predictor is not None:
        learned = _build_predictor_kwargs(cfg, data, num_classes, device)
        run.watch(learned["model"], log="all", log_freq=cfg.model.eval_every_steps)
        trainer_args.update(learned)

    trainer = hydra.utils.instantiate(cfg.model.trainer, **trainer_args)
    trainer.train_loop(train_loader, val_loader)
    trainer.test(
        test_loader,
        val_loader,
        save_predictions_path=cfg.model.save_predictions_path,
    )

    if cfg.model.save_model_path and hasattr(trainer, "model"):
        torch.save(trainer.model.state_dict(), cfg.model.save_model_path)
        log.info(f"Saved model checkpoint to {cfg.model.save_model_path}")

    run.finish()


if __name__ == "__main__":
    main()
