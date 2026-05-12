import logging

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def class_frequencies_from_oracle(
    *,
    train_loader,
    neighbor_loader_factory,
    ground_truth_oracle,
    data,
    all_chan_ids,
    num_nodes: int,
    num_classes: int,
    t_min,
    device,
) -> tuple:
    """Return (class_frequencies, majority_class, class_counts).

    class_frequencies is a 1-d tensor on `device` of length num_classes;
    majority_class is the argmax index. Falls back to uniform if no edges
    were ever observed (total == 0).
    """
    log.info("Computing class frequencies from ground truth oracle...")
    class_counts = torch.zeros(num_classes, device=device)
    tmp_loader = neighbor_loader_factory(num_nodes=num_nodes, device=device)

    for batch in tqdm(train_loader, desc="Computing class frequencies"):
        batch = batch.to(device)

        opening = torch.atleast_1d(batch.edge_status.nonzero().squeeze())
        if opening.numel() > 0:
            tmp_loader.insert(
                batch.src[opening], batch.dst[opening], batch.global_id[opening]
            )

        _, _, edge_original_indices = tmp_loader()
        if edge_original_indices.numel() > 0:
            ts_ms = (batch.t.max().item() + t_min.item()) * 1000
            ground_truth_labels = ground_truth_oracle.get_ground_truth_for_open_edges(
                data.src[edge_original_indices].to(device),
                data.dst[edge_original_indices].to(device),
                data.raw_t[edge_original_indices].to(device) * 1000,
                ts_ms,
                edge_chan_ids=all_chan_ids[edge_original_indices.cpu().numpy()],
            ).to(device)
            for label in ground_truth_labels:
                class_counts[label] += 1

        closing = torch.atleast_1d((batch.edge_status == 0).nonzero().squeeze())
        if closing.numel() > 0:
            tmp_loader.remove(batch.src[closing], batch.dst[closing])

    total = class_counts.sum()
    frequencies = (
        class_counts / total
        if total > 0
        else torch.ones(num_classes, device=device) / num_classes
    )
    majority = class_counts.argmax().item()
    return frequencies, majority, class_counts
