import logging
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh

log = logging.getLogger(__name__)


def compute_spectral_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 16,
    device=None,
) -> torch.Tensor:
    """Compute top-k Laplacian eigenvectors as spectral node encodings.

    Computes the k smallest non-trivial eigenvectors of the normalized graph
    Laplacian. These encode global positional information about each node's
    role in the graph topology (hub vs peripheral, bridge vs cluster member).

    Args:
        edge_index: [2, num_edges] edge index (can be unidirectional/canonical).
        num_nodes: Number of nodes in the graph.
        k: Number of eigenvectors to compute.
        device: Target device for output tensor.

    Returns:
        Tensor of shape [num_nodes, k] with absolute eigenvector values.
        Returns zeros if computation fails or graph is too small.
    """
    if edge_index.size(1) == 0 or num_nodes < 3:
        return torch.zeros(num_nodes, k, device=device)

    start = time.time()

    # Make bidirectional (Laplacian needs both directions)
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    row_bidir = np.concatenate([row, col])
    col_bidir = np.concatenate([col, row])

    # Build sparse adjacency matrix
    data = np.ones(len(row_bidir), dtype=np.float64)
    A = coo_matrix((data, (row_bidir, col_bidir)), shape=(num_nodes, num_nodes))
    # Remove duplicate edges from bidirectionalization
    A = A.tocsr()
    A.data = np.minimum(A.data, 1.0)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    degrees = np.array(A.sum(axis=1)).flatten()
    # Isolated nodes get degree 1 to avoid division by zero
    degrees[degrees == 0] = 1.0
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L = eye(num_nodes, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    # Number of eigenvectors we can compute (need at least k+1 for non-trivial ones)
    max_k = min(k + 1, num_nodes - 2)
    if max_k < 2:
        return torch.zeros(num_nodes, k, device=device)

    try:
        _, eigenvectors = eigsh(L, k=max_k, which="SM", tol=1e-3)
    except Exception as e:
        log.warning(f"Spectral decomposition failed: {e}")
        return torch.zeros(num_nodes, k, device=device)

    # Skip first eigenvector (trivial constant for connected components)
    spectral = eigenvectors[:, 1 : k + 1]

    # Sign ambiguity: use absolute values
    spectral = np.abs(spectral)

    # Pad if fewer than k eigenvectors available
    if spectral.shape[1] < k:
        padding = np.zeros((num_nodes, k - spectral.shape[1]))
        spectral = np.concatenate([spectral, padding], axis=1)

    elapsed = time.time() - start
    if elapsed > 1.0:
        log.debug(f"Spectral encoding: {num_nodes} nodes, k={k}, took {elapsed:.2f}s")

    return torch.from_numpy(spectral.copy()).float().to(device)
