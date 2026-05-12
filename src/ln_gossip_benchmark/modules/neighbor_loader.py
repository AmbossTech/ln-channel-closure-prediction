from typing import Tuple

import torch
from torch import Tensor


class OpenEdgesNeighborLoader:
    def __init__(self, num_nodes: int, device=None):
        """
        OpenEdgesNeighborLoader uses dynamic storage to track ALL open edges.

        Maintains parallel flat lists for fast tensor construction in __call__(),
        and a degree tensor for fast degree lookups.
        """
        self.num_nodes = num_nodes
        self.device = device
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        # Format: {node_id: {neighbor_id: edge_id}}
        self.open_edges = {}

        # Parallel flat arrays for canonical edges (src < dst)
        self._src_list: list[int] = []
        self._dst_list: list[int] = []
        self._eid_list: list[int] = []
        # Maps (min_node, max_node) -> index in the flat arrays
        self._edge_to_idx: dict[tuple[int, int], int] = {}

        # Degree tensor
        self._degree = torch.zeros(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def insert(self, src: Tensor, dst: Tensor, edge_ids: Tensor):
        """
        Insert new edges using their GLOBAL edge_ids.
        """
        assert src.size(0) == dst.size(0)
        assert src.size(0) == edge_ids.size(0)

        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        ids_np = edge_ids.cpu().numpy()

        for i in range(src.size(0)):
            edge_id = int(ids_np[i])
            s, d = int(src_np[i]), int(dst_np[i])

            # Check if this is a genuinely new edge (not already tracked)
            canonical = (min(s, d), max(s, d))
            is_new = canonical not in self._edge_to_idx

            if s not in self.open_edges:
                self.open_edges[s] = {}
            if d not in self.open_edges:
                self.open_edges[d] = {}

            self.open_edges[s][d] = edge_id
            self.open_edges[d][s] = edge_id

            if is_new:
                # Append to flat arrays
                idx = len(self._src_list)
                self._src_list.append(canonical[0])
                self._dst_list.append(canonical[1])
                self._eid_list.append(edge_id)
                self._edge_to_idx[canonical] = idx

                # Update degrees
                self._degree[s] += 1
                self._degree[d] += 1

    def remove(self, src: Tensor, dst: Tensor):
        """
        Remove edges (closing channels). Uses swap-and-pop for O(1) list removal.
        """
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()

        for i in range(src.size(0)):
            s, d = int(src_np[i]), int(dst_np[i])
            canonical = (min(s, d), max(s, d))

            # Remove from dict
            if s in self.open_edges and d in self.open_edges[s]:
                del self.open_edges[s][d]
                if len(self.open_edges[s]) == 0:
                    del self.open_edges[s]

            if d in self.open_edges and s in self.open_edges[d]:
                del self.open_edges[d][s]
                if len(self.open_edges[d]) == 0:
                    del self.open_edges[d]

            # Remove from flat arrays via swap-and-pop
            idx = self._edge_to_idx.pop(canonical, None)
            if idx is not None:
                last_idx = len(self._src_list) - 1
                if idx != last_idx:
                    # Swap with last element
                    last_canonical = (
                        self._src_list[last_idx],
                        self._dst_list[last_idx],
                    )
                    self._src_list[idx] = self._src_list[last_idx]
                    self._dst_list[idx] = self._dst_list[last_idx]
                    self._eid_list[idx] = self._eid_list[last_idx]
                    self._edge_to_idx[last_canonical] = idx
                self._src_list.pop()
                self._dst_list.pop()
                self._eid_list.pop()

                # Update degrees
                self._degree[s] -= 1
                self._degree[d] -= 1

    def __call__(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns ALL currently open edges in the graph.

        Uses pre-maintained flat arrays for fast tensor construction.
        """
        if len(self._src_list) == 0:
            empty_nodes = torch.tensor([], dtype=torch.long, device=self.device)
            empty_edges = torch.tensor([[], []], dtype=torch.long, device=self.device)
            empty_e_id = torch.tensor([], dtype=torch.long, device=self.device)
            return empty_nodes, empty_edges, empty_e_id

        source_nodes = torch.tensor(
            self._src_list, dtype=torch.long, device=self.device
        )
        neighbor_nodes = torch.tensor(
            self._dst_list, dtype=torch.long, device=self.device
        )
        e_id = torch.tensor(self._eid_list, dtype=torch.long, device=self.device)

        n_id = torch.cat([source_nodes, neighbor_nodes]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        relabeled_sources = self._assoc[source_nodes]
        relabeled_neighbors = self._assoc[neighbor_nodes]

        edge_index = torch.stack([relabeled_sources, relabeled_neighbors])

        return n_id, edge_index, e_id

    def reset_state(self):
        self.open_edges = {}
        self._src_list = []
        self._dst_list = []
        self._eid_list = []
        self._edge_to_idx = {}
        self._degree.zero_()

    def get_node_degrees(self, node_ids: Tensor) -> Tensor:
        """
        Get the current degree (number of open edges) for each node.
        Uses pre-maintained degree tensor for O(1) lookup per node.
        """
        return self._degree[node_ids].float()

    def _rebuild_arrays_from_dict(self):
        """Rebuild flat arrays and degree tensor from the open_edges dict."""
        self._src_list = []
        self._dst_list = []
        self._eid_list = []
        self._edge_to_idx = {}
        self._degree.zero_()

        for src_node, dst_dict in self.open_edges.items():
            for dst_node, edge_id in dst_dict.items():
                if src_node < dst_node:
                    idx = len(self._src_list)
                    self._src_list.append(src_node)
                    self._dst_list.append(dst_node)
                    self._eid_list.append(edge_id)
                    self._edge_to_idx[(src_node, dst_node)] = idx
            # Each entry in open_edges[node] counts toward that node's degree
            self._degree[src_node] = len(dst_dict)

    def state_dict(self):
        return {"open_edges": {k: dict(v) for k, v in self.open_edges.items()}}

    def load_state_dict(self, state):
        self.open_edges = {k: dict(v) for k, v in state["open_edges"].items()}
        self._rebuild_arrays_from_dict()
