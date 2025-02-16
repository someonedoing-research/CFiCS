import torch
import torch.nn.functional as F

from typing import List

def evaluate_embeddings(
    emb: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
):
    """
    Evaluate embedding quality using:
      - Precision@k
      - Recall@k
      - Mean Reciprocal Rank (MRR)

    'relevant' is defined here as 'sharing >=1 label' if 'labels' is multi-hot.
    If labels are single-label, we define relevance as identical labels.

    Args:
        emb (torch.Tensor): Node embeddings [N, D].
        labels (torch.Tensor): Multi-hot or single-label [N, L] or [N].
        k_values (list[int]): The top-k thresholds to evaluate.

    Returns:
        dict: e.g. {
            "P@1": float, "R@1": float,
            "P@5": float, "R@5": float,
            "MRR": float
        }
    """
    device = emb.device
    N = emb.size(0)

    # 1) Build relevance mask
    if labels.dim() == 1:
        # single-label => relevant if same label
        relevant_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    else:
        # multi-hot => exact match in label sets => relevant
        relevant_mask = torch.all(labels.unsqueeze(1) == labels.unsqueeze(0), dim=-1)
        
        # at least one label => relevant
        # label_bool = (labels > 0).float()
        # sim_labels = label_bool @ label_bool.t()  # shape [N,N]
        # relevant_mask = (sim_labels > 0)


    # Exclude self
    diag_idx = torch.arange(N, device=device)
    relevant_mask[diag_idx, diag_idx] = False

    # 2) Cosine similarity
    emb_norm = F.normalize(emb, p=2, dim=-1)
    sim_matrix = emb_norm @ emb_norm.t()  # [N,N], high => more similar

    # 3) Sort neighbors by descending similarity
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    # 4) Evaluate
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    rr_list = []  # for MRR

    for i in range(N):
        ranked_list = sorted_indices[i]          # [N], best to worst
        i_relevance = relevant_mask[i, ranked_list]  # bool, length N
        total_relevant = i_relevance.sum().item()

        # MRR => rank of the first relevant
        if total_relevant == 0:
            rr_list.append(0.0)
        else:
            pos_idx = i_relevance.nonzero(as_tuple=True)[0]
            first_rank = pos_idx[0].item() + 1  # 1-based
            rr_list.append(1.0 / first_rank)

        # P@k, R@k
        for k in k_values:
            top_k_relevant = i_relevance[:k].sum().item()
            precision = top_k_relevant / k
            recall = (top_k_relevant / total_relevant) if total_relevant > 0 else 0.0
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

    metrics = {}
    for k in k_values:
        p_mean = torch.tensor(precision_at_k[k]).mean().item()
        r_mean = torch.tensor(recall_at_k[k]).mean().item()
        metrics[f"P@{k}"] = p_mean
        metrics[f"R@{k}"] = r_mean

    mrr = torch.tensor(rr_list).mean().item()
    metrics["MRR"] = mrr

    return metrics