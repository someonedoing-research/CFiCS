import torch
import torch.nn.functional as F
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from typing import Tuple

def multi_task_contrastive_loss(
    node_emb: torch.Tensor,  # [B, hidden_dim]
    labels: torch.Tensor,    # [B, 4 + 3 + 8 = 15], i.e. CF, IC, Skills in one-hot/multi-hot
    temperature: float = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A supervised contrastive loss for multi-task data (CF, IC, Skills).
    We produce three separate contrastive losses:
      - cf_loss:    Contrastive loss for CF
      - ic_loss:    Contrastive loss for IC
      - skills_loss: Contrastive loss for Skills

    Each subtask checks if two nodes share the same 'CF' or 'IC' or 'Skill' label
    (depending on the sub-label range). We'll treat them as "positives" for that subtask.

    Args:
      node_emb: [B, hidden_dim] - The node embeddings (common for all tasks).
      labels:   [B, 15], with columns:
                 0..3 => CF (4 columns)
                 4..6 => IC (3 columns)
                 7..14 => Skills (8 columns)
      temperature: Softmax temperature scaling.

    Returns:
      (cf_loss, ic_loss, skills_loss)
    """

    # 1) L2-normalize for stable cosine similarity
    node_emb = F.normalize(node_emb, p=2, dim=-1)  # shape [B, d]

    # 2) Compute pairwise sim: shape [B, B]
    sim_matrix = node_emb @ node_emb.t()  # cos_sim since normalized

    # We'll define a helper to compute a subtask's contrastive loss given a "pos_mask"
    def subtask_contrastive_loss(pos_mask: torch.Tensor) -> torch.Tensor:
        """
        Given a boolean pos_mask[i,j] => True if node j is a positive for node i,
        compute an InfoNCE-style cross-entropy contrastive loss.
        """
        device = pos_mask.device
        B = pos_mask.size(0)

        # Exclude diagonal
        diag = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask[diag] = False

        # logits => sim / temperature, then row-wise log_softmax
        logits = sim_matrix / temperature
        log_probs = F.log_softmax(logits, dim=1)  # shape [B, B]

        # gather log_probs[i,j] for j in positives
        mask_indices = pos_mask.nonzero(as_tuple=False)  # shape [N_pos, 2]
        if mask_indices.size(0) == 0:
            # If no positives at all => return 0.0
            return torch.tensor(0.0, device=device, requires_grad=True)

        selected_log_probs = log_probs[mask_indices[:,0], mask_indices[:,1]]
        loss = -selected_log_probs.mean()
        return loss

    # 3) Split the label columns: [B, 15]
    cf_labels = labels[:, :4]       # shape [B, 4]
    ic_labels = labels[:, 4:7]      # shape [B, 3]
    skill_labels = labels[:, 7:]    # shape [B, 8]

    # 4) Build "positive" masks for each subtask:
    #    For CF: pos_mask_cf[i,j] = True if i != j and cf_labels[i] ∩ cf_labels[j] != ∅
    #            Because CF is 1-of-4 typically, but if you have multi-hot, we handle that too.
    #    Similarly for IC, Skills.

    # Convert each to bool for matrix multiplication
    cf_bool    = (cf_labels    > 0).float()  # shape [B,4]
    ic_bool    = (ic_labels   > 0).float()   # shape [B,3]
    skill_bool = (skill_labels > 0).float()  # shape [B,8]
    # print(f"CF bool:\n{cf_bool}\nIC bool:\n{ic_bool}\nSkills bool:\n{skill_bool}")

    # pos_mask_cf => shape [B,B], True if share at least 1 CF label
    pos_mask_cf = (cf_bool @ cf_bool.t()) > 0
    pos_mask_ic = (ic_bool @ ic_bool.t()) > 0
    pos_mask_sk = (skill_bool @ skill_bool.t()) > 0
    print(f"CF pos mask:\n{pos_mask_cf}\nIC pos mask:\n{pos_mask_ic}\nSkills pos mask:\n{pos_mask_sk}")

    # 5) Compute each subtask's contrastive loss
    cf_loss    = subtask_contrastive_loss(pos_mask_cf)
    ic_loss    = subtask_contrastive_loss(pos_mask_ic)
    skills_loss = subtask_contrastive_loss(pos_mask_sk)

    return cf_loss, ic_loss, skills_loss

def contrastive_loss(
    node_emb: torch.Tensor,   # [B, hidden_dim]
    labels: torch.Tensor,     # [B, 15], where:
                              #   columns [0..4) => CF (4 one-hot columns)
                              #   columns [4..7) => IC (3 one-hot columns)
                              #   columns [7..15) => Skill (8 one-hot columns)
    temperature: float = 0.1
) -> torch.Tensor:
    """
    A supervised contrastive loss for multi-task labels (CF, IC, Skills) but 
    we treat EXACT matches in all three tasks as positives.

    i.e. pos_mask[i,j] = True iff:
        CF_index[i] == CF_index[j]  AND
        IC_index[i] == IC_index[j]  AND
        Skill_index[i] == Skill_index[j]
    and i != j.

    The rest is a standard InfoNCE approach: 
      - compute sim(i,j) => cos_sim
      - each row's log-softmax
      - gather log probs over positives
      - average negative log-likelihood

    Returns:
      A single scalar contrastive loss. If no positives exist, returns 0.0
    """

    device = node_emb.device
    B = node_emb.size(0)

    # 1) Split the 15 columns into sub-blocks
    cf_labels = labels[:, :4]       # [B,4]
    ic_labels = labels[:, 4:7]      # [B,3]
    skill_labels = labels[:, 7:]    # [B,8]

    # 2) Convert one-hot to indices
    #    For each node, we get exactly one CF, one IC, and one Skill
    cf_index    = cf_labels.argmax(dim=1)       # [B]
    ic_index    = ic_labels.argmax(dim=1)       # [B]
    skill_index = skill_labels.argmax(dim=1)    # [B]

    # 3) L2-normalize embeddings => cos_sim
    node_emb = F.normalize(node_emb, p=2, dim=-1)  # [B,d]

    # 4) Build the exact-match mask
    #    pos_mask[i,j] = True if (cf_i == cf_j) & (ic_i == ic_j) & (skill_i == skill_j) & i != j
    match_cf    = (cf_index.unsqueeze(1)    == cf_index.unsqueeze(0))     # shape [B,B], bool
    match_ic    = (ic_index.unsqueeze(1)    == ic_index.unsqueeze(0))
    match_skill = (skill_index.unsqueeze(1) == skill_index.unsqueeze(0))
    pos_mask = match_cf & match_ic & match_skill

    # Exclude diagonal (i == j)
    diag = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask[diag] = False

    # 5) Compute similarity matrix [B,B], then row-wise log_softmax
    sim_matrix = node_emb @ node_emb.t()    # cos_sim
    logits = sim_matrix / temperature       # shape [B,B]
    log_probs = F.log_softmax(logits, dim=1)

    # 6) Collect all log_probs[i,j] for j in positives
    mask_indices = pos_mask.nonzero(as_tuple=False)  # shape [N_pos, 2]
    if mask_indices.size(0) == 0:
        # If no positives at all => return 0.0
        return torch.tensor(0.0, device=device, requires_grad=True)

    selected_log_probs = log_probs[mask_indices[:, 0], mask_indices[:, 1]]

    # InfoNCE => average negative log probability over positives
    loss = -selected_log_probs.mean()
    return loss

def compute_losses(
    factors_logits: torch.Tensor,
    intervention_concept_logits: torch.Tensor,
    skills_logits: torch.Tensor,
    labels: torch.Tensor,  # Combined labels
    example_indices: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute losses for both tasks.

    Args:
        factors_logits: Tensor of shape [N, num_factors].
        intervention_concept_logits: Tensor of shape [N, num_ic].
        skills_logits: Tensor of shape [N, num_skills].
        labels: Combined labels, assumed to be one-hot for each task.
                First 4 columns for factors, next 3 for intervention concepts, remaining for skills.
        example_indices: Optional tensor of indices to select examples from logits and labels.
                         In full-batch mode, use these indices. In mini-batch mode, leave as None.
    Returns:
        Tuple of (factors_loss, intervention_concept_loss, skills_loss)
    """
    # Split labels into separate tasks
    factors_labels = labels[:, :4]       # First 4 columns for factors
    ic_labels = labels[:, 4:7]             # Next 3 columns for intervention concepts
    skills_labels = labels[:, 7:]          # Remaining columns for skills

    # If example_indices is provided, select only those examples.
    # Otherwise, use all nodes in the batch (which is typical for mini-batch training).
    if example_indices is not None:
        # Ensure indices are within bounds.
        if example_indices.max() >= factors_logits.size(0):
            raise IndexError("example_indices contains indices out of bounds for the logits.")
        f_logits = factors_logits[example_indices]
        ic_logits = intervention_concept_logits[example_indices]
        s_logits = skills_logits[example_indices]
        f_labels = factors_labels[example_indices]
        ic_labels_selected = ic_labels[example_indices]
        s_labels = skills_labels[example_indices]
    else:
        f_logits = factors_logits
        ic_logits = intervention_concept_logits
        s_logits = skills_logits
        f_labels = factors_labels
        ic_labels_selected = ic_labels
        s_labels = skills_labels

    # Convert one-hot labels to class indices using argmax
    f_targets = f_labels.argmax(dim=1)
    ic_targets = ic_labels_selected.argmax(dim=1)
    s_targets = s_labels.argmax(dim=1)

    # Compute cross entropy losses for each task
    factors_loss = F.cross_entropy(f_logits, f_targets)
    intervention_concept_loss = F.cross_entropy(ic_logits, ic_targets)
    skills_loss = F.cross_entropy(s_logits, s_targets)

    return factors_loss, intervention_concept_loss, skills_loss

# ------------------------------------------
#  TEST CASES IN MAIN
# ------------------------------------------
if __name__ == "__main__":
    print("Running tests for multi_task_contrastive_loss...")

    # Test 1: Basic correctness (shape)
    B, D = 10, 16  # Batch size = 10, Embedding dim = 16
    node_emb = torch.randn(B, D)  # Random embeddings
    # labels = torch.randint(0, 2, (B, 15))  # Random multi-hot labels
    # Generate random labels with a single 1 in each segment
    labels = torch.zeros(B, 15)
    labels[:, :4] = F.one_hot(torch.randint(0, 4, (B,)), num_classes=4)
    labels[:, 4:7] = F.one_hot(torch.randint(0, 3, (B,)), num_classes=3)
    labels[:, 7:] = F.one_hot(torch.randint(0, 8, (B,)), num_classes=8)
    
    
    print(f"Labels:\n{labels}")

    cf_loss, ic_loss, skill_loss = multi_task_contrastive_loss(node_emb, labels)
    
    assert cf_loss.shape == (), "CF loss should be a scalar"
    assert ic_loss.shape == (), "IC loss should be a scalar"
    assert skill_loss.shape == (), "Skills loss should be a scalar"
    
    print("Test 1 Passed ✅")

    # Test 2: All nodes share the same label (should have meaningful loss)
    labels_same = torch.ones(B, 15)  # Every node has all labels
    cf_loss, ic_loss, skill_loss = multi_task_contrastive_loss(node_emb, labels_same)

    assert cf_loss > 0, "CF loss should be positive when all nodes share labels"
    assert ic_loss > 0, "IC loss should be positive when all nodes share labels"
    assert skill_loss > 0, "Skills loss should be positive when all nodes share labels"

    print("Test 2 Passed ✅")

    # Test 3: All nodes are unique (zero loss expected)
    labels_unique = torch.eye(B, 15)  # Each node has a unique label
    cf_loss, ic_loss, skill_loss = multi_task_contrastive_loss(node_emb, labels_unique)

    assert torch.isclose(cf_loss, torch.tensor(0.0), atol=1e-6), "CF loss should be near zero"
    assert torch.isclose(ic_loss, torch.tensor(0.0), atol=1e-6), "IC loss should be near zero"
    assert torch.isclose(skill_loss, torch.tensor(0.0), atol=1e-6), "Skills loss should be near zero"

    print("Test 3 Passed ✅")

    # Test 4: Check if loss is differentiable
    node_emb.requires_grad_(True)  # Enable gradients
    cf_loss, ic_loss, skill_loss = multi_task_contrastive_loss(node_emb, labels)
    total_loss = cf_loss + ic_loss + skill_loss
    total_loss.backward()

    assert node_emb.grad is not None, "Gradient should flow through node embeddings"
    print("Test 4 Passed ✅")

    print("All tests passed successfully! 🎉")