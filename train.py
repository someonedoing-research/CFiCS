# train.py

import torch
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
import copy
import math
from typing import Any, Dict, Tuple, List
import wandb

from data_loading import load_data
from model import CFiCS
from loss import contrastive_loss, multi_task_contrastive_loss, compute_losses
from config.config_loader import load_config
from sklearn.metrics import f1_score
from metrics import evaluate_embeddings

def train_loop(
    model: CFiCS,
    data,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    task_weights: List[float],
    patience: int,
    log_every: int = 10,
    num_classes: int = (4, 3, 8), # The classes (CF, IC, S)
    wandb_enabled: bool = False, 
    batch_size: int = 1024,
    contrastive_weight: float = 0.1,  # how strongly we weigh the contrastive term
    temperature: float = 0.1          # for multi_task_contrastive_loss
) -> Tuple[List[float], List[float]]:
    """
    A train loop that uses classification loss + contrastive loss.
    """

    # Make edge_index contiguous 
    data.edge_index = data.edge_index.contiguous()
    
    # the number of classes to slice the logits into
    num_cf_classes, num_ic_classes, num_s_classes = num_classes

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    device = next(model.parameters()).device  # Ensure we know which device the model is on

    for epoch in range(epochs):
        # -----------------
        # TRAINING PHASE
        # -----------------
        model.train()
        optimizer.zero_grad()
        
        # 1) Forward pass (full-batch for simplicity)
        logits, graph_emb = model(
            x_text=data.x,
            x_graph=data.x,
            edge_index=data.edge_index
        )
        
        # 2) Slice out the class-specific logits
        f_logits =  logits[:, :num_cf_classes]
        ic_logits = logits[:, num_cf_classes : num_cf_classes+num_ic_classes]
        s_logits =  logits[:, num_cf_classes+num_ic_classes :]

        # 3) Classification loss on train subset only
        if len(train_indices) > 0:
            f_loss, ic_loss, s_loss = compute_losses(
                f_logits,
                ic_logits,
                s_logits,
                data.y,
                example_indices=train_indices
            )
            class_loss = (
                task_weights[0] * f_loss +
                task_weights[1] * ic_loss +
                task_weights[2] * s_loss
            )
        else:
            class_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)

        # 4) Contrastive loss on the graph embeddings for train subset
        if len(train_indices) > 0:
            emb_train = graph_emb[train_indices]
            labels_train = data.y[train_indices]
            cont_loss = contrastive_loss(emb_train, labels_train, temperature=temperature)
        else:
            cont_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)

        # 5) Combine losses
        total_loss = class_loss + contrastive_weight * cont_loss
        total_loss.backward()
        optimizer.step()
        train_losses.append(total_loss.item())

        # Evaluate training F1 on train_indices
        f_targets_train = data.y[train_indices, :num_cf_classes].argmax(dim=1)
        ic_targets_train = data.y[train_indices, num_cf_classes:num_cf_classes+num_ic_classes].argmax(dim=1)
        s_targets_train = data.y[train_indices, num_cf_classes+num_ic_classes:].argmax(dim=1)

        cf_f1_micro_train = f1_score(f_targets_train.cpu(), f_logits[train_indices].argmax(dim=1).cpu(), average='micro')
        cf_f1_macro_train = f1_score(f_targets_train.cpu(), f_logits[train_indices].argmax(dim=1).cpu(), average='macro')

        # ... (Similar for ic and s if you want them in logging) ...

        # -----------------
        # VALIDATION PHASE
        # -----------------
        model.eval()
        with torch.no_grad():
            val_logits, val_graph_emb = model(
                x_text=data.x,
                x_graph=data.x,
                edge_index=data.edge_index
            )
            
            f_val_logits = val_logits[:, :num_cf_classes]
            ic_val_logits = val_logits[:, num_cf_classes:num_cf_classes+num_ic_classes]
            s_val_logits = val_logits[:, num_cf_classes+num_ic_classes:]
            
            if len(val_indices) > 0:
                vf_loss, vic_loss, vs_loss = compute_losses(
                    f_val_logits,
                    ic_val_logits,
                    s_val_logits,
                    data.y,
                    example_indices=val_indices
                )
                val_class_loss = (
                    task_weights[0] * vf_loss +
                    task_weights[1] * vic_loss +
                    task_weights[2] * vs_loss
                )
                # Possibly also do contrastive on val set (optional).
                emb_val = val_graph_emb[val_indices]
                labels_val = data.y[val_indices]
                val_cont_loss = contrastive_loss(emb_val, labels_val, temperature=temperature)
                val_total_loss = val_class_loss + contrastive_weight * val_cont_loss
                
                # Evaluate embeddings on the val split
                emb_metrics = evaluate_embeddings(emb_val, labels_val, k_values=[5])
            else:
                val_total_loss = torch.tensor(0.0, device=val_logits.device)
                emb_metrics = {"P@5":0.0, "R@5":0.0, "MRR":0.0}

        val_losses.append(val_total_loss.item())
        current_val_loss = val_total_loss.item()

        # ---------------
        # EARLY STOPPING
        # ---------------
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            print(f"Early stopping at epoch {epoch+1} (no improvement).")
            if wandb_enabled:
                wandb.log({"Early Stopping": epoch+1})
            break

        # -------------
        # LOGGING
        # -------------
        if (epoch + 1) % log_every == 0:
            log_dict = {
                "Epoch": epoch + 1,
                "Train Loss": train_losses[-1],
                "Train Cont Loss": cont_loss.item(),
                "Val Loss": val_losses[-1],
                "Patience Count": epochs_no_improve,
                "Train CF F1 (micro)": cf_f1_micro_train,
                "Train CF F1 (macro)": cf_f1_macro_train,
                "Val Emb P@5": emb_metrics["P@5"],
                "Val Emb R@5": emb_metrics["R@5"],
                "Val Emb MRR": emb_metrics["MRR"],
            }
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_losses[-1]:.4f} | "
                  f"Train CE Loss: {class_loss:.4f} | "
                  f"Train Cont Loss: {cont_loss.item():.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | "
                  f"Val CE Loss: {val_class_loss:.4f} | "
                  f"Val Cont Loss: {val_cont_loss.item():.4f} | "
                  f"CF F1 Micro: {cf_f1_micro_train:.4f} | "
                  f"Patience: {epochs_no_improve} | "
                  f"Val P@5: {emb_metrics['P@5']:.4f}, "
                  f"R@5: {emb_metrics['R@5']:.4f}, "
                  f"MRR: {emb_metrics['MRR']:.4f}")
            if wandb_enabled:
                wandb.log(log_dict)

    # If we stopped early, ensure best state is loaded
    if best_model_state is not None and epochs_no_improve < patience:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses

def train_model(
    model: CFiCS,
    data,
    optimizer: torch.optim.Optimizer,
    epochs: int = 200,
    task_weights: List[float] = [1.0, 1.0, 1.0],
    patience: int = 50,
    wandb_config: Dict[str, Any] = None,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1
) -> Tuple[List[float], List[float]]:
    """
    Convenience method that extracts train and val indices from
    data.train_mask and data.val_mask, initializes WANDB if enabled,
    and calls 'train_loop'.
    """
    # -----------------------------------------
    # 1) Move MODEL and DATA to the same device
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    data = data.to(device)

    # Extract indices from masks
    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    val_indices = data.val_mask.nonzero(as_tuple=True)[0]

    # Ensure they're at least 1D and move them to device as well
    if train_indices.dim() == 0:
        train_indices = train_indices.unsqueeze(0)
    if val_indices.dim() == 0:
        val_indices = val_indices.unsqueeze(0)

    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    # Initialize WANDB if enabled
    wandb_enabled = False
    if wandb_config and wandb_config.get('enabled', False):
        wandb_enabled = True
        wandb.init(
            project=wandb_config.get('project'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            notes=wandb_config.get('notes'),
            config={
                "epochs": epochs,
                "patience": patience,
                "task_weights": task_weights,
                "optimizer": type(optimizer).__name__,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "weight_decay": optimizer.param_groups[0]['weight_decay'],
                "model": {
                    "hidden_channels": model.hidden_channels,
                    "num_common_factors": model.num_common_factors,
                    "num_intervention_concepts": model.num_intervention_concepts,
                    "num_skills": model.num_skills,
                    "num_layers": model.num_layers,
                    "dropout": model.dropout
                }
            }
        )

    # Train the model
    train_losses, val_losses = train_loop(
        model,
        data,
        optimizer,
        epochs=epochs,
        train_indices=train_indices,
        val_indices=val_indices,
        task_weights=task_weights,
        patience=patience,
        log_every=10,  # or any interval you like
        wandb_enabled=wandb_enabled,
        contrastive_weight=contrastive_weight,
        temperature=temperature
    )

    # Finish WANDB run
    if wandb_enabled:
        wandb.finish()

    return train_losses, val_losses


def cross_validate(
    data,
    model,
    n_folds: int,
    epochs: int,
    hidden_channels: int,
    num_common_factors: int,
    num_intervention_concepts: int,
    num_skills: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    patience: int,
    task_weights: List[float],
    wandb_config: Dict[str, Any] = None,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1
) -> float:
    """
    Perform k-fold cross-validation on the 'examples' in 'data',
    ignoring data.test_mask (treated as hold-out).
    """

    import math
    import copy
    from sklearn.metrics import f1_score
    from torch.optim import Adam
    
    
    # -----------------------------------------
    # 1) Move MODEL and DATA to the same device
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    data = data.to(device)

    # == For slicing logits ==
    # CF => 4 classes
    # IC => 3 classes
    # SKILLS => 8 classes
    num_cf_classes, num_ic_classes, num_s_classes = (4, 3, 8)

    # Indices not in test set
    not_test_mask = ~data.test_mask
    example_indices = not_test_mask.nonzero(as_tuple=False).squeeze().detach()

    # Shuffle
    example_indices = example_indices[torch.randperm(example_indices.size(0))]

    fold_size = math.ceil(example_indices.size(0) / n_folds)
    val_losses_per_fold = []

    # -- F1 METRICS ADDED --
    # For storing fold-level F1 results for each subtask
    cf_micro_folds, cf_macro_folds = [], []
    ic_micro_folds, ic_macro_folds = [], []
    skill_micro_folds, skill_macro_folds = [], []

    for fold_idx in range(n_folds):
        print(f"\n=== Fold {fold_idx+1} / {n_folds} ===")

        # 1) Create train/val splits
        val_start = fold_idx * fold_size
        val_end = min(val_start + fold_size, example_indices.size(0))

        val_indices = example_indices[val_start:val_end]
        train_indices = torch.cat(
            [example_indices[:val_start], example_indices[val_end:]],
            dim=0
        )

        # 2) New optimizer & (re)initialize WANDB if needed
        from torch.optim import Adam
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        if wandb_config and wandb_config.get('enabled', False):
            import wandb
            run_name = wandb_config.get('name', f"Fold-{fold_idx+1}")
            wandb.init(
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity'),
                name=run_name,
                notes=wandb_config.get('notes'),
                config={
                    "fold": fold_idx + 1,
                    "epochs": epochs,
                    "patience": patience,
                    "task_weights": task_weights,
                    "optimizer": type(optimizer).__name__,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "weight_decay": optimizer.param_groups[0]['weight_decay'],
                    "model": {
                        "hidden_channels": hidden_channels,
                        "num_common_factors": num_common_factors,
                        "num_intervention_concepts": num_intervention_concepts,
                        "num_skills": num_skills,
                        "num_layers": num_layers,
                        "dropout": dropout
                    }
                }
            )

        # 3) Train using your train_loop
        train_losses, val_losses = train_loop(
            model,
            data,
            optimizer,
            epochs=epochs,
            train_indices=train_indices,
            val_indices=val_indices,
            task_weights=task_weights,
            patience=patience,
            log_every=10,
            wandb_enabled=wandb_config.get('enabled', False) if wandb_config else False,
            batch_size=1024,
            contrastive_weight=contrastive_weight,
            temperature=temperature
        )

        # 4) Get final validation loss
        final_val_loss = val_losses[-1] if val_losses else float('inf')
        val_losses_per_fold.append(final_val_loss)
        print(f"[Fold {fold_idx+1}/{n_folds}] Final Val Loss: {final_val_loss:.4f}\n")

        # 5) Evaluate on val_indices to get F1 metrics after best model is loaded
        # -- F1 METRICS ADDED --
        model.eval()
        with torch.no_grad():
            # forward pass
            logits, graph_emb = model(
                x_text=data.x,
                x_graph=data.x,
                edge_index=data.edge_index
            )  # shape: [N, 4+3+8 = 15]

            cf_logits = logits[:, :num_cf_classes]  # [N,4]
            ic_logits = logits[:, num_cf_classes : num_cf_classes + num_ic_classes]  # [N,3]
            s_logits  = logits[:, num_cf_classes + num_ic_classes :]  # [N,8]

            # Slice the target y for these val_indices
            cf_targets = data.y[val_indices, :num_cf_classes].argmax(dim=1)
            ic_targets = data.y[val_indices, num_cf_classes : num_cf_classes+num_ic_classes].argmax(dim=1)
            s_targets  = data.y[val_indices, num_cf_classes+num_ic_classes : ].argmax(dim=1)

            # Preds
            cf_preds = cf_logits[val_indices].argmax(dim=1)
            ic_preds = ic_logits[val_indices].argmax(dim=1)
            s_preds  = s_logits[val_indices].argmax(dim=1)

        # Now compute micro & macro F1 for each subtask
        cf_micro = f1_score(cf_targets.cpu(), cf_preds.cpu(), average='micro', zero_division=0)
        cf_macro = f1_score(cf_targets.cpu(), cf_preds.cpu(), average='macro', zero_division=0)

        ic_micro = f1_score(ic_targets.cpu(), ic_preds.cpu(), average='micro', zero_division=0)
        ic_macro = f1_score(ic_targets.cpu(), ic_preds.cpu(), average='macro', zero_division=0)

        skill_micro = f1_score(s_targets.cpu(), s_preds.cpu(), average='micro', zero_division=0)
        skill_macro = f1_score(s_targets.cpu(), s_preds.cpu(), average='macro', zero_division=0)

        cf_micro_folds.append(cf_micro)
        cf_macro_folds.append(cf_macro)
        ic_micro_folds.append(ic_micro)
        ic_macro_folds.append(ic_macro)
        skill_micro_folds.append(skill_micro)
        skill_macro_folds.append(skill_macro)

        print(f"FOLD {fold_idx+1} F1 Scores:")
        print(f"  CF    => micro: {cf_micro:.4f}, macro: {cf_macro:.4f}")
        print(f"  IC    => micro: {ic_micro:.4f}, macro: {ic_macro:.4f}")
        print(f"  Skill => micro: {skill_micro:.4f}, macro: {skill_macro:.4f}\n")
        
        val_emb = graph_emb[val_indices]
        val_labels = data.y[val_indices]

        emb_metrics = evaluate_embeddings(val_emb, val_labels, k_values=[5])
        print(f"FOLD {fold_idx+1} EMBEDDING METRICS: "
            f"P@5={emb_metrics['P@5']:.4f}, "
            f"R@5={emb_metrics['R@5']:.4f}, "
            f"MRR={emb_metrics['MRR']:.4f}")

        # 6) WANDB logging if enabled
        if wandb_config and wandb_config.get('enabled', False):
            import wandb
            wandb.log({
                "Final Val Loss": final_val_loss,
                "CF micro F1": cf_micro,
                "CF macro F1": cf_macro,
                "IC micro F1": ic_micro,
                "IC macro F1": ic_macro,
                "Skill micro F1": skill_micro,
                "Skill macro F1": skill_macro,                
                "Val Emb P@5": emb_metrics["P@5"],
                "Val Emb R@5": emb_metrics["R@5"],
                "Val Emb MRR": emb_metrics["MRR"]
            })
            wandb.finish()

    # -------------------------------------------------------------------------
    # 7) Compute average validation loss & average F1 across folds
    # -------------------------------------------------------------------------
    mean_val_loss = float(torch.tensor(val_losses_per_fold).mean().item())

    mean_cf_micro = float(torch.tensor(cf_micro_folds).mean().item())
    std_cf_micro  = float(torch.tensor(cf_micro_folds).std().item())
    mean_cf_macro = float(torch.tensor(cf_macro_folds).mean().item())
    std_cf_macro  = float(torch.tensor(cf_macro_folds).std().item())

    mean_ic_micro = float(torch.tensor(ic_micro_folds).mean().item())
    std_ic_micro  = float(torch.tensor(ic_micro_folds).std().item())
    mean_ic_macro = float(torch.tensor(ic_macro_folds).mean().item())
    std_ic_macro  = float(torch.tensor(ic_macro_folds).std().item())

    mean_skill_micro = float(torch.tensor(skill_micro_folds).mean().item())
    std_skill_micro  = float(torch.tensor(skill_micro_folds).std().item())
    mean_skill_macro = float(torch.tensor(skill_macro_folds).mean().item())
    std_skill_macro  = float(torch.tensor(skill_macro_folds).std().item())

    print("\n===== CROSS-VALIDATION RESULTS =====")
    print(f"Avg Val Loss across folds: {mean_val_loss:.4f}")
    print("CF F1 micro: {:.4f} ± {:.4f}, macro: {:.4f} ± {:.4f}"
          .format(mean_cf_micro, std_cf_micro, mean_cf_macro, std_cf_macro))
    print("IC F1 micro: {:.4f} ± {:.4f}, macro: {:.4f} ± {:.4f}"
          .format(mean_ic_micro, std_ic_micro, mean_ic_macro, std_ic_macro))
    print("Skill F1 micro: {:.4f} ± {:.4f}, macro: {:.4f} ± {:.4f}"
          .format(mean_skill_micro, std_skill_micro, mean_skill_macro, std_skill_macro))

    return mean_val_loss


def main():
    # Load configuration
    config = load_config("./config/config.yml")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    print(f"Loading {config['data']['filepath']}\nEmbedding model: {config['data']['model_name']}\nSkipping embedding: {config['data']['skip_text_embedding']}")
    data, _ = load_data(config['data']['filepath'], config['data']['model_name'], config['data']['skip_text_embedding'])

    n_train = data.train_mask.sum().item()
    n_val = data.val_mask.sum().item()
    n_test = data.test_mask.sum().item()

    print(f"Number of training examples: {n_train}")
    print(f"Number of validation examples: {n_val}")
    print(f"Number of test examples: {n_test}")
 
    # Check if cross-validation is enabled
    cross_val_enabled = config.get('cross_validation', {}).get('enabled', False)
    
    # Initialize the model
    model = CFiCS(
        text_dim=data.x.size(1),
        graph_dim=data.x.size(1),
        hidden_channels=config['model']['hidden_channels'],
        num_common_factors=config['model']['num_common_factors'],
        num_intervention_concepts=config['model']['num_intervention_concepts'],
        num_skills=config['model']['num_skills'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        graph_model_type=config['model'].get('graph_model_type', 'sage'),
        aggregator=config['model'].get('aggregator', 'mean')
    )

    if cross_val_enabled:
        # Extract cross-validation parameters
        cv_params = config['cross_validation']
        
        print("Running k-Fold Cross Validation...")

        cross_validate(
            data,
            model,
            n_folds=cv_params['n_folds'],
            epochs=cv_params['epochs'],
            hidden_channels=config['model']['hidden_channels'],
            num_common_factors=config['model']['num_common_factors'],
            num_intervention_concepts=config['model']['num_intervention_concepts'],
            num_skills=config['model']['num_skills'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'],
            patience=cv_params['patience'],
            task_weights=cv_params['task_weights'],
            wandb_config=cv_params.get('wandb'),
            contrastive_weight=config['training']['contrastive_weight'],
            temperature=config['training']['temperature']
        )
    else:
        # Standard training
        print("Running standard training...")

        # Setup optimizer
        optimizer = Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Train if we actually have training examples
        if n_train > 0:
            train_losses, val_losses = train_model(
                model,
                data,
                optimizer,
                epochs=config['training']['epochs'],
                task_weights=config['training']['task_weights'],
                patience=config['training']['patience'],
                wandb_config=config['training'].get('wandb'),
                contrastive_weight=config['training']['contrastive_weight'],
                temperature=config['training']['temperature']
            )
            # Save model
            print("Training completed!")
        else:
            print("Error: No training examples available!")
            
    # Saving the model
    embeddings_used = "skip" if config['data']['skip_text_embedding'] else config['data']['model_name'].replace("/", "_")
    contrastive_weight = config['training']['contrastive_weight']
    if contrastive_weight == 0.0:
        save_path = f"CFiCS_{config['model']['graph_model_type']}_{embeddings_used}.pth"
    else:
        contrastive_weight = f"contrastive_{str(contrastive_weight).replace('.', 'p')}"
        save_path = f"CFiCS_{config['model']['graph_model_type']}_{embeddings_used}_{contrastive_weight}.pth"
    
    print(f"Saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()