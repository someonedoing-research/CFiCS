# eval.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from data_loading import TherapeuticDataset, load_data
from model import CFiCS
from config.config_loader import load_config
from metrics import evaluate_embeddings

def evaluate_model(
    model: CFiCS, 
    data,
    num_classes = (4, 3, 8), 
    use_neighbor_sampling: bool = False, 
    num_neighbors: list = [15, 10], 
    batch_size: int = 1024
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test set. Optionally uses neighbor sampling
    for inductive or large-scale evaluation.

    Args:
        model (CFiCS): Your trained model.
        data: The PyG data object (with .test_mask, .x, .edge_index, etc.).
        use_neighbor_sampling (bool): If True, do mini-batch inference on the test set.
        num_neighbors (list): Number of neighbors to sample at each layer if using neighbor sampling.
        batch_size (int): Mini-batch size for neighbor loading.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics, including accuracies, classification reports, etc.
    """
    model.eval()

    # Identify test nodes
    test_indices = data.test_mask.nonzero().squeeze()
    if test_indices.dim() == 0:
        test_indices = test_indices.unsqueeze(0)
    n_test = test_indices.size(0)
    print(f"\nNumber of test examples: {n_test}")

    # If no test examples, just exit
    if n_test == 0:
        print("No test examples available for evaluation.")
        return {}

    # Number of classes for each task
    num_cf_classes, num_ic_classes, num_skill_classes = num_classes

    # We'll do full-batch or neighbor sampling depending on 'use_neighbor_sampling'
    with torch.no_grad():
        # -----------
        # FULL BATCH
        # -----------
        eval_logits, eval_graph_embds = model(
            x_text=data.x,
            x_graph=data.x,
            edge_index=data.edge_index
        )
        factors_logits = eval_logits[:, :num_cf_classes]
        intervention_concepts_logits = eval_logits[:, num_cf_classes:num_cf_classes+num_ic_classes]
        skills_logits = eval_logits[:, num_cf_classes+num_ic_classes:]
        # We'll simply select test_indices later
            
    # Now gather the test labels
    test_labels = data.y[test_indices]
    factor_labels = test_labels[:, :num_cf_classes]  # First 4 columns for factors
    intervention_concepts_labels = test_labels[:, num_cf_classes:num_cf_classes+num_ic_classes]  # Next 3 columns for ICs
    skill_labels = test_labels[:, num_cf_classes+num_ic_classes:]   # Remaining columns for skills

    # Factor Predictions
    factor_predictions = factors_logits[test_indices]
    factor_pred_classes = factor_predictions.argmax(dim=1)
    factor_true_classes = factor_labels.argmax(dim=1)

    # IC Predictions
    ic_predictions = intervention_concepts_logits[test_indices]
    ic_pred_classes = ic_predictions.argmax(dim=1)
    ic_true_classes = intervention_concepts_labels.argmax(dim=1)

    # Skill Predictions
    skill_predictions = skills_logits[test_indices]
    skill_pred_classes = skill_predictions.argmax(dim=1)
    skill_true_classes = skill_labels.argmax(dim=1)

    # Debug prints
    print(f"\nFactor logits (test subset): {factor_predictions.shape}")
    print(f"IC logits (test subset): {ic_predictions.shape}")
    print(f"Skill logits (test subset): {skill_predictions.shape}")

    # Compute accuracies
    factor_accuracy = factor_pred_classes.eq(factor_true_classes).float().mean().item()
    ic_accuracy = ic_pred_classes.eq(ic_true_classes).float().mean().item()
    skill_accuracy = skill_pred_classes.eq(skill_true_classes).float().mean().item()

    # Numpy conversion for classification_report/confusion_matrix
    y_true_factors = factor_true_classes.cpu().numpy()
    y_pred_factors = factor_pred_classes.cpu().numpy()
    y_true_ic = ic_true_classes.cpu().numpy()
    y_pred_ic = ic_pred_classes.cpu().numpy()
    y_true_skills = skill_true_classes.cpu().numpy()
    y_pred_skills = skill_pred_classes.cpu().numpy()
    
    micro_f1_factors = f1_score(y_true_factors, y_pred_factors, average='micro')
    macro_f1_factors = f1_score(y_true_factors, y_pred_factors, average='macro')
    micro_f1_ic = f1_score(y_true_ic, y_pred_ic, average='micro')
    macro_f1_ic = f1_score(y_true_ic, y_pred_ic, average='macro')
    micro_f1_skills = f1_score(y_true_skills, y_pred_skills, average='micro')
    macro_f1_skills = f1_score(y_true_skills, y_pred_skills, average='macro')
    
    k_values = list(range(1, 11))
    emb_metrics = evaluate_embeddings(eval_graph_embds[test_indices], test_labels, k_values=k_values)
    print("\nGraph Embedding Evaluation:")
    for k in k_values:
        print(f"P@{k}={emb_metrics[f'P@{k}']:.4f}, R@{k}={emb_metrics[f'R@{k}']:.4f}")
    print(f"MRR={emb_metrics['MRR']:.4f}")
    
    # Evaluate the embeddings for the common factors
    emb_metrics = evaluate_embeddings(eval_graph_embds[test_indices], factor_labels, k_values=k_values)
    print("\nCommon Factors Embedding Evaluation:")
    for k in k_values:
        print(f"P@{k}={emb_metrics[f'P@{k}']:.4f}, R@{k}={emb_metrics[f'R@{k}']:.4f}")
        
    # Evaluate the embeddings for the intervention concepts
    emb_metrics = evaluate_embeddings(eval_graph_embds[test_indices], intervention_concepts_labels, k_values=k_values)
    print("\nIntervention Concepts Embedding Evaluation:")
    for k in k_values:
        print(f"P@{k}={emb_metrics[f'P@{k}']:.4f}, R@{k}={emb_metrics[f'R@{k}']:.4f}")
        
    # Evaluate the embeddings for the skills
    emb_metrics = evaluate_embeddings(eval_graph_embds[test_indices], skill_labels, k_values=k_values)
    print("\nSkills Embedding Evaluation:")
    for k in k_values:
        print(f"P@{k}={emb_metrics[f'P@{k}']:.4f}, R@{k}={emb_metrics[f'R@{k}']:.4f}")

    factor_names = ['Bond', 'Goal Alignment', 'Task Agreement', 'Neutral']
    ic_names = ['EAR', 'CP', 'N']
    skill_names = [
        'Reflective Listening', 'Genuineness', 'Validation',
        'Affirmation', 'Respect for Autonomy', 'Asking for Permission',
        'Open-ended Question', 'Neutral'
    ]

    # Classification reports
    try:
        factor_report = classification_report(
            y_true_factors,
            y_pred_factors,
            target_names=factor_names,
            output_dict=True,
            zero_division=0
        )

        ic_report = classification_report(
            y_true_ic,
            y_pred_ic,
            target_names=ic_names,
            output_dict=True,
            zero_division=0
        )

        skill_report = classification_report(
            y_true_skills,
            y_pred_skills,
            target_names=skill_names,
            output_dict=True,
            zero_division=0
        )
    except ValueError as e:
        print(f"Warning: Could not generate classification report: {e}")
        factor_report = {}
        ic_report = {}
        skill_report = {}

    # Confusion matrices
    try:
        factor_conf_matrix = confusion_matrix(y_true_factors, y_pred_factors)
        ic_conf_matrix = confusion_matrix(y_true_ic, y_pred_ic)
        skill_conf_matrix = confusion_matrix(y_true_skills, y_pred_skills)
    except ValueError as e:
        print(f"Warning: Could not generate confusion matrix: {e}")
        factor_conf_matrix = np.array([[0]])
        ic_conf_matrix = np.array([[0]])
        skill_conf_matrix = np.array([[0]])

    return {
        'factor_accuracy': factor_accuracy,
        'ic_accuracy': ic_accuracy,
        'skill_accuracy': skill_accuracy,
        'factor_classification_report': factor_report,
        'ic_classification_report': ic_report,
        'skill_classification_report': skill_report,
        'factor_confusion_matrix': factor_conf_matrix,
        'ic_confusion_matrix': ic_conf_matrix,
        'skill_confusion_matrix': skill_conf_matrix,
        'num_test_examples': n_test,
        'factor_micro_f1': micro_f1_factors,
        'factor_macro_f1': macro_f1_factors,
        'ic_micro_f1': micro_f1_ic,
        'ic_macro_f1': macro_f1_ic,
        'skill_micro_f1': micro_f1_skills,
        'skill_macro_f1': macro_f1_skills
    }


def predict_new_text(
    model: CFiCS,
    dataset: TherapeuticDataset,
    text: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Predict common factors and skills for new therapeutic text example
    """
    model.eval()
    with torch.no_grad():
        # Encode new text
        text_features = dataset.encode_new_text(text)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        
        # Forward pass w/o message passing (isolated)
        factor_probs, ic_probs, skill_probs = model(
            text_features,
            edge_index=None,
            return_logits=False  # Return softmax probs directly
        )
        
        # Class names
        factor_names = ['Bond', 'Goal Alignment', 'Task Agreement', 'Neutral']
        ic_names = ['EAR', 'CP', 'N']
        skill_names = [
            'Reflective Listening', 'Genuineness', 'Validation', 
            'Affirmation', 'Respect for Autonomy', 'Asking for Permission', 
            'Open-ended Question', 'Neutral'
        ]
        
        # Squeeze out batch dimension
        factor_probs = factor_probs.squeeze(0)
        ic_probs = ic_probs.squeeze(0)
        skill_probs = skill_probs.squeeze(0)
        
        factor_predictions = {
            name: prob.item() for name, prob in zip(factor_names, factor_probs)
        }
        ic_predictions = {
            name: prob.item() for name, prob in zip(ic_names, ic_probs)
        }
        skill_predictions = {
            name: prob.item() for name, prob in zip(skill_names, skill_probs)
        }
        
        return factor_predictions, ic_predictions, skill_predictions

def classify_unseen_texts(
    model,
    texts: List[str],
    encode_fn,
    device: torch.device = torch.device("cpu")
):
    """
    Classify an array of texts using the trained GNN-based model.
    
    Args:
        model (CFiCS): Your trained model.
        texts (List[str]): An array of new text samples to classify.
        encode_fn (callable): A function that takes a string and returns a BERT embedding [hidden_dim].
        device (torch.device): CPU or GPU device.
    
    Returns:
        predictions (torch.Tensor): Class predictions for each text [num_texts].
        logits (torch.Tensor): Raw logits [num_texts, num_classes].
    """
    model.eval()  # set to inference mode
    
    # 1. Encode all texts into a batched tensor
    embeddings_list = []
    for text in texts:
        emb = encode_fn(text).to(device)  # shape [hidden_dim]
        embeddings_list.append(emb)
    # shape: [N, hidden_dim]
    x_text = torch.stack(embeddings_list, dim=0)
    
    # 2. We have no known edges for these new nodes
    #    Create an empty edge_index => shape [2, 0]
    #    or a minimal adjacency structure. We'll do empty:
    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # 3. Let x_graph = x_text for inductive logic
    #    i.e. we feed the same features to the GNN aggregator
    x_graph = x_text
    
    with torch.no_grad():
        # forward pass through the aggregator
        logits, graph_embds = model(
            x_text=x_text,
            x_graph=x_graph,
            edge_index=edge_index,
        )
    
    return logits
    
def main():
    # 1. Load config
    config = load_config("config/config.yml")  # Adjust path as needed

    # 2. Load data
    data, dataset = load_data(config['data']['filepath'], config['data']['model_name'], config['data']['skip_text_embedding'])
    
    # 3. Initialize model from config
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
    
    try:
        # 4. Load trained weights
        model.load_state_dict(torch.load('CFiCS_sage_emilyalsentzer_Bio_ClinicalBERT.pth'))

        metrics = evaluate_model(
            model, 
            data,
        )

        if 'factor_accuracy' in metrics:
            print("\nModel Evaluation Results:")
            print(f"Factor Accuracy: {metrics['factor_accuracy']:.4f}")
            print(f"Intervention Concepts Accuracy: {metrics['ic_accuracy']:.4f}")
            print(f"Skill Accuracy: {metrics['skill_accuracy']:.4f}")
            
            # micro/macro F1 scores
            print(f"Factor Micro F1: {metrics['factor_micro_f1']:.4f}")
            print(f"Factor Macro F1: {metrics['factor_macro_f1']:.4f}")
            print(f"Intervention Concepts Micro F1: {metrics['ic_micro_f1']:.4f}")
            print(f"Intervention Concepts Macro F1: {metrics['ic_macro_f1']:.4f}")
            print(f"Skill Micro F1: {metrics['skill_micro_f1']:.4f}")
            print(f"Skill Macro F1: {metrics['skill_macro_f1']:.4f}")

            print("\nCommon Factors Classification Report:")
            for class_name, metrics_dict in metrics['factor_classification_report'].items():
                if isinstance(metrics_dict, dict):
                    print(f"\n{class_name}:")
                    print(f"Precision: {metrics_dict['precision']:.4f}")
                    print(f"Recall: {metrics_dict['recall']:.4f}")
                    print(f"F1-score: {metrics_dict['f1-score']:.4f}")

            print("\nIntervention Concepts Classification Report:")
            for class_name, metrics_dict in metrics['ic_classification_report'].items():
                if isinstance(metrics_dict, dict):
                    print(f"\n{class_name}:")
                    print(f"Precision: {metrics_dict['precision']:.4f}")
                    print(f"Recall: {metrics_dict['recall']:.4f}")
                    print(f"F1-score: {metrics_dict['f1-score']:.4f}")

            print("\nSkills Classification Report:")
            for class_name, metrics_dict in metrics['skill_classification_report'].items():
                if isinstance(metrics_dict, dict):
                    print(f"\n{class_name}:")
                    print(f"Precision: {metrics_dict['precision']:.4f}")
                    print(f"Recall: {metrics_dict['recall']:.4f}")
                    print(f"F1-score: {metrics_dict['f1-score']:.4f}")

            print("\nCommon Factors Confusion Matrix:")
            print(metrics['factor_confusion_matrix'])

            print("\nIntervention Concepts Confusion Matrix:")
            print(metrics.keys())
            print(metrics['ic_confusion_matrix'])

            print("\nSkills Confusion Matrix:")
            print(metrics['skill_confusion_matrix'])
        
        # 7. Predict new text examples
        print("\nPredicting New Examples:")
        unseen_example_texts = [
            "You mentioned wanting to improve your communication with your partner. Can you tell me more about what that would look like for you?'",
            "What specific changes would you like to see in your relationship by the end of therapy?",
            "Let's break down this new communication technique into smaller steps we can practice.",
            "And so let's just start right there and tell me what brought you here today?", 
            "It used to… And the people that you're working with what you're doing, uh… you don't feel good about that. Sometimes is that the way…", 
            "No usage here.",
            "I’d like to try a mindfulness exercise that might help you manage your anxiety. Would you be open to giving it a try?", # TA,Task agreement,3,,,,AP,Asking for Permission,13
            "I hear you, and it must feel awful to be criticized in that way." # B,Bond,1,EAR,"Empathy, Acceptance and Positive Regard",5,V,Validation,10
        ]
        
        # for text in unseen_example_texts:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # trained_model.to(device)
            
        unseen_logits = classify_unseen_texts(
            model=model,
            texts=unseen_example_texts,
            encode_fn=dataset.encode_new_text,
            # device=device
        )
        print(f"\nUnseen logits shape: {unseen_logits.shape}")
        f_logits =  unseen_logits[:, :4]
        ic_logits =  unseen_logits[:, 4:7]
        s_logits = unseen_logits[:, 7:]
        
        # get the class predictions and probabilities
        f_preds = F.softmax(f_logits, dim=-1)
        ic_preds = F.softmax(ic_logits, dim=-1)
        s_preds = F.softmax(s_logits, dim=-1)
        

        # print(f"\nText: {unseen_example_texts}")
        print(f"CF softmax: {f_preds}\nIC softmax: {ic_preds}\nSkills softmax: {s_preds}")
        
        for i, text in enumerate(unseen_example_texts):
            print(f"\nText: {text}")
            
            print("\nPredicted Common Factors:")
            for f, prob in enumerate(f_preds[i]):
                print(f"{f}: {prob:.4f}")
                
            print("\nPredicted Intervention Concepts:")
            for ic, prob in enumerate(ic_preds[i]):
                print(f"{ic}: {prob:.4f}")
                
            print("\nPredicted Skills:")
            for s, prob in enumerate(s_preds[i]):
                print(f"{s}: {prob:.4f}") #
                
    except FileNotFoundError:
        print("Error: Could not find trained model file (enhanced_therapeutic_gnn.pth)")
        print("Please train the model first using train.py")

if __name__ == '__main__':
    main()