# data_loading.py

import sys
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from example_data import get_node_data, get_edge_indices
from typing import Tuple

class TherapeuticDataset:
    def __init__(
        self,
        filepath: str,
        text_embedding_model_name: str = 'bert-base-uncased',
        skip_text_embedding: bool = False,
        hidden_dim: int = 768
    ):
        """
        Initialize the dataset with a given CSV filepath.
        - Loads all node data (root, factors, ICs, skills, examples).
        - If skip_text_embedding=True, no BERT usage; just produce a dummy vector for each text.
        """
        self.filepath = filepath
        
        # Load node data once and store it
        (self.root,
         self.factors,
         self.intervention_concepts,
         self.skills,
         self.examples) = get_node_data(self.filepath)
        
        self.skip_text_embedding = skip_text_embedding
        self.hidden_size = hidden_dim  # Default dimension for BERT or dummy if skipping

        if not skip_text_embedding:
            # Initialize BERT only if we're not skipping
            self.tokenizer = AutoTokenizer.from_pretrained(text_embedding_model_name)
            self.bert_model = AutoModel.from_pretrained(text_embedding_model_name)
            self.hidden_size = 768  # BERT base hidden size
        else:
            # No BERT: attributes for backward-compatibility
            self.tokenizer = None
            self.bert_model = None

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text:
          - If skip_text_embedding=False, use BERT (average-pooled last hidden state).
          - If skip_text_embedding=True, return a dummy feature vector of size self.hidden_size (e.g. zeros).
        """
        if self.skip_text_embedding:
            # Return a dummy vector for each text
            return torch.zeros(self.hidden_size)
        else:
            # BERT-based encoding
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            # Average-pool across the sequence length (dim=1)
            return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def encode_new_text(self, text: str) -> torch.Tensor:
        """Public method to encode arbitrary new text for inference."""
        return self._encode_text(text)
    
    def get_node_name(self, node_idx: int) -> str:
        """
        Retrieve the name of the node given its index.
        """
        if node_idx == 0:
            return "Root Node"
        elif 1 <= node_idx <= 4:
            return self.factors[node_idx - 1]['name']
        elif 5 <= node_idx <= 7:
            return self.intervention_concepts[node_idx - 5]['name']
        elif 8 <= node_idx <= 15:
            return self.skills[node_idx - 8]['name']
        else:
            return f"Example Node {node_idx}"
    
    def _create_node_features(self) -> torch.Tensor:
        """Create node features from text descriptions (or dummy features if skipping)."""
        node_features = []
        
        # 1. Process root, factors, ICs, skills
        for node in self.root + self.factors + self.intervention_concepts + self.skills:
            text = f"{node['name']}: {node['description']}"
            node_features.append(self._encode_text(text))
        
        # 2. Process examples (which primarily have 'text')
        for example in self.examples:
            node_features.append(self._encode_text(example['text']))
            
        return torch.stack(node_features)

    def _create_labels(self) -> torch.Tensor:
        """
        Create a combined labels tensor with:
          - CF: IDs 1..4 => columns 0..3
          - IC: IDs 5..7 => columns 4..6
          - Skill: IDs 8..15 => columns 7..14
          => total = 15 columns
        """
        num_examples = len(self.examples)

        num_factors = 4
        num_ics = 3
        num_skills = 8
        total_cols = num_factors + num_ics + num_skills  # 4+3+8=15

        labels = torch.zeros((num_examples, total_cols), dtype=torch.float)

        for i, example in enumerate(self.examples):
            factor_id = int(example.get('CF_id', 0))
            ic_id = int(example.get('IC_id', 0))
            skill_id = int(example.get('skill_id', 0))

            # 1) CF (1..4) => columns [0..3]
            if 1 <= factor_id <= 4:
                cf_col = factor_id - 1
                labels[i, cf_col] = 1.0

            # 2) IC (5..7) => columns [4..6]
            if ic_id == 5:
                labels[i, 4] = 1.0  # EAR
            elif ic_id == 6:
                labels[i, 5] = 1.0  # CP
            elif ic_id == 7:
                labels[i, 6] = 1.0  # Neutral (IC)

            # 3) Skill (8..15) => columns [7..14]
            if 8 <= skill_id <= 15:
                skill_col = (skill_id - 1)
                labels[i, skill_col] = 1.0

        return labels
    
    def create_pyg_data(self) -> Data:
        """
        Create a PyG Data object:
         - x: node features
         - edge_index: from get_edge_indices
         - y: multi-hot labels
         - train/val/test masks for example nodes
        """
        from torch_geometric.data import Data
        
        # 1. Node features (could be BERT or dummy)
        x = self._create_node_features()
        
        # 2. Edge indices
        edge_index = torch.tensor(get_edge_indices(self.filepath), dtype=torch.long).t()
        
        # 3. Labels
        labels = self._create_labels()
        
        # 4. Create train/val/test split
        num_examples = len(self.examples)
        example_start_idx = (len(self.root)
                             + len(self.factors)
                             + len(self.intervention_concepts)
                             + len(self.skills))

        indices = torch.randperm(num_examples)
        train_end = int(0.6 * num_examples)
        val_end = int(0.8 * num_examples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)

        # Mark which example nodes belong to each split
        train_mask[example_start_idx + train_idx] = True
        val_mask[example_start_idx + val_idx] = True
        test_mask[example_start_idx + test_idx] = True
        
        # Expand labels to all nodes (non-examples are all-zero)
        full_labels = torch.zeros((x.size(0), labels.size(1)), dtype=torch.float)
        full_labels[example_start_idx:example_start_idx + num_examples] = labels
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=full_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

def load_data(filepath: str, text_model_name: str, skip_text_embedding: str) -> Tuple[Data, TherapeuticDataset]:
    """
    Loads data given a config dict that may contain:
      cfg['data']['text_embedding_model_name']
      cfg['data'].get('skip_text_embedding', False)
      ...
    """
    dataset = TherapeuticDataset(
        filepath=filepath,
        text_embedding_model_name=text_model_name,
        skip_text_embedding=skip_text_embedding
    )
    data = dataset.create_pyg_data()
    return data, dataset


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/test.csv"
    
    print(f"Using CSV: {filepath}")
    
    # Example usage: skipping text
    # dataset = TherapeuticDataset(filepath, skip_text_embedding=True)
    dataset = TherapeuticDataset(filepath, skip_text_embedding=False)

    print("Testing _create_labels()...")
    labels = dataset._create_labels()
    print("Labels Tensor Shape:", labels.shape)
    print(labels)
    
    print("\nTesting create_pyg_data()...")
    pyg_data = dataset.create_pyg_data()
    print("PyG Data object created.")
    print("x shape:", pyg_data.x.shape)
    print("y shape:", pyg_data.y.shape)
    print("edge_index shape:", pyg_data.edge_index.shape)
    print("train_mask sum:", pyg_data.train_mask.sum())
    print("val_mask sum:", pyg_data.val_mask.sum())
    print("test_mask sum:", pyg_data.test_mask.sum())
    
    print("\nDone.")

if __name__ == "__main__":
    main()