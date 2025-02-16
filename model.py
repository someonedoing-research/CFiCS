import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CFiCS(nn.Module):
    """
    A flexible GNN-based classification model that learns a joint embedding of:
      - Text embeddings (x_text) [BERT or other embeddings]
      - Graph-based embeddings (x_graph) [node features]

    The model supports different graph neural network backbones such as GraphSAGE, GCN, or GAT.
    The final joint embedding is formed by concatenating the text and graph embeddings, and a linear
    classifier produces the final logits.
    """
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_channels: int,
        num_common_factors: int = 3,
        num_intervention_concepts: int = 3,
        num_skills: int = 8,
        num_layers: int = 2,
        dropout: float = 0.5,
        graph_model_type: str = "sage",  # "sage", "gcn", or "gat"
        aggregator: str = "mean"         # Used only for GraphSAGE
    ):
        super().__init__()
        
        # Choose the convolution class and its kwargs based on graph_model_type
        if graph_model_type.lower() == "sage":
            from torch_geometric.nn import SAGEConv
            conv_class = SAGEConv
            conv_kwargs = {"aggr": aggregator}
        elif graph_model_type.lower() == "gcn":
            from torch_geometric.nn import GCNConv
            conv_class = GCNConv
            conv_kwargs = {}  # GCNConv does not require an aggregator parameter
        elif graph_model_type.lower() == "gat":
            from torch_geometric.nn import GATConv
            conv_class = GATConv
            conv_kwargs = {"heads": 1}  # Default to 1 head; you could expose this as a parameter
        else:
            raise ValueError(
                f"Unsupported graph model type: {graph_model_type}. "
                "Please choose one of 'sage', 'gcn', or 'gat'."
            )
        
        # -- 1) Build a stack of graph convolution layers --
        self.convs = nn.ModuleList()
        # First layer: from graph_dim to hidden_channels
        self.convs.append(conv_class(graph_dim, hidden_channels, **conv_kwargs))
        for _ in range(num_layers - 1):
            # Subsequent layers: hidden_channels -> hidden_channels
            self.convs.append(conv_class(hidden_channels, hidden_channels, **conv_kwargs))
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Total number of classification outputs:
        total_classes = num_common_factors + num_intervention_concepts + num_skills

        # -- 2) Final classifier --
        # The final input dimension is text_dim (from text embedding) plus hidden_channels (from GNN)
        self.classifier = nn.Linear(text_dim + hidden_channels, total_classes)
        
        # Save configuration information for reference (optional)
        self.graph_model_type = graph_model_type.lower()
        self.aggregator = aggregator

    def forward(
        self,
        x_text: torch.Tensor,
        x_graph: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
          1) Processes x_graph through the chosen GNN backbone to obtain a graph embedding.
          2) Concatenates the text embeddings (x_text) with the graph embeddings.
          3) Passes the joint embedding to a linear classifier.

        Args:
            x_text: [num_nodes, text_dim] - Text (e.g., BERT) embeddings.
            x_graph: [num_nodes, graph_dim] - Node features for graph processing.
            edge_index: [2, num_edges] - Graph connectivity.

        Returns:
            (logits, graph_emb):
              - logits: [num_nodes, total_classes] (raw classification logits)
              - graph_emb: [num_nodes, hidden_channels] (learned graph embeddings)
        """
        # Process node features through the graph convolution layers
        graph_emb = x_graph
        for conv in self.convs:
            graph_emb = conv(graph_emb, edge_index)
            graph_emb = F.relu(graph_emb)
            graph_emb = F.dropout(graph_emb, p=self.dropout, training=self.training)
        
        # Concatenate text and graph embeddings to form the joint embedding
        joint_emb = torch.cat([x_text, graph_emb], dim=-1)
        
        # Pass through the classifier
        logits = self.classifier(joint_emb)
        
        return logits, graph_emb