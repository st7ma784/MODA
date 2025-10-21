"""Neural Network for Multi-Modal Signal Diagnosis

Implements a graph-aware neural network that understands relationships
between different signal analysis parameters for automated diagnosis.

Architecture:
1. Multi-Modal Feature Processing (separate pathways per analysis type)
2. Cross-Modal Attention (learns relationships between modalities)
3. Graph Neural Network (encodes physiological parameter relationships)
4. Temporal LSTM (if temporal context is available)
5. Final Classification/Regression head
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural network features disabled.")


if TORCH_AVAILABLE:

    # ==================== Multi-Modal Attention ====================

    class CrossModalAttention(nn.Module):
        """
        Cross-modal attention mechanism to learn relationships between
        different analysis modalities
        """
        def __init__(self, feature_dim: int, n_heads: int = 4):
            super().__init__()
            self.n_heads = n_heads
            self.feature_dim = feature_dim

            self.query = nn.Linear(feature_dim, feature_dim)
            self.key = nn.Linear(feature_dim, feature_dim)
            self.value = nn.Linear(feature_dim, feature_dim)
            self.out = nn.Linear(feature_dim, feature_dim)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
            """
            Args:
                x: (batch, n_modalities, feature_dim)
                mask: (batch, n_modalities) - which modalities are present

            Returns:
                attended_features: (batch, n_modalities, feature_dim)
            """
            Q = self.query(x)  # (batch, n_mod, feat)
            K = self.key(x)
            V = self.value(x)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.feature_dim)

            if mask is not None:
                # Mask out absent modalities
                mask_expanded = mask.unsqueeze(1).expand_as(scores)
                scores = scores.masked_fill(~mask_expanded, -1e9)

            attention = F.softmax(scores, dim=-1)
            attended = torch.matmul(attention, V)

            output = self.out(attended)
            return output, attention


    # ==================== Graph Neural Network ====================

    class ParameterRelationshipGraph(nn.Module):
        """
        Graph Neural Network that encodes known physiological/physical
        relationships between signal parameters

        Example relationships:
        - Frequency <-> Power (spectral)
        - Phase <-> Frequency (instantaneous)
        - Low freq power <-> High freq power (coupling)
        - Coherence <-> Phase difference
        """
        def __init__(self, feature_groups: Dict[str, List[int]],
                     hidden_dim: int = 64):
            super().__init__()
            self.feature_groups = feature_groups
            self.hidden_dim = hidden_dim

            # Create learnable adjacency matrix
            total_groups = len(feature_groups)
            self.edge_weights = nn.Parameter(torch.randn(total_groups, total_groups) * 0.1)

            # Message passing layers
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.update_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, group_features: torch.Tensor):
            """
            Args:
                group_features: (batch, n_groups, hidden_dim)

            Returns:
                updated_features: (batch, n_groups, hidden_dim)
            """
            batch_size, n_groups, _ = group_features.shape

            # Normalize edge weights to get adjacency
            adj = F.softmax(self.edge_weights, dim=-1)

            # Message passing
            messages = []
            for i in range(n_groups):
                # Aggregate messages from neighbors
                neighbor_feats = group_features  # (batch, n_groups, hidden)
                edge_weights = adj[i:i+1, :].unsqueeze(0)  # (1, 1, n_groups)

                # Create messages from each neighbor
                source_feats = group_features[:, i:i+1, :].expand(-1, n_groups, -1)
                combined = torch.cat([source_feats, neighbor_feats], dim=-1)
                message = self.message_mlp(combined)  # (batch, n_groups, hidden)

                # Weighted aggregation
                aggregated = torch.sum(message * edge_weights.transpose(1, 2), dim=1, keepdim=True)
                messages.append(aggregated)

            messages = torch.cat(messages, dim=1)  # (batch, n_groups, hidden)

            # Update node features
            combined_update = torch.cat([group_features, messages], dim=-1)
            updated = self.update_mlp(combined_update)

            return updated + group_features  # Residual connection


    # ==================== Main Diagnosis Network ====================

    class MultiModalDiagnosisNetwork(nn.Module):
        """
        Complete multi-modal diagnosis network with:
        - Modality-specific encoding
        - Cross-modal attention
        - Parameter relationship graph
        - Final classification/regression
        """
        def __init__(self,
                     feature_dims: Dict[str, int],
                     feature_groups: Dict[str, List[int]],
                     n_classes: int = 1,
                     hidden_dim: int = 128,
                     n_attention_heads: int = 4,
                     dropout: float = 0.3):
            """
            Args:
                feature_dims: Dict mapping modality name to feature dimension
                    e.g., {'spectral': 25, 'phase': 15, ...}
                feature_groups: Dict mapping parameter groups to feature indices
                    e.g., {'frequency_features': [0, 5, 12], 'power_features': [1, 3, 8]}
                n_classes: Number of output classes (1 for regression, >1 for classification)
                hidden_dim: Hidden layer dimension
                n_attention_heads: Number of attention heads
                dropout: Dropout probability
            """
            super().__init__()

            self.feature_dims = feature_dims
            self.feature_groups = feature_groups
            self.n_classes = n_classes
            self.hidden_dim = hidden_dim

            # Modality-specific encoders
            self.modality_encoders = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                for name, dim in feature_dims.items()
            })

            # Cross-modal attention
            self.cross_modal_attention = CrossModalAttention(hidden_dim, n_attention_heads)

            # Parameter relationship graph
            self.relationship_graph = ParameterRelationshipGraph(feature_groups, hidden_dim)

            # Final classification/regression head
            total_modalities = len(feature_dims)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * total_modalities, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )

        def forward(self, features: Dict[str, torch.Tensor],
                    return_attention: bool = False):
            """
            Args:
                features: Dict mapping modality names to feature tensors
                    Each tensor: (batch, feature_dim)
                return_attention: Whether to return attention weights

            Returns:
                output: (batch, n_classes)
                attention_weights: (batch, n_modalities, n_modalities) if return_attention=True
            """
            batch_size = list(features.values())[0].shape[0]

            # Encode each modality
            encoded = []
            modality_names = []
            for name in sorted(self.feature_dims.keys()):
                if name in features:
                    enc = self.modality_encoders[name](features[name])
                    encoded.append(enc.unsqueeze(1))  # (batch, 1, hidden)
                    modality_names.append(name)

            if len(encoded) == 0:
                raise ValueError("No modalities provided")

            # Stack modalities
            encoded = torch.cat(encoded, dim=1)  # (batch, n_modalities, hidden)

            # Cross-modal attention
            attended, attention_weights = self.cross_modal_attention(encoded)

            # Parameter relationship graph (placeholder - needs proper grouping)
            # For now, treat each modality as a group
            graph_output = self.relationship_graph(attended)

            # Flatten and classify
            flattened = graph_output.reshape(batch_size, -1)
            output = self.classifier(flattened)

            if return_attention:
                return output, attention_weights
            return output

        def get_feature_importance(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
            """
            Compute feature importance using integrated gradients

            Args:
                features: Input features

            Returns:
                importance_scores: Dict mapping modality names to importance scores
            """
            self.eval()
            importance = {}

            for name in features.keys():
                feat = features[name].requires_grad_(True)
                baseline = torch.zeros_like(feat)

                # Compute gradients at multiple interpolation points
                n_steps = 20
                integrated_grads = torch.zeros_like(feat)

                for alpha in np.linspace(0, 1, n_steps):
                    interpolated = baseline + alpha * (feat - baseline)
                    output = self.forward({name: interpolated})

                    if self.n_classes == 1:
                        output_sum = output.sum()
                    else:
                        output_sum = output.max(dim=1)[0].sum()

                    grads = torch.autograd.grad(output_sum, interpolated)[0]
                    integrated_grads += grads / n_steps

                # Importance is the magnitude of integrated gradients
                importance[name] = float(torch.abs(integrated_grads).mean())

            return importance


    # ==================== Training Utilities ====================

    class DiagnosisTrainer:
        """
        Trainer for the diagnosis network
        """
        def __init__(self, model: MultiModalDiagnosisNetwork,
                     learning_rate: float = 1e-3,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
            self.model = model.to(device)
            self.device = device
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Determine loss function
            if model.n_classes == 1:
                self.criterion = nn.MSELoss()  # Regression
                self.task_type = 'regression'
            else:
                self.criterion = nn.CrossEntropyLoss()  # Classification
                self.task_type = 'classification'

        def train_epoch(self, data_loader):
            """Train for one epoch"""
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch in data_loader:
                features, labels = batch
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)

                # Compute loss
                if self.task_type == 'regression':
                    loss = self.criterion(outputs.squeeze(), labels)
                else:
                    loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            return total_loss / max(n_batches, 1)

        def evaluate(self, data_loader):
            """Evaluate on validation/test set"""
            self.model.eval()
            total_loss = 0
            n_batches = 0
            predictions = []
            true_labels = []

            with torch.no_grad():
                for batch in data_loader:
                    features, labels = batch
                    features = {k: v.to(self.device) for k, v in features.items()}
                    labels = labels.to(self.device)

                    outputs = self.model(features)

                    if self.task_type == 'regression':
                        loss = self.criterion(outputs.squeeze(), labels)
                        predictions.append(outputs.squeeze().cpu().numpy())
                    else:
                        loss = self.criterion(outputs, labels)
                        predictions.append(outputs.argmax(dim=1).cpu().numpy())

                    true_labels.append(labels.cpu().numpy())
                    total_loss += loss.item()
                    n_batches += 1

            predictions = np.concatenate(predictions)
            true_labels = np.concatenate(true_labels)

            metrics = {
                'loss': total_loss / max(n_batches, 1),
                'predictions': predictions,
                'labels': true_labels
            }

            # Add task-specific metrics
            if self.task_type == 'classification':
                accuracy = np.mean(predictions == true_labels)
                metrics['accuracy'] = accuracy
            else:
                mse = np.mean((predictions - true_labels) ** 2)
                metrics['mse'] = mse
                metrics['rmse'] = np.sqrt(mse)

            return metrics


# ==================== Helper Functions ====================

def create_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Automatically group features based on naming patterns

    Args:
        feature_names: List of feature names

    Returns:
        feature_groups: Dict mapping group names to feature indices
    """
    groups = {
        'frequency': [],
        'power': [],
        'amplitude': [],
        'phase': [],
        'entropy': [],
        'coherence': [],
        'coupling': [],
        'temporal': [],
        'spectral': []
    }

    for idx, name in enumerate(feature_names):
        name_lower = name.lower()

        if 'freq' in name_lower:
            groups['frequency'].append(idx)
        if 'power' in name_lower or 'energy' in name_lower:
            groups['power'].append(idx)
        if 'amp' in name_lower:
            groups['amplitude'].append(idx)
        if 'phase' in name_lower:
            groups['phase'].append(idx)
        if 'entropy' in name_lower:
            groups['entropy'].append(idx)
        if 'coherence' in name_lower:
            groups['coherence'].append(idx)
        if 'coupling' in name_lower or 'bispec' in name_lower:
            groups['coupling'].append(idx)
        if 'temporal' in name_lower or 'time' in name_lower:
            groups['temporal'].append(idx)
        if 'spectral' in name_lower:
            groups['spectral'].append(idx)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    return groups


def create_diagnosis_model(feature_names: List[str],
                           n_classes: int = 1) -> Optional['MultiModalDiagnosisNetwork']:
    """
    Create a diagnosis model from feature names

    Args:
        feature_names: List of all feature names
        n_classes: Number of output classes

    Returns:
        model: MultiModalDiagnosisNetwork instance or None if torch unavailable
    """
    if not TORCH_AVAILABLE:
        return None

    # Group features by modality
    modality_dims = {}
    for name in feature_names:
        modality = name.split('_')[0]  # e.g., 'spectral_dominant_freq' -> 'spectral'
        if modality not in modality_dims:
            modality_dims[modality] = 0
        modality_dims[modality] += 1

    # Create feature groups
    feature_groups = create_feature_groups(feature_names)

    # Create model
    model = MultiModalDiagnosisNetwork(
        feature_dims=modality_dims,
        feature_groups=feature_groups,
        n_classes=n_classes,
        hidden_dim=128,
        n_attention_heads=4,
        dropout=0.3
    )

    return model
