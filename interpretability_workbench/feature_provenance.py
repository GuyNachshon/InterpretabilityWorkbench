"""
Feature provenance analysis for cross-layer feature relationships
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from collections import defaultdict


class FeatureProvenanceAnalyzer:
    """Analyzes relationships between features across layers"""
    
    def __init__(self, model, sae_models: Dict[int, Any], tokenizer, activation_data_paths: Dict[int, str]):
        self.model = model
        self.sae_models = sae_models
        self.tokenizer = tokenizer
        self.activation_data_paths = activation_data_paths
        self.feature_relationships = {}
        
    def parse_feature_id(self, feature_id: str) -> Tuple[int, int]:
        """Parse feature ID to get layer and feature indices"""
        # Format: layer_X_feature_Y
        parts = feature_id.split('_')
        if len(parts) >= 4:
            layer_idx = int(parts[1])
            feature_idx = int(parts[3])
            return layer_idx, feature_idx
        else:
            raise ValueError(f"Invalid feature ID format: {feature_id}")
    
    def get_feature_vector(self, layer_idx: int, feature_idx: int) -> torch.Tensor:
        """Get feature vector from SAE encoder"""
        if layer_idx not in self.sae_models:
            raise ValueError(f"No SAE model for layer {layer_idx}")
        
        sae = self.sae_models[layer_idx]
        return sae.encoder.weight[feature_idx].detach()
    
    def find_related_features(self, feature_vector: torch.Tensor, target_layer: int, 
                            similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Find features in target layer that are related to the given feature vector"""
        if target_layer not in self.sae_models:
            return []
        
        sae = self.sae_models[target_layer]
        encoder_weights = sae.encoder.weight.detach()  # [latent_dim, input_dim]
        
        # Compute cosine similarity between feature vector and all features in target layer
        similarities = []
        for feature_idx in range(encoder_weights.shape[0]):
            target_feature = encoder_weights[feature_idx]
            
            # Normalize vectors for cosine similarity
            feature_norm = torch.norm(feature_vector)
            target_norm = torch.norm(target_feature)
            
            if feature_norm > 0 and target_norm > 0:
                similarity = torch.dot(feature_vector, target_feature) / (feature_norm * target_norm)
                similarities.append((feature_idx, similarity.item()))
        
        # Sort by similarity and filter by threshold
        similarities.sort(key=lambda x: x[1], reverse=True)
        related_features = []
        
        for feature_idx, similarity in similarities:
            if similarity >= similarity_threshold:
                related_features.append({
                    'feature_id': f'layer_{target_layer}_feature_{feature_idx}',
                    'layer_idx': target_layer,
                    'feature_idx': feature_idx,
                    'similarity': similarity,
                    'strength': similarity  # For graph visualization
                })
        
        return related_features
    
    def analyze_feature_relationships(self, feature_id: str, upstream_layers: int = 2, 
                                   downstream_layers: int = 1) -> Dict[str, Any]:
        """Analyze relationships between features across layers"""
        layer_idx, feature_idx = self.parse_feature_id(feature_id)
        
        # Get feature vector
        feature_vector = self.get_feature_vector(layer_idx, feature_idx)
        
        # Analyze upstream layers
        upstream_relationships = []
        for upstream_layer in range(max(0, layer_idx - upstream_layers), layer_idx):
            if upstream_layer in self.sae_models:
                upstream_features = self.find_related_features(feature_vector, upstream_layer)
                upstream_relationships.extend(upstream_features)
        
        # Analyze downstream layers
        downstream_relationships = []
        max_layer = max(self.sae_models.keys()) if self.sae_models else layer_idx
        for downstream_layer in range(layer_idx + 1, min(layer_idx + downstream_layers + 1, max_layer + 1)):
            if downstream_layer in self.sae_models:
                downstream_features = self.find_related_features(feature_vector, downstream_layer)
                downstream_relationships.extend(downstream_features)
        
        # Create graph data
        nodes = []
        edges = []
        
        # Add current feature
        nodes.append({
            'id': feature_id,
            'layer': layer_idx,
            'type': 'current',
            'label': f'Feature {feature_idx}',
            'x': layer_idx * 200,
            'y': feature_idx % 10 * 50
        })
        
        # Add upstream features
        for rel in upstream_relationships:
            node_id = rel['feature_id']
            nodes.append({
                'id': node_id,
                'layer': rel['layer_idx'],
                'type': 'upstream',
                'label': f'Feature {rel["feature_idx"]}',
                'x': rel['layer_idx'] * 200,
                'y': rel['feature_idx'] % 10 * 50
            })
            
            edges.append({
                'from': node_id,
                'to': feature_id,
                'strength': rel['strength'],
                'type': 'upstream'
            })
        
        # Add downstream features
        for rel in downstream_relationships:
            node_id = rel['feature_id']
            nodes.append({
                'id': node_id,
                'layer': rel['layer_idx'],
                'type': 'downstream',
                'label': f'Feature {rel["feature_idx"]}',
                'x': rel['layer_idx'] * 200,
                'y': rel['feature_idx'] % 10 * 50
            })
            
            edges.append({
                'from': feature_id,
                'to': node_id,
                'strength': rel['strength'],
                'type': 'downstream'
            })
        
        return {
            'feature_id': feature_id,
            'layer_idx': layer_idx,
            'feature_idx': feature_idx,
            'nodes': nodes,
            'edges': edges,
            'upstream_relationships': upstream_relationships,
            'downstream_relationships': downstream_relationships,
            'total_relationships': len(upstream_relationships) + len(downstream_relationships)
        }
    
    def analyze_feature_clusters(self, layer_idx: int, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """Find clusters of similar features within a layer"""
        if layer_idx not in self.sae_models:
            return []
        
        sae = self.sae_models[layer_idx]
        encoder_weights = sae.encoder.weight.detach()
        
        # Compute similarity matrix
        num_features = encoder_weights.shape[0]
        similarity_matrix = torch.zeros(num_features, num_features)
        
        for i in range(num_features):
            for j in range(i + 1, num_features):
                feature_i = encoder_weights[i]
                feature_j = encoder_weights[j]
                
                norm_i = torch.norm(feature_i)
                norm_j = torch.norm(feature_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = torch.dot(feature_i, feature_j) / (norm_i * norm_j)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Find clusters using simple threshold-based clustering
        clusters = []
        used_features = set()
        
        for i in range(num_features):
            if i in used_features:
                continue
            
            # Start new cluster
            cluster = [i]
            used_features.add(i)
            
            # Find similar features
            for j in range(num_features):
                if j not in used_features and similarity_matrix[i, j] >= min_similarity:
                    cluster.append(j)
                    used_features.add(j)
            
            if len(cluster) > 1:  # Only keep clusters with multiple features
                clusters.append({
                    'cluster_id': f'layer_{layer_idx}_cluster_{len(clusters)}',
                    'layer_idx': layer_idx,
                    'feature_indices': cluster,
                    'size': len(cluster),
                    'avg_similarity': similarity_matrix[i, cluster].mean().item(),
                    'features': [f'layer_{layer_idx}_feature_{idx}' for idx in cluster]
                })
        
        return clusters
    
    def get_feature_activation_pattern(self, feature_id: str, num_samples: int = 100) -> Dict[str, Any]:
        """Analyze activation pattern of a feature across different contexts"""
        layer_idx, feature_idx = self.parse_feature_id(feature_id)
        
        if layer_idx not in self.activation_data_paths:
            return {'error': 'No activation data available for this layer'}
        
        # This would require loading activation data and analyzing patterns
        # For now, return a placeholder structure
        return {
            'feature_id': feature_id,
            'activation_pattern': {
                'mean_activation': 0.0,
                'std_activation': 0.0,
                'max_activation': 0.0,
                'sparsity': 0.0,
                'activation_distribution': []
            },
            'context_patterns': [],
            'sample_size': num_samples
        }
    
    def export_provenance_data(self, output_path: str):
        """Export all provenance analysis data"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = {
            'feature_relationships': self.feature_relationships,
            'layers_analyzed': list(self.sae_models.keys()),
            'export_timestamp': str(torch.tensor(0).item())  # Placeholder for timestamp
        }
        
        with open(output_path / 'provenance_data.json', 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"Provenance data exported to {output_path}")


def create_provenance_analyzer(model, sae_models, tokenizer, activation_data_paths):
    """Factory function to create a FeatureProvenanceAnalyzer"""
    return FeatureProvenanceAnalyzer(model, sae_models, tokenizer, activation_data_paths) 