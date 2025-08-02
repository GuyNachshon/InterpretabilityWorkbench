"""
Activation recording with HuggingFace hooks and streaming Parquet writes
"""
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def convo_to_prompt(dialogue: List[Dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=False
        )

    # ── manual template (Llama-2/3 style) ──────────────────────────
    prompt = []
    for turn in dialogue:
        role_tag = "user" if turn["role"] == "user" else "assistant"
        prompt.append(f"<|{role_tag}|>\n{turn['content']}<|end_of_text|>")
    return "<|begin_of_text|>\n" + "\n".join(prompt)


class ActivationRecorder:
    def __init__(
        self,
        model_name: str,
        layer_idx: int,
        output_path: str,
        max_samples: int = 10000,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        store_tokens: bool = True
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.output_path = Path(output_path)
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Storage for activations
        self.activations = []
        self.current_batch_acts = []
        
        # Load model and tokenizer
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
        # Set up hook
        self._register_hook()
        
    def _register_hook(self):
        """Register forward hook to capture activations"""
        def activation_hook(module, input, output):
            # Store activation from this layer
            if isinstance(output, tuple):
                activation = output[0]  # Usually the hidden states
            else:
                activation = output
                
            # Move to CPU and store
            self.current_batch_acts.append(activation.detach().cpu())
        
        # Get the target layer - try different model architectures
        target_layer = None
        
        # Try common layer patterns
        layer_patterns = [
            # Direct h attribute (GPT-J, GPT-2, etc.)
            (None, 'h'),
            # Llama-style: model.layers
            ('model', 'layers'),
            # GPT-style: transformer.h
            ('transformer', 'h'),
            # Direct layers attribute
            (None, 'layers'),
            # HuggingFace AutoModel: encoder.layer or decoder.layers
            ('encoder', 'layer'),
            ('decoder', 'layers'),
        ]
        
        for base_attr, layer_attr in layer_patterns:
            try:
                if base_attr is None:
                    layers = getattr(self.model, layer_attr)
                else:
                    base = getattr(self.model, base_attr)
                    layers = getattr(base, layer_attr)
                
                if hasattr(layers, '__len__') and len(layers) > self.layer_idx:
                    target_layer = layers[self.layer_idx]
                    break
            except AttributeError:
                continue
        
        # If we still haven't found it, try to inspect the model structure
        if target_layer is None:
            # Print model structure for debugging
            print(f"Model structure for {self.model_name}:")
            for name, module in self.model.named_children():
                print(f"  {name}: {type(module)}")
                if hasattr(module, '__len__'):
                    print(f"    (length: {len(module)})")
            
            raise ValueError(f"Cannot find layer {self.layer_idx} in model {self.model_name}. "
                           f"Please check the model architecture above.")
            
        self.hook_handle = target_layer.register_forward_hook(activation_hook)
        
    def _write_batch_to_parquet(self, batch_data: List[Dict[str, Any]]):
        """Write a batch of activation data to parquet file"""
        if not batch_data:
            return
            
        df = pd.DataFrame(batch_data)
        table = pa.Table.from_pandas(df)
        
        # Append to parquet file
        if self.output_path.exists():
            # Append mode
            existing_table = pq.read_table(self.output_path)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, self.output_path)
        else:
            # Create new file
            pq.write_table(table, self.output_path)

    # ---------------------------------------------------
    # main recording method ─────────────────────────────
    # ---------------------------------------------------
    def record(
            self,
            dataset_name: str = "openwebtext",
    ):
        print(f"Loading dataset {dataset_name} ...")

        is_chat = False
        if dataset_name == "openwebtext":
            try:
                dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
                text_key, is_chat = "text", False
            except Exception:
                print("openwebtext not found; falling back to wikitext-2")
                dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                       split="train", streaming=True)
                text_key, is_chat = "text", False

        elif dataset_name == "lmsys-chat-1m":
            dataset = load_dataset(
                "AarushSah/lmsys-chat-1m",
                split="train",
                streaming=True
            )
            text_key, is_chat = "conversation", True

        elif dataset_name == "the_pile":
            dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
            text_key, is_chat = "text", False
        else:  # generic HF dataset with a 'text' column
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            text_key, is_chat = "text", False

        # ───────────────────── record loop ─────────────────────────────
        processed_samples = 0
        batch_data = []
        stride = self.max_length  # slide window if text is longer

        print(f"Recording activations from layer {self.layer_idx} ...")
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Processing", total=self.max_samples):
                if processed_samples >= self.max_samples:
                    break

                # 1. get raw text --------------------------------------------------
                if is_chat:
                    dialogue = sample[text_key]
                    if not dialogue:
                        continue
                    raw_text = convo_to_prompt(dialogue, self.tokenizer)
                else:
                    raw_text = sample[text_key]

                if not raw_text or len(raw_text.strip()) < 10:
                    continue

                # 2. tokenise full prompt (no padding yet) ------------------------
                tokens = self.tokenizer(
                    raw_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids.to(self.device)

                # 3. slide window over long sequences -----------------------------
                for start in range(0, tokens.shape[1], stride):
                    sub_ids = tokens[:, start:start + stride]
                    attention_mask = torch.ones_like(sub_ids).to(self.device)

                    # clear activation bucket filled by your forward hook
                    self.current_batch_acts = []
                    _ = self.model(input_ids=sub_ids,
                                   attention_mask=attention_mask)

                    # pick up captured activation tensor
                    if not self.current_batch_acts:
                        continue  # hook failed?

                    acts = self.current_batch_acts[0]  # shape: [B, T, H]
                    B, T, H = acts.shape

                    # 4. flatten & store -------------------------------------------
                    for b in range(B):
                        for t in range(T):
                            act_vec = acts[b, t]
                            if act_vec is None:
                                continue
                            batch_data.append({
                                "sample_id": processed_samples,
                                "token_pos": int(t + start),
                                "token_id": int(sub_ids[b, t]),
                                "activation": act_vec.cpu().numpy().tolist(),
                                "layer_idx": int(self.layer_idx),
                                "text_snippet": raw_text[:120]
                            })

                    processed_samples += 1
                    if processed_samples >= self.max_samples:
                        break

                    # periodically flush ------------------------------------------
                    if len(batch_data) >= 1000:
                        self._write_batch_to_parquet(batch_data)
                        batch_data = []

                if processed_samples >= self.max_samples:
                    break

        # flush remainder
        if batch_data:
            self._write_batch_to_parquet(batch_data)

        # cleanup
        self.hook_handle.remove()
        size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(f"Recorded {processed_samples} samples → {self.output_path} ({size_mb:.1f} MB)")


class FeatureAnalyzer:
    """Analyzes which tokens activate SAE features most strongly"""
    
    def __init__(self, sae_model, tokenizer, activation_data_path: str, layer_idx: int):
        self.sae = sae_model
        self.tokenizer = tokenizer
        self.activation_data_path = Path(activation_data_path)
        self.layer_idx = layer_idx
        
    def analyze_feature_tokens(self, feature_idx: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find tokens that activate a specific feature most strongly"""
        # Load activation data
        table = pq.read_table(self.activation_data_path)
        df = table.to_pandas()
        df = df[df['layer_idx'] == self.layer_idx]
        
        if len(df) == 0:
            return []
        
        # Convert activations to tensor
        activations = []
        token_info = []
        
        for _, row in df.iterrows():
            activation = torch.tensor(row['activation'], dtype=torch.float32)
            activations.append(activation)
            token_info.append({
                'token_id': row['token_id'],
                'text_snippet': row['text_snippet'],
                'token_pos': row['token_pos']
            })
        
        activations = torch.stack(activations)
        
        # Get feature activations through SAE encoder
        with torch.no_grad():
            feature_activations = self.sae.encode(activations)  # [N, latent_dim]
            specific_feature_acts = feature_activations[:, feature_idx]  # [N]
        
        # Get top-k activating tokens
        top_values, top_indices = torch.topk(specific_feature_acts, k=min(top_k, len(specific_feature_acts)))
        
        # Prepare results
        top_tokens = []
        for i, (value, idx) in enumerate(zip(top_values, top_indices)):
            token_data = token_info[idx.item()]
            token_text = self.tokenizer.decode([token_data['token_id']])
            
            top_tokens.append({
                'rank': i + 1,
                'token': token_text,
                'token_id': token_data['token_id'],
                'activation_strength': value.item(),
                'context_snippet': token_data['text_snippet'],
                'position': token_data['token_pos']
            })
        
        return top_tokens
    
    def analyze_all_features(self, top_k_tokens: int = 5, max_features: int = 100) -> Dict[str, List[Dict]]:
        """Analyze top tokens for multiple features"""
        feature_token_map = {}
        
        max_features = min(max_features, self.sae.latent_dim)
        
        print(f"Analyzing top tokens for {max_features} features...")
        for feature_idx in tqdm(range(max_features)):
            try:
                top_tokens = self.analyze_feature_tokens(feature_idx, top_k_tokens)
                feature_token_map[f"feature_{feature_idx}"] = top_tokens
            except Exception as e:
                print(f"Error analyzing feature {feature_idx}: {e}")
                continue
        
        return feature_token_map
    
    def get_feature_summary(self, feature_idx: int) -> Dict[str, Any]:
        """Get comprehensive summary of a feature"""
        top_tokens = self.analyze_feature_tokens(feature_idx, top_k=10)
        
        # Compute feature statistics
        encoder_weight = self.sae.encoder.weight[feature_idx]  # Feature vector
        sparsity = (encoder_weight.abs() < 1e-6).float().mean().item()
        activation_strength = encoder_weight.norm().item()
        
        # Get activation statistics from top tokens
        if top_tokens:
            max_activation = max(token['activation_strength'] for token in top_tokens)
            avg_activation = np.mean([token['activation_strength'] for token in top_tokens])
        else:
            max_activation = 0.0
            avg_activation = 0.0
        
        return {
            'feature_idx': feature_idx,
            'layer_idx': self.layer_idx,
            'sparsity': sparsity,
            'weight_norm': activation_strength,
            'max_activation': max_activation,
            'avg_top_activation': avg_activation,
            'top_tokens': [token['token'] for token in top_tokens[:5]],
            'top_token_details': top_tokens
        }


if __name__ == "__main__":
    # Test with small model
    recorder = ActivationRecorder(
        model_name="Qwen/Qwen3-0.6B",
        layer_idx=10,
        output_path="test_activations.parquet",
        max_samples=100
    )
    recorder.record()