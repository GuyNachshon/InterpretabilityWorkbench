"""
Tests for activation recording functionality
"""
import pytest
import torch
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

from interpretability_workbench.trace import ActivationRecorder


class TestActivationRecorder:
    """Test cases for ActivationRecorder"""
    
    def test_init_with_small_model(self):
        """Test initialization with a small model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_activations.parquet"
            
            recorder = ActivationRecorder(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=0,
                output_path=str(output_path),
                max_samples=10
            )
            
            assert recorder.model is not None
            assert recorder.tokenizer is not None
            assert recorder.layer_idx == 0
    
    def test_hook_registration(self):
        """Test that forward hooks are properly registered"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_activations.parquet"
            
            recorder = ActivationRecorder(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=0,
                output_path=str(output_path),
                max_samples=10
            )
            
            # Check that hook handle exists
            assert hasattr(recorder, 'hook_handle')
            assert recorder.hook_handle is not None
            
            # Clean up
            recorder.hook_handle.remove()
    
    @pytest.mark.slow
    def test_record_small_dataset(self):
        """Test recording from a small dataset"""
        import unittest.mock as mock
        from datasets import Dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_activations.parquet"
            
            # Create a mock dataset with our test texts
            mock_texts = [
                "Hello world",
                "This is a test", 
                "Another example text",
                "Short text",
                "Slightly longer example text for testing"
            ]
            
            # Create a HuggingFace Dataset object
            mock_dataset = Dataset.from_dict({"text": mock_texts})
            
            # Create recorder (force CPU for test compatibility)
            recorder = ActivationRecorder(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=0,
                output_path=str(output_path),
                max_samples=5,
                max_length=64,  # Shorter for test
                device="cpu"  # Force CPU to avoid MPS issues in tests
            )
            
            # Mock the dataset loading to return our test dataset
            with mock.patch('interpretability_workbench.trace.load_dataset') as mock_load_dataset:
                mock_load_dataset.return_value = mock_dataset
                
                # Run the recording
                recorder.record(dataset_name="mock_dataset")
            
            # Verify the output file was created
            assert output_path.exists(), "Output parquet file should be created"
            
            # Load and verify the recorded data
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            # Verify we have data
            assert len(df) > 0, "Should have recorded some activations"
            
            # Verify expected columns exist
            expected_columns = ["sample_id", "token_pos", "token_id", "activation", "layer_idx", "text_snippet"]
            for col in expected_columns:
                assert col in df.columns, f"Column '{col}' should exist in recorded data"
            
            # Verify layer_idx is correct
            assert (df["layer_idx"] == 0).all(), "All activations should be from layer 0"
            
            # Verify we have at most max_samples worth of data
            unique_samples = df["sample_id"].nunique()
            assert unique_samples <= 5, f"Should have max 5 samples, got {unique_samples}"
            
            # Verify activation vectors have reasonable shape
            first_activation = df.iloc[0]["activation"]
            assert isinstance(first_activation, (list, np.ndarray)), "Activation should be stored as list or array"
            assert len(first_activation) > 0, "Activation vector should not be empty"
            
            # Verify text snippets are from our mock data
            text_snippets = set(df["text_snippet"].str[:20])  # First 20 chars
            expected_snippets = {text[:20] for text in mock_texts}
            assert text_snippets.issubset(expected_snippets), "Text snippets should come from our mock data"
            
            # Verify token IDs are valid (non-negative integers)
            assert (df["token_id"] >= 0).all(), "All token IDs should be non-negative"
            assert (df["token_id"] < recorder.tokenizer.vocab_size).all(), "Token IDs should be within vocab size"
            
            # Clean up hook
            recorder.hook_handle.remove()
            
            print(f"✅ Successfully recorded {len(df)} activations from {unique_samples} samples")
            print(f"   Activation vector size: {len(first_activation)}")
            print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    def test_parquet_write_batch(self):
        """Test writing batch data to parquet"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_activations.parquet"
            
            recorder = ActivationRecorder(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=0,
                output_path=str(output_path),
                max_samples=10
            )
            
            # Create sample data
            batch_data = [
                {
                    "sample_id": 0,
                    "token_pos": 0,
                    "token_id": 123,
                    "activation": [0.1, 0.2, 0.3],
                    "layer_idx": 0,
                    "text_snippet": "test"
                },
                {
                    "sample_id": 0,
                    "token_pos": 1,
                    "token_id": 456,
                    "activation": [0.4, 0.5, 0.6],
                    "layer_idx": 0,
                    "text_snippet": "test"
                }
            ]
            
            # Write batch
            recorder._write_batch_to_parquet(batch_data)
            
            # Verify file exists and has correct data
            assert output_path.exists()
            
            # Read back and verify
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            assert len(df) == 2
            assert df.iloc[0]["token_id"] == 123
            assert df.iloc[1]["token_id"] == 456
            assert list(df.iloc[0]["activation"]) == [0.1, 0.2, 0.3]


if __name__ == "__main__":
    pytest.main([__file__])