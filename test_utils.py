import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from flax import nnx
from flax.core import FrozenDict
from utils import create_device_mesh, count_parameters
from configs import default
from modules import TransformerConfig
from model import TransformerLM


class TestCreateDeviceMesh:
    
    def test_create_device_mesh_single_slice(self):
        """Test device mesh creation in single slice environment"""
        # Mock config
        config = Mock(spec=default.Config)
        config.dcn_data_parallelism = 1
        config.dcn_fsdp_parallelism = 1
        config.dcn_tensor_parallelism = 1
        config.ici_data_parallelism = 2
        config.ici_fsdp_parallelism = -1  # Auto-determine
        config.ici_tensor_parallelism = 2
        
        # Mock devices without slice_index (single slice)
        mock_devices = [Mock() for _ in range(8)]
        for device in mock_devices:
            # Devices don't have slice_index attribute in single slice
            if hasattr(device, 'slice_index'):
                delattr(device, 'slice_index')
        
        # Mock the mesh creation
        mock_mesh = Mock()
        mock_mesh.shape = (2, 2, 2)
        
        with patch('jax.devices', return_value=mock_devices), \
             patch('utils.mesh_utils.create_device_mesh', return_value=mock_mesh) as mock_create_mesh, \
             patch('utils.logging'):
            
            result = create_device_mesh(config)
            
            # Verify mesh creation was called with correct ICI parallelism
            # ici_fsdp_parallelism should be auto-determined as 2 (8 devices / 2 / 2 = 2)
            expected_ici_parallelism = [2, 2, 2]
            mock_create_mesh.assert_called_once_with(expected_ici_parallelism)
            
            # Verify result
            assert result == mock_mesh
            assert result.shape == (2, 2, 2)


class TestCountParameters:
    
    def test_count_parameters_with_simple_pytree(self):
        """Test parameter counting with a simple parameter pytree"""
        # Create a simple parameter structure
        params = {
            'embeddings': {
                'weight': jnp.ones((1000, 128))  # 128,000 parameters
            },
            'dense': {
                'kernel': jnp.ones((128, 64)),   # 8,192 parameters
                'bias': jnp.ones((64,))          # 64 parameters
            }
        }
        
        total_params, _ = count_parameters(params)
        
        # Expected: 128,000 + 8,192 + 64 = 136,256
        expected_total = 1000 * 128 + 128 * 64 + 64
        assert total_params == expected_total
        assert total_params == 136256
    
    def test_count_parameters_with_frozen_dict(self):
        """Test parameter counting with FrozenDict structure"""
        # Create a FrozenDict parameter structure
        params = FrozenDict({
            'layer1': {
                'weight': jnp.ones((10, 20)),    # 200 parameters
                'bias': jnp.ones((20,))          # 20 parameters
            },
            'layer2': {
                'weight': jnp.ones((20, 5)),     # 100 parameters
                'bias': jnp.ones((5,))           # 5 parameters
            }
        })
        
        total_params, _ = count_parameters(params)
        
        # Expected: 200 + 20 + 100 + 5 = 325
        expected_total = 10 * 20 + 20 + 20 * 5 + 5
        assert total_params == expected_total
        assert total_params == 325
    
    def test_count_parameters_with_real_model(self):
        """Test parameter counting with a real TransformerLM model"""
        # Create a small transformer config for testing
        config = TransformerConfig(
            vocab_size=100,
            output_vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            qkv_dim=32,
            mlp_dim=64,
            max_len=10,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        
        # Initialize model
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerLM(config, decode=False, rngs=rngs)
        
        # Get parameters
        params = nnx.state(model, nnx.Param)
        
        # Count parameters
        total_params, _ = count_parameters(params)
        
        # Verify we get a reasonable number (should be > 0)
        assert total_params > 0
        
        # For this small model, let's verify it's in a reasonable range
        # Rough calculation: embeddings (100*32=3200) + layers + output head
        assert total_params > 3000  # Should be at least embedding size
        assert total_params < 50000  # But not too large for this tiny model
    
    def test_count_parameters_empty_dict(self):
        """Test parameter counting with empty parameter dict"""
        params = {}
        total_params, _ = count_parameters(params)
        assert total_params == 0
    
    def test_create_device_mesh_multi_slice(self):
        """Test device mesh creation in multi-slice environment"""
        # Mock config
        config = Mock(spec=default.Config)
        config.dcn_data_parallelism = -1  # Auto-determine
        config.dcn_fsdp_parallelism = 1
        config.dcn_tensor_parallelism = 2
        config.ici_data_parallelism = 2
        config.ici_fsdp_parallelism = 2
        config.ici_tensor_parallelism = 2
        
        # Mock devices with slice_index (multi-slice)
        mock_devices = []
        for slice_idx in range(2):  # 2 slices
            for device_idx in range(8):  # 8 devices per slice
                device = Mock()
                device.slice_index = slice_idx
                mock_devices.append(device)
        
        # Mock the hybrid mesh creation
        mock_mesh = Mock()
        mock_mesh.shape = (2, 2, 2, 2)  # Hybrid mesh has both DCN and ICI dimensions
        
        with patch('jax.devices', return_value=mock_devices), \
             patch('utils.mesh_utils.create_hybrid_device_mesh', return_value=mock_mesh) as mock_create_hybrid, \
             patch('utils.logging'):
            
            result = create_device_mesh(config)
            
            # Verify hybrid mesh creation was called with correct parallelism
            # dcn_data_parallelism should be auto-determined as 1 (2 slices / 1 / 2 = 1)
            expected_ici_parallelism = [2, 2, 2]
            expected_dcn_parallelism = [1, 1, 2]  # Auto-determined: 2 slices / 1 / 2 = 1
            mock_create_hybrid.assert_called_once_with(expected_ici_parallelism, expected_dcn_parallelism)
            
            # Verify result
            assert result == mock_mesh 