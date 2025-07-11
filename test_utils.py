import pytest
import jax
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from utils import create_device_mesh
from configs import default


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