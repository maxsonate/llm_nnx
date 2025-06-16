"""Tests for input_pipeline.py module."""

import os
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import tensorflow as tf
import numpy as np

# Import the module under test
import input_pipeline
from configs import default


class TestNormalizeFeatureNamesOp(unittest.TestCase):
    """Test cases for NormalizeFeatureNamesOp class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ds_info = Mock()
        self.normalize_op = input_pipeline.NormalizeFeatureNamesOp(self.mock_ds_info)

    def test_normalize_feature_names(self):
        """Test that feature names are normalized correctly."""
        features = {'text': tf.constant(['Hello world', 'Test sentence'])}
        result = self.normalize_op(features)
        
        self.assertIn('inputs', result)
        self.assertIn('targets', result)
        self.assertNotIn('text', result)
        # Check that inputs and targets are the same (as per the implementation)
        tf.debugging.assert_equal(result['inputs'], result['targets'])


class TestPackDataset(unittest.TestCase):
    """Test cases for pack_dataset function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        self.test_data = [
            {'inputs': [1, 2, 3], 'targets': [4, 5]},
            {'inputs': [6, 7], 'targets': [8, 9, 10]},
            {'inputs': [11], 'targets': [12, 13, 14, 15]}
        ]
        
    def create_test_dataset(self):
        """Create a TensorFlow dataset from test data."""
        def gen():
            for item in self.test_data:
                yield {
                    'inputs': tf.constant(item['inputs'], dtype=tf.int32),
                    'targets': tf.constant(item['targets'], dtype=tf.int32)
                }
        
        return tf.data.Dataset.from_generator(
            gen,
            output_signature={
                'inputs': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )

    def test_pack_dataset_with_int_length(self):
        """Test pack_dataset with integer length parameter."""
        dataset = self.create_test_dataset()
        packed_ds = input_pipeline.pack_dataset(dataset, key2length=10)
        
        # Check that the dataset can be iterated
        for batch in packed_ds.take(1):
            self.assertIn('inputs', batch)
            self.assertIn('targets', batch)
            self.assertIn('inputs_segmentation', batch)
            self.assertIn('inputs_position', batch)
            self.assertIn('targets_segmentation', batch)
            self.assertIn('targets_position', batch)
            
            # Check shapes
            self.assertEqual(batch['inputs'].shape[0], 10)
            self.assertEqual(batch['targets'].shape[0], 10)

    def test_pack_dataset_with_dict_length(self):
        """Test pack_dataset with dictionary length parameter."""
        dataset = self.create_test_dataset()
        key2length = {'inputs': 8, 'targets': 6}
        packed_ds = input_pipeline.pack_dataset(dataset, key2length=key2length)
        
        for batch in packed_ds.take(1):
            self.assertEqual(batch['inputs'].shape[0], 8)
            self.assertEqual(batch['targets'].shape[0], 6)

    def test_pack_dataset_invalid_key(self):
        """Test pack_dataset raises error for invalid keys."""
        dataset = self.create_test_dataset()
        
        with self.assertRaises(ValueError):
            packed_ds = input_pipeline.pack_dataset(
                dataset, 
                key2length=10, 
                keys=['invalid_key']
            )
            # Force evaluation
            list(packed_ds.take(1))

    def test_pack_dataset_multidimensional_tensor_error(self):
        """Test pack_dataset raises error for multidimensional tensors."""
        def gen():
            yield {
                'inputs': tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
                'targets': tf.constant([[5, 6], [7, 8]], dtype=tf.int32)
            }
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                'inputs': tf.TensorSpec(shape=(2, 2), dtype=tf.int32),
                'targets': tf.TensorSpec(shape=(2, 2), dtype=tf.int32)
            }
        )
        
        with self.assertRaises(ValueError):
            input_pipeline.pack_dataset(dataset, key2length=10)


class TestPreprocessData(unittest.TestCase):
    """Test cases for preprocess_data function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        def gen():
            for i in range(10):
                yield {
                    'inputs': tf.constant([1, 2, 3, i], dtype=tf.int32),
                    'targets': tf.constant([4, 5, 6, i], dtype=tf.int32)
                }
        
        self.dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                'inputs': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )

    def test_preprocess_data_basic(self):
        """Test basic preprocessing functionality."""
        processed_ds = input_pipeline.preprocess_data(
            self.dataset,
            shuffle=False,
            num_epochs=1,
            pack_examples=False,
            max_length=10,
            batch_size=2
        )
        
        # Check that we can iterate through the dataset
        batches = list(processed_ds.take(2))
        self.assertEqual(len(batches), 2)
        
        for batch in batches:
            self.assertIn('inputs', batch)
            self.assertIn('targets', batch)
            self.assertEqual(batch['inputs'].shape[0], 2)  # batch_size
            self.assertEqual(batch['inputs'].shape[1], 10)  # max_length

    def test_preprocess_data_with_packing(self):
        """Test preprocessing with packing enabled."""
        processed_ds = input_pipeline.preprocess_data(
            self.dataset,
            shuffle=False,
            num_epochs=1,
            pack_examples=True,
            max_length=8,
            batch_size=2
        )
        
        # Check that we can iterate through the dataset
        batches = list(processed_ds.take(1))
        self.assertEqual(len(batches), 1)
        
        batch = batches[0]
        self.assertIn('inputs', batch)
        self.assertIn('targets', batch)
        # With packing, we should have segmentation and position keys
        self.assertIn('inputs_segmentation', batch)
        self.assertIn('inputs_position', batch)

    def test_preprocess_data_length_filtering(self):
        """Test that length filtering works correctly."""
        # Create dataset with varying lengths
        def gen():
            lengths = [2, 15, 3, 20, 4]  # Some exceed max_length=10
            for length in lengths:
                yield {
                    'inputs': tf.constant(list(range(length)), dtype=tf.int32),
                    'targets': tf.constant(list(range(length)), dtype=tf.int32)
                }
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                'inputs': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )
        
        processed_ds = input_pipeline.preprocess_data(
            dataset,
            shuffle=False,
            num_epochs=1,
            pack_examples=False,
            max_length=10,
            batch_size=5,
            drop_remainder=False
        )
        
        # Should filter out sequences longer than max_length
        batches = list(processed_ds)
        # We should have fewer examples due to filtering
        total_examples = sum(batch['inputs'].shape[0] for batch in batches)
        self.assertLess(total_examples, 5)  # Some should be filtered out


class TestGetRawDataset(unittest.TestCase):
    """Test cases for get_raw_dataset function."""

    @patch('input_pipeline.deterministic_data.get_read_instruction_for_host')
    def test_get_raw_dataset(self, mock_get_read_instruction):
        """Test get_raw_dataset function."""
        # Mock the dataset builder
        mock_builder = Mock()
        mock_builder.info.splits = {'train': Mock(num_examples=1000)}
        
        # Mock dataset
        mock_dataset = Mock()
        mock_builder.as_dataset.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        
        # Mock the read instruction
        mock_get_read_instruction.return_value = 'mock_instruction'
        
        result = input_pipeline.get_raw_dataset(mock_builder, 'train')
        
        # Verify calls
        mock_get_read_instruction.assert_called_once_with(
            'train', 1000, drop_remainder=False
        )
        mock_builder.as_dataset.assert_called_once_with(
            split='mock_instruction', shuffle_files=False
        )
        mock_dataset.map.assert_called_once()
        
        self.assertEqual(result, mock_dataset)


class TestGetDatasets(unittest.TestCase):
    """Test cases for get_datasets function."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = default.Config()
        self.config.dataset_name = 'test_dataset'
        self.config.eval_dataset_name = 'test_eval_dataset'
        self.config.eval_split = 'test'
        self.config.vocab_size = 1000
        self.config.max_corpus_chars = 10000
        self.config.per_device_batch_size = 4
        self.config.eval_per_device_batch_size = 2
        self.config.max_target_length = 64
        self.config.max_eval_target_length = 128
        self.config.max_predict_length = 32

    @patch('input_pipeline.tokenizer.load_or_train_tokenizer')
    @patch('input_pipeline.preprocess_data')
    @patch('input_pipeline.get_raw_dataset')
    @patch('input_pipeline.tfds.builder')
    def test_get_datasets(self, mock_tfds_builder, mock_get_raw_dataset, 
                         mock_preprocess_data, mock_load_tokenizer):
        """Test get_datasets function."""
        # Mock builders
        mock_train_builder = Mock()
        mock_eval_builder = Mock()
        mock_tfds_builder.side_effect = [mock_train_builder, mock_eval_builder]
        
        # Mock datasets
        mock_train_data = Mock()
        mock_eval_data = Mock()
        mock_get_raw_dataset.side_effect = [mock_train_data, mock_eval_data]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        # Mock tokenized datasets
        mock_train_data.map.return_value = mock_train_data
        mock_eval_data.map.return_value = mock_eval_data
        
        # Mock preprocessed datasets
        mock_train_ds = Mock()
        mock_eval_ds = Mock()
        mock_predict_ds = Mock()
        mock_preprocess_data.side_effect = [mock_train_ds, mock_eval_ds, mock_predict_ds]
        
        # Call the function
        n_devices = 8
        result = input_pipeline.get_datasets(self.config, n_devices=n_devices)
        
        # Verify the result
        train_ds, eval_ds, predict_ds, sp_tokenizer = result
        self.assertEqual(train_ds, mock_train_ds)
        self.assertEqual(eval_ds, mock_eval_ds)
        self.assertEqual(predict_ds, mock_predict_ds)
        self.assertEqual(sp_tokenizer, mock_tokenizer)
        
        # Verify calls
        self.assertEqual(mock_tfds_builder.call_count, 2)
        mock_tfds_builder.assert_any_call('test_dataset')
        mock_tfds_builder.assert_any_call('test_eval_dataset')
        
        self.assertEqual(mock_get_raw_dataset.call_count, 2)
        mock_get_raw_dataset.assert_any_call(mock_train_builder, 'train')
        mock_get_raw_dataset.assert_any_call(mock_eval_builder, 'test')
        
        # Verify tokenizer loading
        mock_load_tokenizer.assert_called_once_with(
            mock_train_data,
            vocab_path=os.path.expanduser('~/lm1b_sentencepiece_model'),
            vocab_size=1000,
            max_corpus_chars=10000
        )
        
        # Verify preprocessing calls
        self.assertEqual(mock_preprocess_data.call_count, 3)
        
        # Check train preprocessing
        mock_preprocess_data.assert_any_call(
            mock_train_data,
            shuffle=True,
            num_epochs=None,
            pack_examples=True,
            batch_size=32,  # 4 * 8 devices
            max_length=64
        )

    @patch('input_pipeline.tokenizer.load_or_train_tokenizer')
    @patch('input_pipeline.preprocess_data')
    @patch('input_pipeline.get_raw_dataset')
    @patch('input_pipeline.tfds.builder')
    def test_get_datasets_with_custom_vocab_path(self, mock_tfds_builder, 
                                                mock_get_raw_dataset, 
                                                mock_preprocess_data, 
                                                mock_load_tokenizer):
        """Test get_datasets with custom vocab path."""
        # Setup mocks
        mock_train_builder = Mock()
        mock_tfds_builder.return_value = mock_train_builder
        
        mock_train_data = Mock()
        mock_get_raw_dataset.return_value = mock_train_data
        
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_train_data.map.return_value = mock_train_data
        mock_preprocess_data.return_value = Mock()
        
        # Test with custom vocab path
        custom_vocab_path = '/custom/path/vocab.model'
        input_pipeline.get_datasets(
            self.config, 
            n_devices=1, 
            vocab_path=custom_vocab_path
        )
        
        # Verify tokenizer was called with custom path
        mock_load_tokenizer.assert_called_once()
        args, kwargs = mock_load_tokenizer.call_args
        self.assertEqual(kwargs['vocab_path'], custom_vocab_path)

    @patch('input_pipeline.tokenizer.load_or_train_tokenizer')
    @patch('input_pipeline.preprocess_data')
    @patch('input_pipeline.get_raw_dataset')
    @patch('input_pipeline.tfds.builder')
    def test_get_datasets_same_train_eval_dataset(self, mock_tfds_builder, 
                                                 mock_get_raw_dataset, 
                                                 mock_preprocess_data, 
                                                 mock_load_tokenizer):
        """Test get_datasets when eval_dataset_name is None."""
        # Modify config to not have separate eval dataset
        self.config.eval_dataset_name = None
        
        # Setup mocks
        mock_builder = Mock()
        mock_tfds_builder.return_value = mock_builder
        
        mock_train_data = Mock()
        mock_eval_data = Mock()
        mock_get_raw_dataset.side_effect = [mock_train_data, mock_eval_data]
        
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_train_data.map.return_value = mock_train_data
        mock_eval_data.map.return_value = mock_eval_data
        mock_preprocess_data.return_value = Mock()
        
        # Call function
        input_pipeline.get_datasets(self.config, n_devices=1)
        
        # Should only call tfds.builder once since eval uses same dataset
        mock_tfds_builder.assert_called_once_with('test_dataset')


class TestIntegration(unittest.TestCase):
    """Integration tests for the input pipeline."""

    def test_normalize_and_preprocess_integration(self):
        """Test integration between normalize and preprocess functions."""
        # Create a dataset that mimics TFDS structure
        def gen():
            for i in range(5):
                yield {'text': tf.constant(f'Sample text {i}', dtype=tf.string)}
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={'text': tf.TensorSpec(shape=(), dtype=tf.string)}
        )
        
        # Mock dataset info
        mock_ds_info = Mock()
        normalize_op = input_pipeline.NormalizeFeatureNamesOp(mock_ds_info)
        
        # Apply normalization
        normalized_ds = dataset.map(normalize_op)
        
        # Verify normalization worked
        for example in normalized_ds.take(1):
            self.assertIn('inputs', example)
            self.assertIn('targets', example)
            self.assertNotIn('text', example)

    def test_pack_dataset_realistic_example(self):
        """Test pack_dataset with more realistic data."""
        # Create dataset with token-like integer sequences
        def gen():
            sequences = [
                [1, 2, 3, 4, 0],  # 0 is padding/end token
                [5, 6, 0],
                [7, 8, 9, 10, 11, 0],
                [12, 0]
            ]
            for seq in sequences:
                yield {
                    'inputs': tf.constant(seq, dtype=tf.int32),
                    'targets': tf.constant(seq, dtype=tf.int32)
                }
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                'inputs': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )
        
        # Pack with reasonable length
        packed_ds = input_pipeline.pack_dataset(dataset, key2length=12)
        
        # Verify packed dataset structure
        for batch in packed_ds.take(1):
            # Check all required keys exist
            required_keys = [
                'inputs', 'targets', 
                'inputs_segmentation', 'inputs_position',
                'targets_segmentation', 'targets_position'
            ]
            for key in required_keys:
                self.assertIn(key, batch)
                self.assertEqual(batch[key].shape[0], 12)
            
            # Verify segmentation makes sense (should have values 1, 2, etc.)
            segmentation = batch['inputs_segmentation'].numpy()
            unique_segments = np.unique(segmentation[segmentation > 0])
            self.assertGreater(len(unique_segments), 0)


if __name__ == '__main__':
    # Configure TensorFlow for testing
    tf.config.experimental.enable_op_determinism()
    
    unittest.main() 