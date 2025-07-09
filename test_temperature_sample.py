import jax.numpy as jnp
import jax.random as random
import pytest
from temperature_sampler import temperature_sample


def mock_tokens_to_logits(tokens, cache):
    """Mock decoder function that returns predictable logits."""
    batch_size = tokens.shape[0]
    vocab_size = 4
    # Return logits favoring token 1, and unchanged cache
    logits = jnp.array([[0.0, 2.0, 1.0, 0.5]] * batch_size)
    return logits, cache


def test_temperature_sample_basic():
    """Test basic temperature sampling functionality."""
    key = random.PRNGKey(42)
    prompt_inputs = jnp.array([[0, 0, 0, 0]])  # batch_size=1, max_len=4
    init_cache = {}
    
    result = temperature_sample(
        prompt_inputs, init_cache, mock_tokens_to_logits, key, temperature=1.0, topk=2
    )
    
    assert result.shape == (1, 4)
    assert jnp.all(result >= 0)


def test_temperature_sample_eos_termination():
    """Test that sampling terminates at EOS token."""
    key = random.PRNGKey(123)
    prompt_inputs = jnp.array([[0, 0, 0, 0]])
    init_cache = {}
    
    def mock_eos_logits(tokens, cache):
        # Always return EOS token (2) with high probability
        logits = jnp.array([[0.0, 0.0, 10.0, 0.0]])
        return logits, cache
    
    result = temperature_sample(
        prompt_inputs, init_cache, mock_eos_logits, key, eos_token=2, topk=2
    )
    
    # Should contain EOS token
    assert jnp.any(result == 2)


def test_temperature_sample_greedy():
    """Test that temperature=0 produces greedy sampling."""
    key = random.PRNGKey(456)
    prompt_inputs = jnp.array([[0, 0, 0]])
    init_cache = {}
    
    def mock_greedy_logits(tokens, cache):
        # Token 1 has highest logit
        logits = jnp.array([[0.0, 10.0, 1.0, 0.0]])
        return logits, cache
    
    result = temperature_sample(
        prompt_inputs, init_cache, mock_greedy_logits, key, temperature=0.0, topk=2
    )
    
    # With temperature=0, should consistently pick token 1
    assert jnp.all((result == 1) | (result == 0))  # 0 for padding/prompt 


def test_temperature_sample_topp():
    """Test top-p (nucleus) sampling functionality."""
    key = random.PRNGKey(789)
    prompt_inputs = jnp.array([[0, 0, 0, 0]])  # batch_size=1, max_len=4
    init_cache = {}
    
    def mock_topp_logits(tokens, cache):
        # Create logits that will give specific probabilities after softmax
        # logits = [0.0, 3.0, 2.0, 1.0, 0.0] 
        # After softmax: ~[0.05, 0.67, 0.25, 0.09, 0.05]
        # Cumulative: [0.67, 0.92, 1.0, 1.0] (sorted descending)
        # With topp=0.9, should only consider tokens 1 and 2
        logits = jnp.array([[0.0, 3.0, 2.0, 1.0, 0.0]])
        return logits, cache
    
    # Test with topp=0.9 - should only sample from tokens 1 and 2
    result = temperature_sample(
        prompt_inputs, init_cache, mock_topp_logits, key, 
        temperature=1.0, topk=0, topp=0.9
    )
    
    assert result.shape == (1, 4)
    # All non-zero tokens should be either 1 or 2 (the top-p nucleus)
    non_zero_tokens = result[result != 0]
    assert jnp.all((non_zero_tokens == 1) | (non_zero_tokens == 2))


def test_temperature_sample_topp_always_includes_top():
    """Test that top-p always includes the highest probability token."""
    key = random.PRNGKey(101112)
    prompt_inputs = jnp.array([[0, 0, 0]])
    init_cache = {}
    
    def mock_peaked_logits(tokens, cache):
        # Very peaked distribution where top token has 0.9 probability
        # Even with very small topp, should still include top token
        logits = jnp.array([[0.0, 10.0, 1.0, 0.0]])
        return logits, cache
    
    # Test with very small topp - should still sample token 1
    result = temperature_sample(
        prompt_inputs, init_cache, mock_peaked_logits, key,
        temperature=1.0, topk=0, topp=0.1
    )
    
    # Should still be able to sample the top token (1)
    non_zero_tokens = result[result != 0]
    assert jnp.all(non_zero_tokens == 1)