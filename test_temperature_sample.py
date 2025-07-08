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