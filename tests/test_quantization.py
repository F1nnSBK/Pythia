"""
Unit tests for Matryoshka-PolarQuant 1-bit binarization and error bounds.
"""

import numpy as np
import pytest
import scipy.linalg


def hadamard_matrix(d: int) -> np.ndarray:
    return scipy.linalg.hadamard(d) / np.sqrt(d)


def polarquant_binarize(z: np.ndarray, seed: int = 42) -> np.ndarray:
    """1-Bit Randomized Hadamard Sign Quantization."""
    d = z.shape[-1]
    np.random.seed(seed)
    h = hadamard_matrix(d)
    d_pre = np.random.choice([-1.0, 1.0], size=d)
    
    rotated = np.dot(z * d_pre, h.T)
    bits = (rotated > 0).astype(np.uint8)
    return bits


def test_hadamard_orthogonality():
    d = 64
    H = hadamard_matrix(d)
    # H * H.T should equal identity
    assert np.allclose(np.dot(H, H.T), np.eye(d), atol=1e-5)


def test_polarquant_angular_preservation():
    dim = 64
    n_pairs = 1000
    
    np.random.seed(42)
    z1 = np.random.randn(n_pairs, dim).astype(np.float32)
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True)
    
    # Create z2 with known angular distance
    z2 = z1 + np.random.randn(n_pairs, dim) * 0.3
    z2 /= np.linalg.norm(z2, axis=1, keepdims=True)
    
    cos_theta = np.sum(z1 * z2, axis=1)
    true_angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    expected_hamming = true_angles / np.pi
    
    b1 = polarquant_binarize(z1)
    b2 = polarquant_binarize(z2)
    
    empirical_hamming = np.mean(b1 != b2, axis=1)
    
    # Check that empirical Hamming distance closely tracks true angular distance (within concentration bounds)
    error = np.abs(empirical_hamming - expected_hamming)
    assert np.mean(error) < 0.08
