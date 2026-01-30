import mlx.core as mx
import pytest
import numpy as np
from metal_lora.ops import lora_forward, lora_backward
from metal_lora.kernels import is_metal_available, lora_adamw_step_metal

@pytest.mark.skipif(not is_metal_available(), reason="Metal not available")
class TestMetalParity:
    def test_forward_parity(self):
        batch, seq, in_f, out_f, rank = 2, 64, 512, 1024, 16
        x = mx.random.normal((batch, seq, in_f))
        w0 = mx.random.normal((out_f, in_f))
        a = mx.random.normal((rank, in_f))
        b = mx.random.normal((out_f, rank))
        
        # Pure MLX
        out_mlx = lora_forward(x, w0, a, b, use_metal=False)
        # Metal (SIMD with simdgroup_matrix)
        out_metal = lora_forward(x, w0, a, b, use_metal=True, use_simd=True)
        
        mx.eval(out_mlx, out_metal)
        np.testing.assert_allclose(np.array(out_mlx), np.array(out_metal), rtol=1e-3, atol=1e-3)

    def test_backward_parity(self):
        batch, seq, in_f, out_f, rank = 1, 32, 256, 512, 8
        x = mx.random.normal((batch, seq, in_f))
        a = mx.random.normal((rank, in_f))
        b = mx.random.normal((out_f, rank))
        grad_output = mx.random.normal((batch, seq, out_f))
        
        # Pure MLX
        ga_mlx, gb_mlx = lora_backward(grad_output, x, a, b, use_metal=False)
        # Metal Fused
        ga_metal, gb_metal = lora_backward(grad_output, x, a, b, use_metal=True)
        
        mx.eval(ga_mlx, gb_mlx, ga_metal, gb_metal)
        np.testing.assert_allclose(np.array(ga_mlx), np.array(ga_metal), rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(np.array(gb_mlx), np.array(gb_metal), rtol=1e-2, atol=1e-2)

    def test_adamw_parity(self):
        size = 1000
        param = mx.random.normal((size,))
        grad = mx.random.normal((size,))
        m = mx.zeros((size,))
        v = mx.zeros((size,))
        
        param_metal = mx.array(param)
        m_metal = mx.array(m)
        v_metal = mx.array(v)
        grad_metal = mx.array(grad)
        
        lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1
        
        # Metal Step
        lora_adamw_step_metal(param_metal, grad_metal, m_metal, v_metal, lr, b1, b2, eps, wd, step)
        
        # Manual MLX Step for verification
        b1t = b1 ** step
        b2t = b2 ** step
        m_mlx = b1 * m + (1 - b1) * grad
        v_mlx = b2 * v + (1 - b2) * (grad * grad)
        m_hat = m_mlx / (1 - b1t)
        v_hat = v_mlx / (1 - b2t)
        param_mlx = param - lr * (m_hat / (mx.sqrt(v_hat) + eps) + wd * param)
        
        mx.eval(param_metal, param_mlx)
        np.testing.assert_allclose(np.array(param_mlx), np.array(param_metal), rtol=1e-5, atol=1e-5)
