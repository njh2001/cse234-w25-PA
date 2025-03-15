from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        a, b = input_values
        mul = torch.matmul(a, b)
        m = torch.mean(mul, dim=-1, keepdim=True)
        v = torch.var(mul, dim=-1, keepdim=True, unbiased=False)

        eps = node.attrs["eps"]
        x = (mul - m) / torch.sqrt(v + eps)
        return x

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""

        a, b = node.inputs
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs['eps']
        dims = tuple(range(-len(normalized_shape), 0))

        mul = matmul(a, b)
        m = mean(mul, dim=dims, keepdim=True)
        m2 = mean(power(mul, 2), dim=dims, keepdim=True)
        var = m2 - power(m, 2)

        normalized = (mul - m) / sqrt(var + eps)
        g = output_grad
        g_mean = mean(g, dim=dims, keepdim=True)
        g_normalized_mean = mean(g * normalized, dim=dims, keepdim=True)
        grad_x = (g - g_mean - normalized * g_normalized_mean) / sqrt(var + eps)

        grad_A = matmul(grad_x, transpose(b, -1, -2))
        grad_B = matmul(transpose(a, -1, -2), grad_x)
        return [grad_A, grad_B]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        a, b = input_values
        mul = torch.matmul(a, b)
        return torch.exp(mul - mul.max(dim=node.dim, keepdim=True)[0]) / torch.exp(mul - mul.max(dim=node.dim, keepdim=True)[0]).sum(dim=node.dim, keepdim=True)
        

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        a, b = node.inputs
        mul = matmul(a, b)
        
        softmax_val = softmax(mul, node.dim)
        grad_x = softmax_val * (output_grad - sum_op(softmax_val * output_grad, dim=node.dim, keepdim=True))
        
        grad_A = matmul(grad_x, transpose(b, -1, -2))
        grad_B = matmul(transpose(a, -1, -2), grad_x)
        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()