"""Narrow BF16-input / FP32-output router GEMM for the pinned SGLang build.

Prime-RL's inference flag selects FP32 router output without rounding the
GEMM output to BF16. Keep SGLang's parameter names, storage and loading intact.
Only the unquantized Qwen MoE gate uses this subclass, not shared-expert gates.
"""
import torch
from sglang.srt.layers.linear import ReplicatedLinear


class FP32RouterLinear(ReplicatedLinear):
    def forward(self, x):
        assert self.bias is None
        assert self.weight.dtype == x.dtype == torch.bfloat16
        return torch.mm(x, self.weight.T, out_dtype=torch.float32), None


def test_gpu():
    layer = FP32RouterLinear.__new__(FP32RouterLinear)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16))
    layer.bias = None
    for count in [0, 1, 8, 16, 128]:
        x = torch.randn(count, 2048, device="cuda", dtype=torch.bfloat16)
        actual, bias = layer(x)
        expected = torch.mm(x, layer.weight.T, out_dtype=torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.dtype == torch.float32 and bias is None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = layer(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    print("FP32 router: five GPU shapes and CUDA graph replay passed", flush=True)


if __name__ == "__main__":
    test_gpu()
