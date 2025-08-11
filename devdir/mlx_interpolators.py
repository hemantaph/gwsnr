import mlx.core as mx
import time

# 4096 x 4096 matrix multiplication
def test():
    size = 4096
    a = mx.random.uniform(size=(size, size)).to("gpu")
    b = mx.random.uniform(size=(size, size)).to("gpu")

    start = time.time()
    c = mx.matmul(a, b)  # On GPU
    mx.eval(c)           # Forces computation
    print("Elapsed (GPU):", time.time() - start)