
Paste this code:

```python
import cupy as cp
import time

N = 1024

print("Allocating matrices on GPU...")
A = cp.random.rand(N, N)
B = cp.random.rand(N, N)

cp.cuda.Stream.null.synchronize()

print("Performing GPU matrix multiplication...")
start = time.time()
C = cp.dot(A, B)
cp.cuda.Stream.null.synchronize()
end = time.time()

print("GPU Matrix Multiplication Time:", end - start)
print("Result shape:", C.shape)
