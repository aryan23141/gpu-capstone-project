# gpu-capstone-project
GPU Accelerated Matrix Multiplication using CUDA (CuPy)
# GPU Accelerated Matrix Multiplication

## Overview
This project demonstrates GPU acceleration using CUDA by performing large-scale
matrix multiplication on the GPU. The implementation uses **CuPy**, a NumPy-like
library that runs on NVIDIA GPUs using CUDA.

This project was developed as part of the **GPU Specialization Capstone Project**.

---

## Motivation
Matrix multiplication is a core operation in scientific computing, machine learning,
and image processing. Executing this operation on the GPU provides massive
performance improvements due to parallel execution.

---

## Technologies Used
- Python 3
- CUDA-enabled NVIDIA GPU
- CuPy (GPU accelerated NumPy)
- NumPy (for comparison)

---

## How It Works
1. Two large matrices are generated on the GPU.
2. Matrix multiplication is performed using CUDA cores.
3. Execution time is measured to demonstrate GPU acceleration.

---

## How to Run
```bash
pip install cupy-cuda11x
python gpu_matmul.py
