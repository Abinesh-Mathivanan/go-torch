import torch
import torch.nn as nn
import time
import random
import numpy as np

# why cpu? 
# 1) i don't have a gpu 
# 2) go-torch only supports cpu 
DEVICE = torch.device("cpu")
print(f"PyTorch using device: {DEVICE}")




# bench params 
NUM_ITERATIONS = 100
SMALL_DIM = 128
MEDIUM_DIM = 512
LARGE_DIM = 1024
DEFAULT_BATCH_SIZE = 32




# generate a random pytorch tensor 
def generate_random_tensor_pytorch(shape, requires_grad=False):
    return torch.randn(shape, device=DEVICE, requires_grad=requires_grad)

def benchmark_tensor_creation_pytorch(dim, iterations):
    total_duration = 0
    shape = (dim, dim)
    for _ in range(iterations):
        start = time.perf_counter()
        _ = torch.randn(shape, device=DEVICE) 
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 





# we perform the following dimension-based tasks (for 100 iterations):
# 1) element wise addition        - 128, 512, 1024 
# 2) element wise multiplication  - 128, 512, 1024 
# 3) matrix multiplication        - 128, 512, 1024 
# 4) relu activation 

# all these metrics are calculated as ms (time * 1000). 

def benchmark_add_pytorch(dim, iterations):
    total_duration = 0
    tA = generate_random_tensor_pytorch((dim, dim))
    tB = generate_random_tensor_pytorch((dim, dim))
    for _ in range(iterations):
        start = time.perf_counter()
        _ = tA + tB
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 

def benchmark_mul_pytorch(dim, iterations):
    total_duration = 0
    tA = generate_random_tensor_pytorch((dim, dim))
    tB = generate_random_tensor_pytorch((dim, dim))
    for _ in range(iterations):
        start = time.perf_counter()
        _ = tA * tB
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 

def benchmark_matmul_pytorch(m, k, n, iterations):
    total_duration = 0
    tA = generate_random_tensor_pytorch((m, k))
    tB = generate_random_tensor_pytorch((k, n))
    for _ in range(iterations):
        start = time.perf_counter()
        _ = torch.matmul(tA, tB)
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 

def benchmark_relu_pytorch(dim, iterations):
    total_duration = 0
    t = generate_random_tensor_pytorch((dim, dim))
    for _ in range(iterations):
        start = time.perf_counter()
        _ = torch.relu(t)
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000





# we perform the following layer and loss benchmarks 
# 1) linear layer forward   
# 2) cross-Entropy Loss     
# 3) Full-forward Backward  

def benchmark_linear_forward_pytorch(batch_size, input_dim, output_dim, iterations):
    total_duration = 0
    linear_layer = nn.Linear(input_dim, output_dim).to(DEVICE)
    input_tensor = generate_random_tensor_pytorch((batch_size, input_dim))
    for _ in range(iterations):
        start = time.perf_counter()
        _ = linear_layer(input_tensor)
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 

def benchmark_cross_entropy_loss_pytorch(batch_size, num_classes, iterations):
    total_duration = 0
    loss_fn = nn.CrossEntropyLoss()
    logits = generate_random_tensor_pytorch((batch_size, num_classes))
    targets = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
    for _ in range(iterations):
        start = time.perf_counter()
        _ = loss_fn(logits, targets)
        total_duration += (time.perf_counter() - start)
    return (total_duration / iterations) * 1000 

def benchmark_forward_backward_pytorch(batch_size, input_dim, hidden_dim, output_dim, iterations):
    total_duration = 0
    
    for _ in range(iterations):
        x = generate_random_tensor_pytorch((batch_size, input_dim), requires_grad=True)
        layer1 = nn.Linear(input_dim, hidden_dim).to(DEVICE)
        layer2 = nn.Linear(hidden_dim, output_dim).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        targets = torch.randint(0, output_dim, (batch_size,), device=DEVICE)

        start = time.perf_counter()

        # forward pass
        h = layer1(x)
        h_relu = torch.relu(h)
        logits = layer2(h_relu)
        loss = loss_fn(logits, targets)

        # backward pass
        if x.grad is not None: x.grad.zero_()
        for param in layer1.parameters():
            if param.grad is not None: param.grad.zero_()
        for param in layer2.parameters():
            if param.grad is not None: param.grad.zero_()
        
        loss.backward()
        total_duration += (time.perf_counter() - start)
        # we end the timer here
        
    return (total_duration / iterations) * 1000 




# the print statements are self-explanatory. maybe try increasing the num of increasing, if you feel the test is unstable

if __name__ == "__main__":
    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    print("--- PyTorch Benchmarks (CPU) ---")
    print(f"Iterations per benchmark: {NUM_ITERATIONS}\n")

    dims_to_test = [SMALL_DIM, MEDIUM_DIM, LARGE_DIM]
    for d in dims_to_test:
        print(f"--- Dimension: {d}x{d} ---")
        print(f"Element-wise Add: {benchmark_add_pytorch(d, NUM_ITERATIONS):.3f} ms")
        print(f"Element-wise Mul: {benchmark_mul_pytorch(d, NUM_ITERATIONS):.3f} ms")
        print(f"Matrix Multiply ({d}x{d} @ {d}x{d}): {benchmark_matmul_pytorch(d, d, d, NUM_ITERATIONS):.3f} ms")
        print(f"ReLU Activation: {benchmark_relu_pytorch(d, NUM_ITERATIONS):.3f} ms")
        print()

    print("--- Layer and Loss Benchmarks ---")
    input_dim = 128
    hidden_dim = 256
    output_dim = 10  

    print(f"Linear Layer Forward (Batch: {DEFAULT_BATCH_SIZE}, In: {input_dim}, Out: {output_dim}): "
          f"{benchmark_linear_forward_pytorch(DEFAULT_BATCH_SIZE, input_dim, output_dim, NUM_ITERATIONS):.3f} ms")

    print(f"CrossEntropyLoss (Batch: {DEFAULT_BATCH_SIZE}, Classes: {output_dim}): "
          f"{benchmark_cross_entropy_loss_pytorch(DEFAULT_BATCH_SIZE, output_dim, NUM_ITERATIONS):.3f} ms")
    
    fwd_bwd_iterations = NUM_ITERATIONS // 10
    if fwd_bwd_iterations < 10: 
        fwd_bwd_iterations = 10 
        
    print(f"Full Forward-Backward (Net: {input_dim}-{hidden_dim}-{output_dim}, Batch: {DEFAULT_BATCH_SIZE}, Iter: {fwd_bwd_iterations}): "
          f"{benchmark_forward_backward_pytorch(DEFAULT_BATCH_SIZE, input_dim, hidden_dim, output_dim, fwd_bwd_iterations):.3f} ms")


    print("\n--- Benchmarks Complete ---")