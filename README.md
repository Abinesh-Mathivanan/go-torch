# go-torch 

i built a simple pytorch implementation in go. till now, we support the basic linear layer support and you could perform a 'mnist character prediction' with the current setup. 

i aim to improve this to match torch's performance. 

blog - https://abinesh-mathivanan.vercel.app/en/posts/post-5/

<br>

## TODO
- [ ] add support for CNN, RNN
- [ ] optimize the Matmul with BLAS
- [ ] support building native opencl kernels for intel
- [ ] add gpu support 
- [ ] support building Transformers


some todo's are written inside the files. use 'better comments' extension for best experience. 

<br>

## Benchmark

| Operation           | 128×128       | 512×512       | 1024×1024     |
|---------------------|---------------|---------------|---------------|
| **Element-wise Add**    | 187.602 µs    | 2.326982 ms   | 9.558306 ms   |
| **Element-wise Mul**    | 126.740 µs    | 2.256796 ms   | 10.684073 ms  |
| **Matrix Multiply**     | 1.92448ms     | 87.649736 ms  | 780.434055 ms |
| **ReLU Activation**     | 226.385 µs    | 4.192483 ms   | 6.26745 ms    |




| Operation                      | Configuration                  | Avg Time per Iteration |
|-------------------------------|--------------------------------|-------------------------|
| **Linear Layer Forward**      | Batch: 32, In: 128, Out: 10     | 310.494 µs              |
| **CrossEntropyLoss**          | Batch: 32, Classes: 10          | 39.996 µs               |
| **Full Forward-Backward Pass**| Net: 128-256-10, Batch: 32      | 10.54919 ms           |


<br>
<br>
mail - abineshmathivanan31@gmail.com 
