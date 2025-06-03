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

| Benchmark Detail                          | 128x128      | 512x512     | 1024x1024    |
|:------------------------------------------|:-------------|:------------|:-------------|
| **Matrix Multiply**                       | 510.33 µs    | 13.54 ms    | 130.50 ms    |
| Element-wise Add                          | 71.72 µs     | 1.29 ms     | 4.13 ms      |
| Element-wise Mul                          | 47.83 µs     | 1.63 ms     | 3.91 ms      |
| ReLU Activation                           | 121.18 µs    | 1.75 ms     | 6.45 ms      |
| **Linear Layer Forward (B32,I128,O10)**   | 71.93 µs     |             |              |
| **CrossEntropyLoss (B32,C10)**            | 11.16 µs     |             |              |
| **Full Fwd-Bwd (Net:128-256-10, B32)**    | 4.02 ms      |             |              |


<br>
<br>
mail - abineshmathivanan31@gmail.com 
