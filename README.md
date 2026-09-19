# go-torch 

go-torch is an open-source deep learning framework built from the ground up in pure Go. It provides a modular, PyTorch-like API for building and training neural networks with a stable auto-differentiation engine. Purely a hobby project. 

mail - abineshmathivanan31@gmail.com 

blog - https://abinesh-mathivanan.vercel.app/en/posts/post-5/


## features 
- **dynamic computation graph**: tensors track their history, allowing for automatic gradient calculation during the backward pass. every custom backward pass is verified against numerical (finite-difference) gradients 
- **layers**: Conv2D, Linear, MaxPooling2D, Flatten, BatchNorm1d, BatchNorm2d, LayerNorm, Embedding, RNN, LSTM, Dropout
- **activations**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, SiLU/Swish
- **losses**: CrossEntropyLoss
- **optimizers**: SGD (with momentum), Adam, AdamW (decoupled weight decay), RMSProp
- **training utilities**: gradient clipping (by norm or by value), LR schedulers (StepLR, CosineAnnealingLR, WarmupScheduler)
- **model checkpointing**: save/load - named and shape-checked, so a renamed/reordered/added layer fails loudly instead of silently loading into the wrong slot. A legacy positional `gob` format is also still available.
- **optimized performance**: BLAS-backed matrix multiply (falls back to a parallel pure-Go path below a tuned size threshold), and a size-aware parallelization policy applied across the board so small tensors (e.g. RNN/LSTM per-timestep ops) don't pay goroutine overhead for no benefit. Tunable via the `GOTORCH_PARALLEL_THRESHOLD` environment variable

<br/>

## dependencies

Just one: [`gonum.org/v1/gonum`](https://pkg.go.dev/gonum.org/v1/gonum), used for BLAS-backed matrix multiplication. Rest are stdlib.

<br/>

## TODO
- [ ] LoRA / GaLore
- [x] model.save()/load() without gob → safetensors (gob kept as a legacy option)
- [ ] Transformers: LayerNorm, GELU/SiLU, and Embedding are in place; multi-head attention is the remaining piece
- [ ] multi-layer RNN/LSTM (currently single-layer only)
- [ ] GRU
- [ ] text-model training scope (tokenizer + causal LM training loop)
- [ ] CUDA support
- [ ] ONNX import/export
- [ ] plotting (likely `gonum.org/v1/plot`, to keep the dependency footprint minimal)
- [ ] replace Im2Col/Col2Im with an implicit-GEMM convolution (avoids the memory duplication inherent to explicit im2col) - not yet profiled to confirm it's actually a bottleneck

<br/>

## pre-requisites 
- Go 1.22 or later.
- system-installed BLAS library is recommended for maximum performance but not required.
- some todo's are written inside the files. use 'better comments' extension for best experience. 

<br/>

## usage 

### clone the repository
```bash
git clone https://github.com/abinesh-mathivanan/go-torch.git
cd go-torch
``` 
### install dependencies 
``` bash
go mod tidy
```

### run the demos
```bash
# MNIST training with safetensors checkpointing
go run ./mnist_trainer

# CNN with BatchNorm + Dropout, plain-text training logger
go run ./benchmark

# RNN/LSTM forward and forward-backward benchmarks
go run ./benchmark/rnn_lstm

# op-level microbenchmarks (matmul, elementwise ops, layers, loss)
go run ./utils

# RNN/LSTM smoke tests
go run ./test
```

<br/>

## Verification

Custom backward-pass math (BatchNorm, LayerNorm, the newer activations, Embedding's scatter-add) is checked against numerical (finite-difference) gradients rather than trusted on inspection alone.

```bash
go run ./verify                    # numerical gradient checks for every custom backward pass
go run ./verify/safetensors_check  # safetensors round-trip (bit-exact) + error-path checks
```

Both should print `PASS` for every case. If you add a new layer with a hand-derived backward pass, add a check for it here before trusting it in training.

<br/>


## Known limitations

- RNN/LSTM: single layer only (`numLayers > 1` returns an error).
- `Embedding` and `CrossEntropyLoss` take `[]int` rather than a `*tensor.Tensor` for indices/targets, so they don't implement the `nn.Layer` interface used by `Sequential` - call them directly.
- Convolution uses explicit im2col + GEMM, which trades memory (the column matrix duplicates input data once per kernel position) for GEMM speed. Not yet replaced with an implicit-GEMM approach, and not yet profiled to confirm it's worth doing.
- No GPU/CUDA support yet.

<br/>
<br/>

[![Star History Chart](https://api.star-history.com/svg?repos=Abinesh-Mathivanan/go-torch&type=Date)](https://www.star-history.com/#Abinesh-Mathivanan/go-torch&Date)

