package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"go-torch/nn"
	"go-torch/tensor"
)



// bench params 
const (
	numIterations    = 100 
	smallDim         = 128
	mediumDim        = 512
	largeDim         = 1024
	defaultBatchSize = 32
)


// utility functions for benchmark creation 

func generateRandomData(size int) []float64 {
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 
	}
	return data
}


// we perform the following dimension-based tasks (for 100 iterations):
// 1) element wise addition        - 128, 512, 1024 
// 2) element wise multiplication  - 128, 512, 1024 
// 3) matrix multiplication        - 128, 512, 1024 
// 4) relu activation 

// all these metrics are calculated as ms (time * 1000). 

func benchmarkAdd_GoTorch(dim int, iterations int) time.Duration {
	var totalDuration time.Duration
	shape := []int{dim, dim}
	numElements := dim * dim
	dataA := generateRandomData(numElements)
	dataB := generateRandomData(numElements)
	tA, _ := tensor.NewTensor(shape, dataA)
	tB, _ := tensor.NewTensor(shape, dataB)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := tensor.AddTensor(tA, tB)
		if err != nil {
			log.Fatalf("GoTorch: Error in AddTensor: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}

func benchmarkMul_GoTorch(dim int, iterations int) time.Duration {
	var totalDuration time.Duration
	shape := []int{dim, dim}
	numElements := dim * dim
	dataA := generateRandomData(numElements)
	dataB := generateRandomData(numElements)
	tA, _ := tensor.NewTensor(shape, dataA)
	tB, _ := tensor.NewTensor(shape, dataB)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := tensor.MulTensor(tA, tB)
		if err != nil {
			log.Fatalf("GoTorch: Error in MulTensor: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}

func benchmarkMatMul_GoTorch(m, k, n int, iterations int) time.Duration {
	var totalDuration time.Duration
	shapeA := []int{m, k}
	shapeB := []int{k, n}
	dataA := generateRandomData(m * k)
	dataB := generateRandomData(k * n)
	tA, _ := tensor.NewTensor(shapeA, dataA)
	tB, _ := tensor.NewTensor(shapeB, dataB)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := tensor.MatMulTensor(tA, tB)
		if err != nil {
			log.Fatalf("GoTorch: Error in MatMulTensor: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}

func benchmarkReLU_GoTorch(dim int, iterations int) time.Duration {
	var totalDuration time.Duration
	shape := []int{dim, dim}
	numElements := dim * dim
	data := generateRandomData(numElements)
	t, _ := tensor.NewTensor(shape, data)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := nn.RELU(t)
		if err != nil {
			log.Fatalf("GoTorch: Error in RELU: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}



// we perform the following layer and loss benchmarks 
// 1) linear layer forward   
// 2) cross-Entropy Loss     
// 3) Full-forward Backward  

func benchmarkLinearForward_GoTorch(batchSize, inputDim, outputDim int, iterations int) time.Duration {
	var totalDuration time.Duration
	linearLayer, err := nn.NewLinear(inputDim, outputDim)
	if err != nil {
		log.Fatalf("GoTorch: Error creating linear layer: %v", err)
	}
	inputData := generateRandomData(batchSize * inputDim)
	inputTensor, _ := tensor.NewTensor([]int{batchSize, inputDim}, inputData)

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := linearLayer.Forward(inputTensor)
		if err != nil {
			log.Fatalf("GoTorch: Error in Linear Forward: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}

func benchmarkCrossEntropyLoss_GoTorch(batchSize, numClasses int, iterations int) time.Duration {
	var totalDuration time.Duration
	logitsData := generateRandomData(batchSize * numClasses)
	logits, _ := tensor.NewTensor([]int{batchSize, numClasses}, logitsData)
	targets := make([]int, batchSize)
	for i := range targets {
		targets[i] = rand.Intn(numClasses)
	}

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := nn.CrossEntropyLoss(logits, targets)
		if err != nil {
			log.Fatalf("GoTorch: Error in CrossEntropyLoss: %v", err)
		}
		totalDuration += time.Since(start)
	}
	return totalDuration / time.Duration(iterations)
}

func benchmarkForwardBackward_GoTorch(batchSize, inputDim, hiddenDim, outputDim int, iterations int) time.Duration {
	var totalDuration time.Duration

	for i := 0; i < iterations; i++ {
		inputData := generateRandomData(batchSize * inputDim)
		x, _ := tensor.NewTensor([]int{batchSize, inputDim}, inputData)
		x.RequiresGrad = true

		layer1, _ := nn.NewLinear(inputDim, hiddenDim)
		layer2, _ := nn.NewLinear(hiddenDim, outputDim)

		targets := make([]int, batchSize)
		for i := range targets {
			targets[i] = rand.Intn(outputDim)
		}

		// begin the timer here
		start := time.Now()

		// forward pass
		h, _ := layer1.Forward(x)
		hRelu, _ := nn.RELU(h)
		logits, _ := layer2.Forward(hRelu)
		loss, err := nn.CrossEntropyLoss(logits, targets)
		if err != nil {
			log.Fatalf("GoTorch: Error in loss during forward/backward: %v", err)
		}

		// backward pass
		if loss.RequiresGrad {
			loss.Backward(nil)
		}
		totalDuration += time.Since(start)
		// end there timer here

		// zero grads are optional 
		x.ZeroGrad()
		layer1.ZeroGrad()
		layer2.ZeroGrad()
	}
	return totalDuration / time.Duration(iterations)
}

func main() {
	rand.Seed(time.Now().UnixNano())  // deprecated but works, ig. you could change it. :)
	fmt.Println("--- Go-Torch Benchmarks ---")
	fmt.Printf("Iterations per benchmark: %d\n\n", numIterations)

	dims := []int{smallDim, mediumDim, largeDim}
	for _, d := range dims {
		fmt.Printf("--- Dimension: %dx%d ---\n", d, d)
		fmt.Printf("Element-wise Add: %v\n", benchmarkAdd_GoTorch(d, numIterations))
		fmt.Printf("Element-wise Mul: %v\n", benchmarkMul_GoTorch(d, numIterations))
		fmt.Printf("Matrix Multiply (%dx%d @ %dx%d): %v\n", d, d, d, d, benchmarkMatMul_GoTorch(d, d, d, numIterations))
		fmt.Printf("ReLU Activation: %v\n", benchmarkReLU_GoTorch(d, numIterations))
		fmt.Println()
	}

	fmt.Println("--- Layer and Loss Benchmarks ---")
	inputDim := 128
	hiddenDim := 256
	outputDim := 10 

	fmt.Printf("Linear Layer Forward (Batch: %d, In: %d, Out: %d): %v\n",
		defaultBatchSize, inputDim, outputDim,
		benchmarkLinearForward_GoTorch(defaultBatchSize, inputDim, outputDim, numIterations))

	fmt.Printf("CrossEntropyLoss (Batch: %d, Classes: %d): %v\n",
		defaultBatchSize, outputDim,
		benchmarkCrossEntropyLoss_GoTorch(defaultBatchSize, outputDim, numIterations))

	fmt.Printf("Full Forward-Backward (Net: %d-%d-%d, Batch: %d): %v\n",
		inputDim, hiddenDim, outputDim, defaultBatchSize,
		benchmarkForwardBackward_GoTorch(defaultBatchSize, inputDim, hiddenDim, outputDim, numIterations/10)) // Reduced iterations for slower Fwd-Bwd

	fmt.Println("\n--- Benchmarks Complete ---")
}