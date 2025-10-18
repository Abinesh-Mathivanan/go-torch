package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"strings"
	"time"

	"go-torch/autograd"
	"go-torch/nn"
	"go-torch/optimizer"
	"go-torch/tensor"
)

// BenchmarkConfig holds configuration for RNN/LSTM benchmarks
type BenchmarkConfig struct {
	BatchSize    int
	SeqLength    int
	InputSize    int
	HiddenSize   int
	Iterations   int
	ReturnSeq    bool
}

// BenchmarkResult holds the results of a benchmark run
type BenchmarkResult struct {
	Config          BenchmarkConfig
	ForwardTime     time.Duration
	BackwardTime    time.Duration
	TotalTime       time.Duration
	MemoryBefore    uint64
	MemoryAfter     uint64
	MemoryDelta     uint64
	ParameterCount  int
}

// Standard benchmark configurations
var benchmarkConfigs = []BenchmarkConfig{
	// Small configurations
	{BatchSize: 1, SeqLength: 10, InputSize: 32, HiddenSize: 64, Iterations: 100, ReturnSeq: false},
	{BatchSize: 8, SeqLength: 20, InputSize: 32, HiddenSize: 64, Iterations: 50, ReturnSeq: false},
	{BatchSize: 16, SeqLength: 50, InputSize: 64, HiddenSize: 128, Iterations: 20, ReturnSeq: false},
	
	// Medium configurations
	{BatchSize: 32, SeqLength: 100, InputSize: 128, HiddenSize: 256, Iterations: 10, ReturnSeq: false},
	{BatchSize: 16, SeqLength: 200, InputSize: 256, HiddenSize: 512, Iterations: 5, ReturnSeq: false},
	
	// Return sequence configurations
	{BatchSize: 8, SeqLength: 20, InputSize: 32, HiddenSize: 64, Iterations: 20, ReturnSeq: true},
	{BatchSize: 16, SeqLength: 50, InputSize: 64, HiddenSize: 128, Iterations: 10, ReturnSeq: true},
}

// generateSequenceData creates random input data for RNN/LSTM
func generateSequenceData(batchSize, seqLength, inputSize int) []float64 {
	totalSize := batchSize * seqLength * inputSize
	data := make([]float64, totalSize)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 // Range: [-1, 1]
	}
	return data
}

// getMemoryUsage returns current memory usage in bytes
func getMemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	return m.Alloc
}

// benchmarkRNNForward tests RNN forward pass performance
func benchmarkRNNForward(config BenchmarkConfig) (time.Duration, int, error) {
	// Create RNN
	rnn, err := nn.NewRNN(config.InputSize, config.HiddenSize, 1, config.ReturnSeq)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to create RNN: %w", err)
	}

	// Generate input data
	inputData := generateSequenceData(config.BatchSize, config.SeqLength, config.InputSize)
	input, err := tensor.NewTensor([]int{config.BatchSize, config.SeqLength, config.InputSize}, inputData)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Warmup
	for i := 0; i < 3; i++ {
		_, err := rnn.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("warmup failed: %w", err)
		}
	}

	// Benchmark
	var totalDuration time.Duration
	for i := 0; i < config.Iterations; i++ {
		start := time.Now()
		_, err := rnn.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("forward pass failed: %w", err)
		}
		totalDuration += time.Since(start)
	}

	avgDuration := totalDuration / time.Duration(config.Iterations)
	paramCount := len(rnn.Parameters())
	
	return avgDuration, paramCount, nil
}

// benchmarkLSTMForward tests LSTM forward pass performance
func benchmarkLSTMForward(config BenchmarkConfig) (time.Duration, int, error) {
	// Create LSTM
	lstm, err := nn.NewLSTM(config.InputSize, config.HiddenSize, 1, config.ReturnSeq)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to create LSTM: %w", err)
	}

	// Generate input data
	inputData := generateSequenceData(config.BatchSize, config.SeqLength, config.InputSize)
	input, err := tensor.NewTensor([]int{config.BatchSize, config.SeqLength, config.InputSize}, inputData)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Warmup
	for i := 0; i < 3; i++ {
		_, err := lstm.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("warmup failed: %w", err)
		}
	}

	// Benchmark
	var totalDuration time.Duration
	for i := 0; i < config.Iterations; i++ {
		start := time.Now()
		_, err := lstm.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("forward pass failed: %w", err)
		}
		totalDuration += time.Since(start)
	}

	avgDuration := totalDuration / time.Duration(config.Iterations)
	paramCount := len(lstm.Parameters())
	
	return avgDuration, paramCount, nil
}

// benchmarkRNNForwardBackward tests RNN forward + backward pass performance
func benchmarkRNNForwardBackward(config BenchmarkConfig) BenchmarkResult {
	result := BenchmarkResult{Config: config}
	
	// Record initial memory
	result.MemoryBefore = getMemoryUsage()

	// Create RNN and classifier
	rnn, err := nn.NewRNN(config.InputSize, config.HiddenSize, 1, false) // Last timestep only for classification
	if err != nil {
		log.Fatalf("Failed to create RNN: %v", err)
	}

	classifier, err := nn.NewLinear(config.HiddenSize, 10) // 10 classes
	if err != nil {
		log.Fatalf("Failed to create classifier: %v", err)
	}

	// Create optimizer
	allParams := append(rnn.Parameters(), classifier.Parameters()...)
	opt, err := optimizer.NewAdam(allParams, 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		log.Fatalf("Failed to create optimizer: %v", err)
	}

	result.ParameterCount = len(allParams)

	// Generate targets
	targets := make([]int, config.BatchSize)
	for i := range targets {
		targets[i] = rand.Intn(10)
	}

	var totalForwardTime, totalBackwardTime time.Duration

	// Benchmark iterations
	for i := 0; i < config.Iterations; i++ {
		// Generate fresh input data for each iteration
		inputData := generateSequenceData(config.BatchSize, config.SeqLength, config.InputSize)
		input, err := tensor.NewTensor([]int{config.BatchSize, config.SeqLength, config.InputSize}, inputData)
		if err != nil {
			log.Fatalf("Failed to create input tensor: %v", err)
		}
		input.RequiresGrad = true

		// Zero gradients
		rnn.ZeroGrad()
		classifier.ZeroGrad()

		// Forward pass timing
		forwardStart := time.Now()
		
		rnnOutput, err := rnn.Forward(input)
		if err != nil {
			log.Fatalf("RNN forward failed: %v", err)
		}

		logits, err := classifier.Forward(rnnOutput)
		if err != nil {
			log.Fatalf("Classifier forward failed: %v", err)
		}

		loss, err := nn.CrossEntropyLoss(logits, targets)
		if err != nil {
			log.Fatalf("Loss computation failed: %v", err)
		}
		
		forwardTime := time.Since(forwardStart)
		totalForwardTime += forwardTime

		// Backward pass timing
		backwardStart := time.Now()
		autograd.Backward(loss)
		backwardTime := time.Since(backwardStart)
		totalBackwardTime += backwardTime

		// Optimizer step (not timed as it's not part of core RNN computation)
		opt.Step()
	}

	result.ForwardTime = totalForwardTime / time.Duration(config.Iterations)
	result.BackwardTime = totalBackwardTime / time.Duration(config.Iterations)
	result.TotalTime = result.ForwardTime + result.BackwardTime

	// Record final memory
	result.MemoryAfter = getMemoryUsage()
	result.MemoryDelta = result.MemoryAfter - result.MemoryBefore

	return result
}

// benchmarkLSTMForwardBackward tests LSTM forward + backward pass performance
func benchmarkLSTMForwardBackward(config BenchmarkConfig) BenchmarkResult {
	result := BenchmarkResult{Config: config}
	
	// Record initial memory
	result.MemoryBefore = getMemoryUsage()

	// Create LSTM and classifier
	lstm, err := nn.NewLSTM(config.InputSize, config.HiddenSize, 1, false) // Last timestep only for classification
	if err != nil {
		log.Fatalf("Failed to create LSTM: %v", err)
	}

	classifier, err := nn.NewLinear(config.HiddenSize, 10) // 10 classes
	if err != nil {
		log.Fatalf("Failed to create classifier: %v", err)
	}

	// Create optimizer
	allParams := append(lstm.Parameters(), classifier.Parameters()...)
	opt, err := optimizer.NewAdam(allParams, 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		log.Fatalf("Failed to create optimizer: %v", err)
	}

	result.ParameterCount = len(allParams)

	// Generate targets
	targets := make([]int, config.BatchSize)
	for i := range targets {
		targets[i] = rand.Intn(10)
	}

	var totalForwardTime, totalBackwardTime time.Duration

	// Benchmark iterations
	for i := 0; i < config.Iterations; i++ {
		// Generate fresh input data for each iteration
		inputData := generateSequenceData(config.BatchSize, config.SeqLength, config.InputSize)
		input, err := tensor.NewTensor([]int{config.BatchSize, config.SeqLength, config.InputSize}, inputData)
		if err != nil {
			log.Fatalf("Failed to create input tensor: %v", err)
		}
		input.RequiresGrad = true

		// Zero gradients
		lstm.ZeroGrad()
		classifier.ZeroGrad()

		// Forward pass timing
		forwardStart := time.Now()
		
		lstmOutput, err := lstm.Forward(input)
		if err != nil {
			log.Fatalf("LSTM forward failed: %v", err)
		}

		logits, err := classifier.Forward(lstmOutput)
		if err != nil {
			log.Fatalf("Classifier forward failed: %v", err)
		}

		loss, err := nn.CrossEntropyLoss(logits, targets)
		if err != nil {
			log.Fatalf("Loss computation failed: %v", err)
		}
		
		forwardTime := time.Since(forwardStart)
		totalForwardTime += forwardTime

		// Backward pass timing
		backwardStart := time.Now()
		autograd.Backward(loss)
		backwardTime := time.Since(backwardStart)
		totalBackwardTime += backwardTime

		// Optimizer step (not timed as it's not part of core RNN computation)
		opt.Step()
	}

	result.ForwardTime = totalForwardTime / time.Duration(config.Iterations)
	result.BackwardTime = totalBackwardTime / time.Duration(config.Iterations)
	result.TotalTime = result.ForwardTime + result.BackwardTime

	// Record final memory
	result.MemoryAfter = getMemoryUsage()
	result.MemoryDelta = result.MemoryAfter - result.MemoryBefore

	return result
}

// printBenchmarkHeader prints the header for benchmark results
func printBenchmarkHeader() {
	fmt.Printf("%-8s %-8s %-8s %-8s %-8s %-12s %-12s %-12s %-8s %-8s\n",
		"Type", "Batch", "SeqLen", "InSize", "HidSize", "Forward(ms)", "Backward(ms)", "Total(ms)", "Params", "Memory(MB)")
	fmt.Println(strings.Repeat("-", 110))
}

// printBenchmarkResult prints a single benchmark result
func printBenchmarkResult(rnnType string, result BenchmarkResult) {
	fmt.Printf("%-8s %-8d %-8d %-8d %-8d %-12.3f %-12.3f %-12.3f %-8d %-8.1f\n",
		rnnType,
		result.Config.BatchSize,
		result.Config.SeqLength,
		result.Config.InputSize,
		result.Config.HiddenSize,
		float64(result.ForwardTime.Nanoseconds())/1e6,
		float64(result.BackwardTime.Nanoseconds())/1e6,
		float64(result.TotalTime.Nanoseconds())/1e6,
		result.ParameterCount,
		float64(result.MemoryDelta)/(1024*1024))
}

// runForwardOnlyBenchmarks runs forward-only benchmarks for comparison
func runForwardOnlyBenchmarks() {
	fmt.Println("\n=== Forward-Only Performance Comparison ===")
	fmt.Printf("%-8s %-8s %-8s %-8s %-8s %-12s %-8s\n",
		"Type", "Batch", "SeqLen", "InSize", "HidSize", "Time(ms)", "Params")
	fmt.Println(strings.Repeat("-", 75))

	for _, config := range benchmarkConfigs[:5] { // Use first 5 configs for forward-only
		// RNN Forward
		rnnTime, rnnParams, err := benchmarkRNNForward(config)
		if err != nil {
			log.Printf("RNN forward benchmark failed: %v", err)
			continue
		}

		// LSTM Forward
		lstmTime, lstmParams, err := benchmarkLSTMForward(config)
		if err != nil {
			log.Printf("LSTM forward benchmark failed: %v", err)
			continue
		}

		fmt.Printf("%-8s %-8d %-8d %-8d %-8d %-12.3f %-8d\n",
			"RNN", config.BatchSize, config.SeqLength, config.InputSize, config.HiddenSize,
			float64(rnnTime.Nanoseconds())/1e6, rnnParams)

		fmt.Printf("%-8s %-8d %-8d %-8d %-8d %-12.3f %-8d\n",
			"LSTM", config.BatchSize, config.SeqLength, config.InputSize, config.HiddenSize,
			float64(lstmTime.Nanoseconds())/1e6, lstmParams)

		// Performance ratio
		ratio := float64(lstmTime.Nanoseconds()) / float64(rnnTime.Nanoseconds())
		fmt.Printf("LSTM/RNN ratio: %.2fx slower\n\n", ratio)
	}
}



func main() {
	rand.Seed(time.Now().UnixNano())
	
	fmt.Println("=== Go-Torch RNN/LSTM Comprehensive Benchmark ===")
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Start time: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	
	// Run forward-only benchmarks first
	runForwardOnlyBenchmarks()

	// Run full forward-backward benchmarks
	fmt.Println("\n=== Full Forward-Backward Performance ===")
	printBenchmarkHeader()

	for _, config := range benchmarkConfigs {
		// RNN benchmark
		fmt.Printf("Running RNN benchmark: Batch=%d, Seq=%d, Hidden=%d...\n", 
			config.BatchSize, config.SeqLength, config.HiddenSize)
		rnnResult := benchmarkRNNForwardBackward(config)
		printBenchmarkResult("RNN", rnnResult)

		// LSTM benchmark
		fmt.Printf("Running LSTM benchmark: Batch=%d, Seq=%d, Hidden=%d...\n", 
			config.BatchSize, config.SeqLength, config.HiddenSize)
		lstmResult := benchmarkLSTMForwardBackward(config)
		printBenchmarkResult("LSTM", lstmResult)

		// Performance comparison
		rnnTotal := float64(rnnResult.TotalTime.Nanoseconds()) / 1e6
		lstmTotal := float64(lstmResult.TotalTime.Nanoseconds()) / 1e6
		if rnnTotal > 0 {
			ratio := lstmTotal / rnnTotal
			fmt.Printf("LSTM/RNN total time ratio: %.2fx\n", ratio)
		}
		fmt.Println()
	}

	fmt.Println("=== Benchmark Complete ===")
	fmt.Printf("End time: %s\n", time.Now().Format("2006-01-02 15:04:05"))
}