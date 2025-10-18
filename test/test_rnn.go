package main

import (
	"fmt"
	"go-torch/autograd"
	"go-torch/nn"
	"go-torch/optimizer"
	"go-torch/tensor"
	"log"
	"math/rand"
	"time"
)

func testBasicRNN() {
	fmt.Println("=== Testing Basic RNN ===")
	
	// Create simple RNN: input_size=3, hidden_size=4, return_sequence=false
	rnn, err := nn.NewRNN(3, 4, 1, false)
	if err != nil {
		log.Fatalf("Failed to create RNN: %v", err)
	}

	// Create input sequence: [batch_size=2, seq_length=3, input_size=3]
	inputData := []float64{
		// Batch 1, timestep 1: [0.1, 0.2, 0.3]
		0.1, 0.2, 0.3,
		// Batch 1, timestep 2: [0.4, 0.5, 0.6]
		0.4, 0.5, 0.6,
		// Batch 1, timestep 3: [0.7, 0.8, 0.9]
		0.7, 0.8, 0.9,
		// Batch 2, timestep 1: [0.2, 0.3, 0.4]
		0.2, 0.3, 0.4,
		// Batch 2, timestep 2: [0.5, 0.6, 0.7]
		0.5, 0.6, 0.7,
		// Batch 2, timestep 3: [0.8, 0.9, 1.0]
		0.8, 0.9, 1.0,
	}

	input, err := tensor.NewTensor([]int{2, 3, 3}, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	fmt.Printf("Input shape: %v\n", input.GetShape())

	// Forward pass
	output, err := rnn.Forward(input)
	if err != nil {
		log.Fatalf("RNN forward pass failed: %v", err)
	}

	fmt.Printf("Output shape: %v\n", output.GetShape())
	fmt.Printf("Output data: %v\n", output.GetData())
	fmt.Printf("RNN Parameters count: %d\n", len(rnn.Parameters()))
}

func testBasicLSTM() {
	fmt.Println("\n=== Testing Basic LSTM ===")
	
	// Create simple LSTM: input_size=3, hidden_size=4, return_sequence=true
	lstm, err := nn.NewLSTM(3, 4, 1, true)
	if err != nil {
		log.Fatalf("Failed to create LSTM: %v", err)
	}

	// Create input sequence: [batch_size=1, seq_length=2, input_size=3]
	inputData := []float64{
		// Batch 1, timestep 1: [1.0, 0.5, -0.5]
		1.0, 0.5, -0.5,
		// Batch 1, timestep 2: [0.8, -0.3, 0.2]
		0.8, -0.3, 0.2,
	}

	input, err := tensor.NewTensor([]int{1, 2, 3}, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	fmt.Printf("Input shape: %v\n", input.GetShape())

	// Forward pass
	output, err := lstm.Forward(input)
	if err != nil {
		log.Fatalf("LSTM forward pass failed: %v", err)
	}

	fmt.Printf("Output shape: %v\n", output.GetShape())
	fmt.Printf("Output data length: %d\n", len(output.GetData()))
	fmt.Printf("LSTM Parameters count: %d\n", len(lstm.Parameters()))
}

func testRNNGradients() {
	fmt.Println("\n=== Testing RNN with Gradients ===")
	
	// Create RNN for classification task
	rnn, err := nn.NewRNN(2, 3, 1, false) // Last timestep only
	if err != nil {
		log.Fatalf("Failed to create RNN: %v", err)
	}

	// Create a simple classifier on top
	classifier, err := nn.NewLinear(3, 2) // 2 classes
	if err != nil {
		log.Fatalf("Failed to create classifier: %v", err)
	}

	// Create input sequence: [batch_size=1, seq_length=3, input_size=2]
	inputData := []float64{
		0.5, -0.2, // timestep 1
		0.3, 0.1,  // timestep 2
		-0.1, 0.7, // timestep 3
	}

	input, err := tensor.NewTensor([]int{1, 3, 2}, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}
	input.RequiresGrad = true

	// Forward pass
	rnnOutput, err := rnn.Forward(input)
	if err != nil {
		log.Fatalf("RNN forward pass failed: %v", err)
	}

	logits, err := classifier.Forward(rnnOutput)
	if err != nil {
		log.Fatalf("Classifier forward pass failed: %v", err)
	}

	// Compute loss
	targets := []int{1} // Target class 1
	loss, err := nn.CrossEntropyLoss(logits, targets)
	if err != nil {
		log.Fatalf("Loss computation failed: %v", err)
	}

	fmt.Printf("Loss: %v\n", loss.GetData())

	// Backward pass - use autograd.Backward for better integration
	fmt.Println("Running backward pass...")
	autograd.Backward(loss)

	// Check if gradients were computed
	rnnParams := rnn.Parameters()
	classifierParams := classifier.Parameters()
	
	fmt.Printf("RNN parameter gradients computed: ")
	for i, param := range rnnParams {
		if param.Grad != nil {
			fmt.Printf("param_%d: ✓ ", i)
		} else {
			fmt.Printf("param_%d: ✗ ", i)
		}
	}
	fmt.Println()

	fmt.Printf("Classifier parameter gradients computed: ")
	for i, param := range classifierParams {
		if param.Grad != nil {
			fmt.Printf("param_%d: ✓ ", i)
		} else {
			fmt.Printf("param_%d: ✗ ", i)
		}
	}
	fmt.Println()

	// Test optimizer
	allParams := append(rnnParams, classifierParams...)
	opt, err := optimizer.NewSGD(allParams, 0.01, 0.0)
	if err != nil {
		log.Fatalf("Failed to create optimizer: %v", err)
	}

	fmt.Println("Testing optimizer step...")
	err = opt.Step()
	if err != nil {
		log.Fatalf("Optimizer step failed: %v", err)
	}
	fmt.Println("Optimizer step successful!")
}

func main() {
	rand.Seed(time.Now().UnixNano())
	
	fmt.Println("Go-Torch RNN/LSTM Test Suite")
	fmt.Println("===============================")
	
	testBasicRNN()
	testBasicLSTM()
	testRNNGradients()
	
	fmt.Println("\n=== All RNN/LSTM tests completed! ===")
}