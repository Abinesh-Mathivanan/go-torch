// FILE: main.go
package main

import (
	"fmt"
	"go-torch/nn"
	"go-torch/optimizer"
	"go-torch/tensor"
	"log"
	"math/rand"
	"time"
)

// --- Model Definition ---

// SimpleCNN is a basic LeNet-style model for image classification.type SimpleCNN struct {
type SimpleCNN struct {
	Conv1   *nn.Conv2D
	Pool1   *nn.MaxPooling2D
	Conv2   *nn.Conv2D
	Pool2   *nn.MaxPooling2D
	Flatten *nn.Flatten
	Linear1 *nn.Linear
	Linear2 *nn.Linear
}

// NewSimpleCNN creates and initializes a new CNN model.
func NewSimpleCNN(numClasses int) (*SimpleCNN, error) {
	conv1, err := nn.NewConv2D(1, 16, 5, 1, 2) // In: 1x28x28, Out: 16x28x28
	if err != nil { return nil, err }
	pool1 := nn.NewMaxPooling2D(2, 2) // Out: 16x14x14

	conv2, err := nn.NewConv2D(16, 32, 5, 1, 2) // In: 16x14x14, Out: 32x14x14
	if err != nil { return nil, err }
	pool2 := nn.NewMaxPooling2D(2, 2) // Out: 32x7x7

	flatten := nn.NewFlatten() // Out: 32*7*7 = 1568 features

	linear1, err := nn.NewLinear(32*7*7, 128)
	if err != nil { return nil, err }

	linear2, err := nn.NewLinear(128, numClasses)
	if err != nil { return nil, err }

	return &SimpleCNN{
		Conv1:   conv1,
		Pool1:   pool1,
		Conv2:   conv2,
		Pool2:   pool2,
		Flatten: flatten,
		Linear1: linear1,
		Linear2: linear2,
	}, nil
}

// Forward pass for the entire CNN model.
func (m *SimpleCNN) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	x, err = m.Conv1.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }
	x, err = m.Pool1.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Conv2.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }
	x, err = m.Pool2.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Flatten.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Linear1.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }

	logits, err := m.Linear2.Forward(x)
	if err != nil { return nil, err }

	return logits, nil
}

// Parameters returns all learnable parameters of the model.
func (m *SimpleCNN) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Conv1.Parameters()...)
	params = append(params, m.Conv2.Parameters()...)
	params = append(params, m.Linear1.Parameters()...)
	params = append(params, m.Linear2.Parameters()...)
	return params
}

// ZeroGrad calls ZeroGrad on all layers with parameters.
func (m *SimpleCNN) ZeroGrad() {
	m.Conv1.ZeroGrad()
	m.Conv2.ZeroGrad()
	m.Linear1.ZeroGrad()
	m.Linear2.ZeroGrad()
}

// --- Demos ---

func tensorDemo() {
	fmt.Println("--> tensor test")

	shapeA := []int{2, 2}
	dataA := []float64{1, 2, 3, 4}
	tensorA, err := tensor.NewTensor(shapeA, dataA)
	if err != nil { log.Fatalf("Error creating tensorA: %v", err) }
	tensor.PrintTensor(tensorA)
	fmt.Printf("TensorA Data: %v, Shape: %v, Numel: %d\n", tensorA.GetData(), tensorA.GetShape(), tensor.Numel(tensorA))

	tensorAClone := tensor.CloneTensor(tensorA)
	tensor.PrintTensor(tensorAClone)
	fmt.Println("Is tensorA same size as tensorAClone?", tensor.IsSameSize(tensorA, tensorAClone))

	onesTensor, err := tensor.OnesLike(tensorA)
	if err != nil { log.Fatalf("Error creating onesTensor: %v", err) }
	fmt.Print("OnesLike(tensorA): ")
	tensor.PrintTensor(onesTensor)

	shapeB := []int{2, 2}
	dataB := []float64{5, 6, 7, 8}
	tensorB, err := tensor.NewTensor(shapeB, dataB)
	if err != nil { log.Fatalf("Error creating tensorB: %v", err) }

	sumTensor, err := tensor.AddTensor(tensorA, tensorB)
	if err != nil { log.Fatalf("Error adding tensors: %v", err) }
	fmt.Print("Sum (A+B): ")
	tensor.PrintTensor(sumTensor)

	prodTensor, err := tensor.MulTensor(tensorA, tensorB)
	if err != nil { log.Fatalf("Error multiplying tensors: %v", err) }
	fmt.Print("Product (A*B element-wise): ")
	tensor.PrintTensor(prodTensor)

	reshapedSum, err := tensor.Reshape(sumTensor, []int{4, 1})
	if err != nil { log.Fatalf("Error reshaping tensor: %v", err) }
	fmt.Print("Reshaped Sum: ")
	tensor.PrintTensor(reshapedSum)
}

func activationDemo() {
	fmt.Println("\n--> activation functions")
	dataActivation := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	tensorActivationInput, err := tensor.NewTensor([]int{1, 5}, dataActivation)
	if err != nil { log.Fatalf("Error creating tensorActivationInput: %v", err) }
	tensor.PrintTensor(tensorActivationInput)

	reluOut, err := nn.RELU(tensorActivationInput)
	if err != nil { log.Fatalf("Error in RELU: %v", err) }
	fmt.Print("RELU Output: ")
	tensor.PrintTensor(reluOut)

	sigmoidOut, err := nn.Sigmoid(tensorActivationInput)
	if err != nil { log.Fatalf("Error in Sigmoid: %v", err) }
	fmt.Print("Sigmoid Output: ")
	tensor.PrintTensor(sigmoidOut)

	tanhOut, err := nn.Tanh(tensorActivationInput)
	if err != nil { log.Fatalf("Error in Tanh: %v", err) }
	fmt.Print("Tanh Output: ")
	tensor.PrintTensor(tanhOut)

	softmaxInputData := []float64{1.0, 2.0, 0.5}
	softmaxInput, err := tensor.NewTensor([]int{1, 3}, softmaxInputData)
	if err != nil { log.Fatalf("Error creating softmaxInput: %v", err) }
	fmt.Print("Softmax Input: ")
	tensor.PrintTensor(softmaxInput)
	softmaxOut, err := nn.Softmax(softmaxInput)
	if err != nil { log.Fatalf("Error in Softmax: %v", err) }
	fmt.Print("Softmax Output: ")
	tensor.PrintTensor(softmaxOut)
}

func linearAutogradDemo() {
	fmt.Println("\n--- Linear Layer, Loss, and Autograd Demo ---")
	inputDim, outputDim, batchSize := 2, 3, 1
	inputData := []float64{0.5, -0.2}
	x, err := tensor.NewTensor([]int{batchSize, inputDim}, inputData)
	if err != nil { log.Fatalf("Error creating input x: %v", err) }
	x.RequiresGrad = true
	fmt.Print("Input x: "); tensor.PrintTensor(x)

	linearLayer, err := nn.NewLinear(inputDim, outputDim)
	if err != nil { log.Fatalf("Error creating linear layer: %v", err) }

	linearOut, err := linearLayer.Forward(x)
	if err != nil { log.Fatalf("Error in linear layer forward pass: %v", err) }
	
	activatedOut, err := nn.RELU(linearOut)
	if err != nil { log.Fatalf("Error in RELU after linear layer: %v", err) }
	
	targets := []int{1}
	loss, err := nn.CrossEntropyLoss(activatedOut, targets)
	if err != nil { log.Fatalf("Error calculating cross entropy loss: %v", err) }
	fmt.Print("Loss: "); tensor.PrintTensor(loss)

	fmt.Println("\nPerforming Backward Pass..."); loss.Backward(nil)

	fmt.Println("\nGradients after backward pass:")
	fmt.Print("Gradient for input x: "); tensor.PrintTensor(x)
	fmt.Println("Gradients for Linear Layer Parameters:");
	for _, p := range linearLayer.Parameters() {
		tensor.PrintTensor(p)
	}

	fmt.Println("\nZeroing Gradients..."); linearLayer.ZeroGrad()
}

func optimizerDemo() {
	fmt.Println("\n--> optimizer test")
	inputDim, hiddenDim, outputDim := 2, 3, 1
	inputData := []float64{0.8, -0.5}
	x, err := tensor.NewTensor([]int{1, inputDim}, inputData)
	if err != nil { log.Fatalf("Error creating input x: %v", err) }
	x.RequiresGrad = true

	layer1, err := nn.NewLinear(inputDim, hiddenDim); if err != nil { log.Fatalf("Error creating layer1: %v", err) }
	layer2, err := nn.NewLinear(hiddenDim, outputDim); if err != nil { log.Fatalf("Error creating layer2: %v", err) }
	
	var allParams []*tensor.Tensor
	allParams = append(allParams, layer1.Parameters()...)
	allParams = append(allParams, layer2.Parameters()...)

	learningRate := 0.1
	sgdOptimizer, err := optimizer.NewSGD(allParams, learningRate)
	if err != nil { log.Fatalf("Error creating SGD optimizer: %v", err) }
	
	fmt.Println("\n--- Before Optimization Step ---")
	fmt.Println("Layer 1 Weight: "); tensor.PrintTensor(layer1.Parameters()[0])

	sgdOptimizer.ZeroGrad()
	h, _ := layer1.Forward(x)
	logits, _ := layer2.Forward(h)
	targets := []int{0}
	loss, _ := nn.CrossEntropyLoss(logits, targets)
	
	loss.Backward(nil)
	
	err = sgdOptimizer.Step()
	if err != nil { log.Fatalf("Error during optimizer step: %v", err) }

	fmt.Println("\n--- After Optimization Step ---")
	fmt.Println("Layer 1 Weight (After Update): "); tensor.PrintTensor(layer1.Parameters()[0])
}

func cnnDemo() {
	fmt.Println("\n--- CNN Full Forward & Backward Demo ---")

	model, err := NewSimpleCNN(10)
	if err != nil { log.Fatalf("Error creating model: %v", err) }

	batchSize := 2
	inputShape := []int{batchSize, 1, 28, 28}

	// FIX #1: Calculate numel directly from the shape slice, not from a struct literal.
	numElements := 1
	for _, dim := range inputShape {
		numElements *= dim
	}
	inputData := make([]float64, numElements)

	for i := range inputData {
		inputData[i] = rand.Float64()
	}
	input, _ := tensor.NewTensor(inputShape, inputData)
	input.RequiresGrad = true

	targets := []int{3, 7}
	model.ZeroGrad()

	fmt.Println("Running forward pass...")
	logits, err := model.Forward(input)
	if err != nil { log.Fatalf("Forward pass failed: %v", err) }
	fmt.Println("Logits (model output):"); tensor.PrintTensor(logits)

	loss, err := nn.CrossEntropyLoss(logits, targets)
	if err != nil { log.Fatalf("Loss calculation failed: %v", err) }
	fmt.Println("\nLoss:"); tensor.PrintTensor(loss)

	fmt.Println("\nRunning backward pass..."); loss.Backward(nil)
	fmt.Println("Backward pass complete.")

	fmt.Println("\nVerifying gradients were computed...")

	conv1WeightGrad := model.Conv1.Weight.Grad // .Weight is public in Conv2D
	if conv1WeightGrad != nil {
		fmt.Printf("Gradient computed for Conv1 Weights, shape: %v\n", conv1WeightGrad.GetShape())
	} else {
		fmt.Println("Gradient for Conv1 Weights is NIL.")
	}

	// FIX #2: Access parameters via the public Parameters() method.
	// Parameters() for Linear returns [weight, bias]. So bias is at index 1.
	linear1Params := model.Linear1.Parameters()
	if len(linear1Params) > 1 {
		linear1BiasGrad := linear1Params[1].Grad
		if linear1BiasGrad != nil {
			fmt.Printf("Gradient computed for Linear1 Bias, shape: %v\n", linear1BiasGrad.GetShape())
		} else {
			fmt.Println("Gradient for Linear1 Bias is NIL.")
		}
	}


	if input.Grad != nil {
		fmt.Printf("Gradient computed for input tensor, shape: %v\n", input.Grad.GetShape())
	} else {
		fmt.Println("Gradient for input tensor is NIL.")
	}
}


func main() {
	rand.Seed(time.Now().UnixNano())

	tensorDemo()
	activationDemo()
	linearAutogradDemo()
	optimizerDemo()
	cnnDemo()

	fmt.Println("\n\n--- All Demos Complete ---")
}