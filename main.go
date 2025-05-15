// File: main.go
package main

import (
	"fmt"
	"log"

	"go-torch/nn"
	"go-torch/tensor"
	"go-torch/optimizer"
)

func main() {
	fmt.Println("--> tensor test")

	// 1. NewTensor
	shapeA := []int{2, 2}
	dataA := []float64{1, 2, 3, 4}
	tensorA, err := tensor.NewTensor(shapeA, dataA)
	if err != nil {
		log.Fatalf("Error creating tensorA: %v", err)
	}
	tensor.PrintTensor(tensorA)



	// GetData, GetShape, Numel
	fmt.Printf("TensorA Data: %v, Shape: %v, Numel: %d\n", tensorA.GetData(), tensorA.GetShape(), tensor.Numel(tensorA))



	// 2. CloneTensor
	tensorAClone := tensor.CloneTensor(tensorA)
	tensor.PrintTensor(tensorAClone)
	fmt.Println("Is tensorA same size as tensorAClone?", tensor.IsSameSize(tensorA, tensorAClone))



	// 3. OnesLike
	onesTensor, err := tensor.OnesLike(tensorA)
	if err != nil {
		log.Fatalf("Error creating onesTensor: %v", err)
	}
	fmt.Print("OnesLike(tensorA): ")
	tensor.PrintTensor(onesTensor)



	// 4. AddTensor & MulTensor
	shapeB := []int{2, 2}
	dataB := []float64{5, 6, 7, 8}
	tensorB, err := tensor.NewTensor(shapeB, dataB)
	if err != nil {
		log.Fatalf("Error creating tensorB: %v", err)
	}

	sumTensor, err := tensor.AddTensor(tensorA, tensorB)
	if err != nil {
		log.Fatalf("Error adding tensors: %v", err)
	}
	fmt.Print("Sum (A+B): ")
	tensor.PrintTensor(sumTensor)

	prodTensor, err := tensor.MulTensor(tensorA, tensorB)
	if err != nil {
		log.Fatalf("Error multiplying tensors: %v", err)
	}
	fmt.Print("Product (A*B element-wise): ")
	tensor.PrintTensor(prodTensor)



	// 5. Reshape
	reshapedSum, err := tensor.Reshape(sumTensor, []int{4, 1})
	if err != nil {
		log.Fatalf("Error reshaping tensor: %v", err)
	}
	fmt.Print("Reshaped Sum: ")
	tensor.PrintTensor(reshapedSum)
	fmt.Println()



// -------------------- Activation section -------------------- //



	fmt.Println("--> activation functions")

	dataActivation := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	tensorActivationInput, err := tensor.NewTensor([]int{1, 5}, dataActivation)
	if err != nil {
		log.Fatalf("Error creating tensorActivationInput: %v", err)
	}
	tensor.PrintTensor(tensorActivationInput)


	// RELU
	reluOut, err := nn.RELU(tensorActivationInput)
	if err != nil {
		log.Fatalf("Error in RELU: %v", err)
	}
	fmt.Print("RELU Output: ")
	tensor.PrintTensor(reluOut)



	// sigmoid
	sigmoidOut, err := nn.Sigmoid(tensorActivationInput)
	if err != nil {
		log.Fatalf("Error in Sigmoid: %v", err)
	}
	fmt.Print("Sigmoid Output: ")
	tensor.PrintTensor(sigmoidOut)



	// tanh
	tanhOut, err := nn.Tanh(tensorActivationInput)
	if err != nil {
		log.Fatalf("Error in Tanh: %v", err)
	}
	fmt.Print("Tanh Output: ")
	tensor.PrintTensor(tanhOut)



	// Softmax
	softmaxInputData := []float64{1.0, 2.0, 0.5}
	softmaxInput, err := tensor.NewTensor([]int{1, 3}, softmaxInputData)
	if err != nil {
		log.Fatalf("Error creating softmaxInput: %v", err)
	}
	fmt.Print("Softmax Input: ")
	tensor.PrintTensor(softmaxInput)
	softmaxOut, err := nn.Softmax(softmaxInput)
	if err != nil {
		log.Fatalf("Error in Softmax: %v", err)
	}
	fmt.Print("Softmax Output: ")
	tensor.PrintTensor(softmaxOut)
	fmt.Println()



// ------------------- Layer and Loss section ------------- //

	fmt.Println("--- Linear Layer, Loss, and Autograd Demo ---")


	// Define a simple network: input -> Linear -> RELU -> Loss
	inputDim := 2
	outputDim := 3
	batchSize := 1



	// 1. Create Input Tensor (requires grad)
	inputData := []float64{0.5, -0.2} // batch_size=1, input_dim=2
	x, err := tensor.NewTensor([]int{batchSize, inputDim}, inputData)
	if err != nil {
		log.Fatalf("Error creating input x: %v", err)
	}
	x.RequiresGrad = true
	fmt.Print("Input x: ")
	tensor.PrintTensor(x)




	// 2. Create Linear Layer
	linearLayer, err := nn.NewLinear(inputDim, outputDim)
	if err != nil {
		log.Fatalf("Error creating linear layer: %v", err)
	}
	fmt.Println("Linear Layer Parameters (Initial):")
	for _, p := range linearLayer.Parameters() {
		tensor.PrintTensor(p)
	}




	// 3. Forward Pass through Linear Layer
	linearOut, err := linearLayer.Forward(x)
	if err != nil {
		log.Fatalf("Error in linear layer forward pass: %v", err)
	}
	fmt.Print("Linear Layer Output: ")
	tensor.PrintTensor(linearOut)




	// 4. Forward Pass through Activation (e.g., RELU)
	activatedOut, err := nn.RELU(linearOut)
	if err != nil {
		log.Fatalf("Error in RELU after linear layer: %v", err)
	}
	fmt.Print("RELU Output (after linear): ")
	tensor.PrintTensor(activatedOut)




	// 5. Define Targets
	// For CrossEntropyLoss, targets are class indices.
	// If outputDim is 3, target could be 0, 1, or 2.
	targets := []int{1}




	// 6. Calculate Loss
	// activatedOut is now our logits
	loss, err := nn.CrossEntropyLoss(activatedOut, targets)
	if err != nil {
		log.Fatalf("Error calculating cross entropy loss: %v", err)
	}
	fmt.Print("Loss: ")
	tensor.PrintTensor(loss)




	// 7. Backward Pass
	// initial gradient for the final loss is typically 1.0
	// Our loss.Backward(nil) handles this for scalar losses.
	fmt.Println("\nPerforming Backward Pass...")
	if loss.RequiresGrad { // Should be true if x or linear layer params require grad
		loss.Backward(nil) // Pass nil, tensor.Backward will create initial grad of 1.0 if needed
	} else {
		fmt.Println("Loss does not require grad, skipping backward pass.")
	}





	// 8. Inspect Gradients
	fmt.Println("\nGradients after backward pass:")
	fmt.Print("Gradient for input x: ")
	tensor.PrintTensor(x) // x.Grad should be populated

	fmt.Println("Gradients for Linear Layer Parameters:")
	for _, p := range linearLayer.Parameters() {
		tensor.PrintTensor(p) // p.Grad should be populated
	}




	// 9. Zero Gradients
	fmt.Println("\nZeroing Gradients...")
	x.ZeroGrad()
	linearLayer.ZeroGrad()

	fmt.Println("Gradients after ZeroGrad:")
	fmt.Print("Gradient for input x (after ZeroGrad): ")
	tensor.PrintTensor(x)

	fmt.Println("Gradients for Linear Layer Parameters (after ZeroGrad):")
	for _, p := range linearLayer.Parameters() {
		tensor.PrintTensor(p)
	}



// -------------- Transpose and MatMul ------------------ //


	fmt.Println("\n--- Direct Transpose and MatMul Demo ---")
	matrixA, _ := tensor.NewTensor([]int{2,3}, []float64{1,2,3,4,5,6})
	matrixB, _ := tensor.NewTensor([]int{3,2}, []float64{7,8,9,10,11,12})



	// Transpose
	fmt.Print("Matrix A: ")
	tensor.PrintTensor(matrixA)
	matrixATransposed, err := tensor.Transpose(matrixA)
	if err != nil {
		log.Fatalf("Error transposing matrixA: %v", err)
	}
	fmt.Print("Matrix A Transposed: ")
	tensor.PrintTensor(matrixATransposed)

	fmt.Print("Matrix B: ")
	tensor.PrintTensor(matrixB)



	// MatMul
	matMulResult, err := tensor.MatMulTensor(matrixA, matrixB)
	if err != nil {
		log.Fatalf("Error in MatMulTensor(A, B): %v", err)
	}
	fmt.Print("MatMul(A, B): ")
	tensor.PrintTensor(matMulResult)


	fmt.Println("\n--- stats achieved ---")




	// ------------------- Optimizer  ------------- //



	fmt.Println("--> optimizer test")

	// simple network params 
	inputDimOpt := 2
	hiddenDimOpt := 3
	outputDimOpt := 1 



	// Input tensor
	inputDataOpt := []float64{0.8, -0.5} // batch_size=1, input_dim=2
	xOpt, err := tensor.NewTensor([]int{1, inputDimOpt}, inputDataOpt)
	if err != nil {
		log.Fatalf("Error creating input xOpt: %v", err)
	}
	xOpt.RequiresGrad = true



	// Create layers
	layer1Opt, err := nn.NewLinear(inputDimOpt, hiddenDimOpt)
	if err != nil {
		log.Fatalf("Error creating layer1Opt: %v", err)
	}

	layer2Opt, err := nn.NewLinear(hiddenDimOpt, outputDimOpt)
	if err != nil {
		log.Fatalf("Error creating layer2Opt: %v", err)
	}



	// Collect all parameters from both layers
	var allParams []*tensor.Tensor
	allParams = append(allParams, layer1Opt.Parameters()...)
	allParams = append(allParams, layer2Opt.Parameters()...)



	// Create an SGD optimizer
	learningRate := 0.1
	sgdOptimizer, err := optimizer.NewSGD(allParams, learningRate)
	if err != nil {
		log.Fatalf("Error creating SGD optimizer: %v", err)
	}
	fmt.Printf("Created SGD optimizer with learning rate %f\n", learningRate)


	// Target for the loss function
	targetsOpt := []int{0} 


	// --- one training step ---

	fmt.Println("\n--- Before Optimization Step ---")
	fmt.Println("Layer 1 Parameters (Before):") 
    for _, p := range layer1Opt.Parameters() { 
        tensor.PrintTensor(p)
    }
	fmt.Println("Layer 2 Parameters (Before):") 
    for _, p := range layer2Opt.Parameters() { 
        tensor.PrintTensor(p)
    }


	// 1. Zero gradients (important before a new backward pass)
	fmt.Println("\nZeroing Gradients...")
	sgdOptimizer.ZeroGrad()
    fmt.Println("Layer 1 Parameters Grad (After ZeroGrad): ") 
    for _, p := range layer1Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }
	fmt.Println("Layer 2 Parameters Grad (After ZeroGrad): ") 
    for _, p := range layer2Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }



	// 2. Forward pass
	fmt.Println("\nPerforming Forward Pass...")
	hOpt, err := layer1Opt.Forward(xOpt)
	if err != nil { log.Fatalf("Error in layer1Opt forward: %v", err) }

	logitsOpt, err := layer2Opt.Forward(hOpt) // Use hOpt directly if no activation here
	if err != nil { log.Fatalf("Error in layer2Opt forward: %v", err) }
	fmt.Print("Logits Output: ")
	tensor.PrintTensor(logitsOpt)



	// 3. Calculate Loss
	lossOpt, err := nn.CrossEntropyLoss(logitsOpt, targetsOpt)
	if err != nil { log.Fatalf("Error calculating lossOpt: %v", err) }
	fmt.Print("Loss: ")
	tensor.PrintTensor(lossOpt)



	// 4. Backward pass
	fmt.Println("\nPerforming Backward Pass...")
	if lossOpt.RequiresGrad {
		lossOpt.Backward(nil) 
	} else {
		fmt.Println("Loss does not require grad, cannot perform backward.")
	}

	fmt.Println("\nGradients after Backward Pass:")
	fmt.Println("Layer 1 Parameters Grad (After Backward):") 
    for _, p := range layer1Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }
	fmt.Println("Layer 2 Parameters Grad (After Backward):") 
    for _, p := range layer2Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }



	// 5. Optimizer Step
	fmt.Printf("\nPerforming Optimizer Step with LR=%f...\n", learningRate)
	err = sgdOptimizer.Step()
	if err != nil {
		log.Fatalf("Error during optimizer step: %v", err)
	}

	fmt.Println("\n--- After Optimization Step ---")
	fmt.Println("Layer 1 Parameters (After Update): ") 
    for _, p := range layer1Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }
	fmt.Println("Layer 2 Parameters (After Update): ") 
    for _, p := range layer2Opt.Parameters() { 
        tensor.PrintTensor(p) 
    }


	fmt.Println("\n--- stats achieved ---")
}