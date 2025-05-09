package main

import (
	"fmt"
	"log"

	"go-torch/nn"
	"go-torch/tensor"
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
}