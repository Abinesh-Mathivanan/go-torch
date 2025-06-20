package nn

import (
	"fmt"
	"go-torch/tensor"
	"math/rand"
	"time"
)


// Conv2D Struct implements a 2D convolutional layer fully integrated with the autograd system.
type Conv2D struct {
	Weight *tensor.Tensor // Shape: [OutChannels, InChannels, KernelHeight, KernelWidth]
	Bias   *tensor.Tensor // Shape: [OutChannels]
	Stride  int
	Padding int
}



// creates a new Conv2D layer.
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int) (*Conv2D, error) {
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	weightShape := []int{outChannels, inChannels, kernelSize, kernelSize}
	weightData := make([]float64, outChannels*inChannels*kernelSize*kernelSize)
	for i := range weightData {
		weightData[i] = (2*random.Float64() - 1) * 0.1
	}
	weights, err := tensor.NewTensor(weightShape, weightData)
	if err != nil { return nil, fmt.Errorf("failed to create weight tensor: %w", err) }
	weights.RequiresGrad = true

	biasData := make([]float64, outChannels)
	bias, err := tensor.NewTensor([]int{outChannels}, biasData)
	if err != nil { return nil, fmt.Errorf("failed to create bias tensor: %w", err) }
	bias.RequiresGrad = true

	return &Conv2D{
		Weight:  weights,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}, nil
}



// Forward performs the forward pass and builds the autograd graph.
func (c *Conv2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	wShape := c.Weight.GetShape()
	
	outChannels, inChannels, kernelHeight, kernelWidth := wShape[0], wShape[1], wShape[2], wShape[3]
	_ = inChannels 

	inputCols, err := tensor.Im2Col(input, kernelHeight, kernelWidth, c.Stride, c.Padding)
	if err != nil { return nil, fmt.Errorf("conv forward failed during im2col: %w", err) }
	
	// For the backward pass, we need a way to link the gradient flow from the MatMul
	// output back to the original `input` tensor through the `col2im` transformation.
	// We'll define a custom backward function for the MatMul operation within this context.

	// reshape kernel weights into a 2D matrix
	numKernelElements := tensor.Numel(c.Weight)
	inferredDim := numKernelElements / outChannels
	kernelMatrixShape := []int{outChannels, inferredDim}

	kernelMatrix, err := tensor.Reshape(c.Weight, kernelMatrixShape)
	if err != nil { return nil, fmt.Errorf("conv forward failed reshaping kernel: %w", err) }


	// perform the matmul. we'll set a custom backward func.
	outputMatMul, err := tensor.MatMulTensor(kernelMatrix, inputCols)
	if err != nil { return nil, fmt.Errorf("conv forward failed during matmul: %w", err) }

	if outputMatMul.RequiresGrad {
		outputMatMul.BackwardFunc = func(grad *tensor.Tensor) {
			grad2D, _ := tensor.Reshape(grad, outputMatMul.GetShape())

			if kernelMatrix.RequiresGrad {
				inputColsT, _ := tensor.Transpose(inputCols)
				gradForKernelMatrix, _ := tensor.MatMulTensor(grad2D, inputColsT)
				kernelMatrix.Backward(gradForKernelMatrix)
			}
			
			if input.RequiresGrad {
				kernelMatrixT, _ := tensor.Transpose(kernelMatrix)
				gradForInputCols, _ := tensor.MatMulTensor(kernelMatrixT, grad2D)
				gradForInput, _ := tensor.Col2Im(gradForInputCols, input.GetShape(), kernelHeight, kernelWidth, c.Stride, c.Padding)
				input.Backward(gradForInput)
			}
		}
	}
	
	// reshape the output back into an image-like format
	inShape := input.GetShape()
	batchSize := inShape[0]
	outHeight := (inShape[2] + 2*c.Padding - kernelHeight) / c.Stride + 1
	outWidth := (inShape[3] + 2*c.Padding - kernelWidth) / c.Stride + 1
	outputReshaped, err := tensor.Reshape(outputMatMul, []int{outChannels, batchSize, outHeight, outWidth})
	if err != nil { return nil, fmt.Errorf("conv forward failed reshaping output: %w", err) }

	outputPermuted, err := tensor.Permute(outputReshaped, []int{1, 0, 2, 3})
	if err != nil { return nil, fmt.Errorf("conv forward failed during permute: %w", err) }

	finalOutput, err := tensor.AddTensorBroadcast(outputPermuted, c.Bias)
	if err != nil { return nil, fmt.Errorf("conv forward failed during bias add: %w", err) }
	
	return finalOutput, nil
}


func (c *Conv2D) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{c.Weight, c.Bias}
}


func (c *Conv2D) ZeroGrad() {
	c.Weight.ZeroGrad()
	c.Bias.ZeroGrad()
}