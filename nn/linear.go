package nn

import (
	"fmt"
	"math/rand"
	"time"
	"go-torch/tensor"
)


// linear dense layer: output = input @ weight + bias
type Linear struct {
	weight *tensor.Tensor // Shape: [inputDimensions, outputDimensions]
	bias   *tensor.Tensor   // Shape: [outputDimensions]
}




// NewLinear creates a new Linear layer with randomly initialized weights and biases.
// The weights and biases are set to RequireGrad=true by default as they are parameters.
func NewLinear(inputDimensions, outputDimensions int) (*Linear, error) {
	if inputDimensions <= 0 || outputDimensions <= 0 {
		return nil, fmt.Errorf("linear layer dimensions must be positive, got input %d, output %d", inputDimensions, outputDimensions)
	}

	random := rand.New(rand.NewSource(time.Now().UnixNano()))

	//TODO: simple random initialization for now, consider scaling the weight assignment.
	weightData := make([]float64, inputDimensions*outputDimensions)
	for i := range weightData {
		weightData[i] = 2*random.Float64() - 1
	}

	biasData := make([]float64, outputDimensions)
	for i := range biasData {
		biasData[i] = 2*random.Float64() - 1 
	}

	weights, err := tensor.NewTensor([]int{inputDimensions, outputDimensions}, weightData)
	if err != nil {
		return nil, fmt.Errorf("linear layer failed to create weight tensor: %w", err)
	}
	// Weights are parameters, they require gradients by default
	weights.RequiresGrad = true

	bias, err := tensor.NewTensor([]int{outputDimensions}, biasData)
	if err != nil {
		return nil, fmt.Errorf("linear layer failed to create bias tensor: %w", err)
	}
	// Biases are parameters, they require gradients by default
	bias.RequiresGrad = true


	return &Linear{weight: weights, bias: bias}, nil
}




// Forward performs the forward pass of the Linear layer, with input and output tensor of shape [batch_size, input_dimensions].
func (l *Linear) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	if len(inputShape) != 2 {
		// early return, invalid shape.
		return nil, fmt.Errorf("linear layer expects 2D input tensor [batch_size, input_dimensions], got shape %v", inputShape)
	}
	batchSize := inputShape[0]
	inputDims := inputShape[1]

	weightShape := l.weight.GetShape()
	weightInputDims := weightShape[0] 
	outputDims := weightShape[1]

	if inputDims != weightInputDims {
		return nil, fmt.Errorf("linear layer input dimension mismatch: input %d, weight expected %d", inputDims, weightInputDims)
	}

	biasShape := l.bias.GetShape()
	if len(biasShape) != 1 || biasShape[0] != outputDims {
		return nil, fmt.Errorf("linear layer bias dimension mismatch: bias shape %v, expected [%d]", biasShape, outputDims)
	}


	// --- Matrix Multiplication ---
	// perform input @ weight
	// input: [batch_size, input_dimensions]
	// weight: [input_dimensions, output_dimensions]
	// result (step): [batch_size, output_dimensions]
	step, err := tensor.MatMulTensor(input, l.weight)
	if err != nil {
		return nil, fmt.Errorf("linear layer matmul failed: %w", err)
	}

	// --- Bias Addition ---
	// add bias to the result of matmul.
	// step shape: [batch_size, output_dimensions]
	// bias shape: [output_dimensions]
	// We need to broadcast bias to match step shape for element-wise addition.
	// TODO: i implemented a broadcastedBiasData: which is inefficient. consider changing. 
	broadcastedBiasData := make([]float64, tensor.Numel(step))
	biasData := l.bias.GetData()
	outputNumel := outputDims 

	for i := 0; i < batchSize; i++ {
		startIdx := i * outputNumel
		copy(broadcastedBiasData[startIdx:startIdx+outputNumel], biasData)
	}

	// Create a tensor for the broadcasted bias
	broadcastedBiasTensor, err := tensor.NewTensor(step.GetShape(), broadcastedBiasData)
	if err != nil {
         return nil, fmt.Errorf("linear layer failed to create broadcasted bias tensor: %w", err)
    }
    // The broadcasted bias tensor should inherit RequiresGrad from the original bias
    // This is crucial for gradient flow to the bias parameter
    broadcastedBiasTensor.RequiresGrad = l.bias.RequiresGrad
    broadcastedBiasTensor.Parents = []*tensor.Tensor{l.bias} 
    broadcastedBiasTensor.Operation = "broadcast_bias" 


	// add the broadcasted bias
	output, err := tensor.AddTensor(step, broadcastedBiasTensor)
	if err != nil {
		return nil, fmt.Errorf("linear layer bias addition failed: %w", err)
	}

    // The output tensor's RequiresGrad, Parents, and BackwardFunc are automatically
    // set by the AddTensor operation based on its parents (step and broadcastedBiasTensor),
    // which in turn link back to the original input, weights, and bias.

	return output, nil
}



// Parameters() returns the list of parameters in the layer that require gradients. i feed this for optimizers.
func (l *Linear) Parameters() []*tensor.Tensor {
    params := []*tensor.Tensor{}
    if l.weight != nil && l.weight.RequiresGrad {
        params = append(params, l.weight)
    }
     if l.bias != nil && l.bias.RequiresGrad {
        params = append(params, l.bias)
    }
    return params
}



// ZeroGrad() calls ZeroGrad() on all parameters in the layer.
func (l *Linear) ZeroGrad() {
     if l.weight != nil {
        l.weight.ZeroGrad()
     }
     if l.bias != nil {
        l.bias.ZeroGrad()
     }
}