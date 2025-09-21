package nn

import (
	"go-torch/tensor"
)

// Flatten reshapes a multi-dimensional tensor into a 2D tensor [Batch, Features].
type Flatten struct{}


func NewFlatten() *Flatten {
	return &Flatten{}
}


func (f *Flatten) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	batchSize := inputShape[0]
	numFeatures := tensor.Numel(input) / batchSize
	// Reshape is already autograd-aware
	return tensor.Reshape(input, []int{batchSize, numFeatures})
}

func (f *Flatten) Parameters() []*tensor.Tensor { return []*tensor.Tensor{} }
func (f *Flatten) ZeroGrad() {}

func (f *Flatten) Name() string {
	return "Flatten"
}