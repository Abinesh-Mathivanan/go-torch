package nn

import "go-torch/tensor"


// Layer defines the interface that all neural network layers must implement.
type Layer interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor
	ZeroGrad()
	Name() string
}

// --- Activation Layers ---

type RELUActivation struct{}
func (r *RELUActivation) Forward(input *tensor.Tensor) (*tensor.Tensor, error) { return RELU(input) }
func (r *RELUActivation) Parameters() []*tensor.Tensor { return []*tensor.Tensor{} }
func (r *RELUActivation) ZeroGrad() {}
func (r *RELUActivation) Name() string { return "ReLU" }

func NewRELU() *RELUActivation {
	return &RELUActivation{}
}