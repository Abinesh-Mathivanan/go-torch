package nn

import (
	"fmt"
	"go-torch/tensor"
	"math/rand"
)

// randomly sets a fraction of input units to 0 during training.
type Dropout struct {
	p           float32 // probability of an element to be zeroed
	training    bool
	mask        *tensor.Tensor // dropout mask for the backward pass
	inputTensor *tensor.Tensor // input cache
}

// creates a new Dropout layer.
func NewDropout(p float32) *Dropout {
	return &Dropout{
		p:        p,
		training: true,
	}
}

func (d *Dropout) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	d.inputTensor = input

	// in eval mode or if p=0, it's a pass-through layer
	if !d.training || d.p == 0 {
		return input, nil
	}

	// in train
	maskData := make([]float32, tensor.Numel(input))
	scale := 1.0 / (1.0 - d.p)
	for i := range maskData {

		if rand.Float32() > d.p {
			maskData[i] = scale
		} else {
			maskData[i] = 0
		}

	}
	d.mask, _ = tensor.NewTensor(input.GetShape(), maskData)

	output, err := tensor.MulTensor(input, d.mask)
	if err != nil {
		return nil, err
	}

	if input.RequiresGrad {
		output.RequiresGrad = true
		output.Parents = []*tensor.Tensor{input}
		output.Operation = "dropout"
		output.BackwardFunc = d.backward
	}

	return output, nil
}

func (d *Dropout) backward(grad *tensor.Tensor) {
	// grad only flows through the neurons that were not dropped out.
	if d.inputTensor != nil && d.inputTensor.RequiresGrad {

		gradInput, _ := tensor.MulTensor(grad, d.mask)
		d.inputTensor.Backward(gradInput)

	}
}

func (d *Dropout) Parameters() []*tensor.Tensor { return []*tensor.Tensor{} }
func (d *Dropout) ZeroGrad()                    {}
func (d *Dropout) Name() string                 { return fmt.Sprintf("Dropout(p=%.2f)", d.p) }
func (d *Dropout) Train()                       { d.training = true }
func (d *Dropout) Eval()                        { d.training = false }
