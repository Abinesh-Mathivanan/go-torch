package optimizer

import (
	"fmt"
	"go-torch/tensor"
)

// the interface that all optimizers must implement.
type Optimizer interface {
	Step() error
	ZeroGrad()
}

// implements the Stochastic Gradient Descent optimizer, with optional momentum.
type SGD struct {
	learningRate float32
	momentum     float32
	parameters   []*tensor.Tensor
	velocities   map[*tensor.Tensor][]float32 // Stores momentum velocities
}

// creates a new SGD optimizer.
// for standard SGD, set momentum to 0.0. A common value for momentum is 0.9.
func NewSGD(parameters []*tensor.Tensor, learningRate float32, momentum float32) (*SGD, error) {
	if learningRate <= 0 {
		return nil, fmt.Errorf("optimizer: learning rate must be positive, got %f", learningRate)
	}
	if momentum < 0.0 {
		return nil, fmt.Errorf("optimizer: momentum must be non-negative, got %f", momentum)
	}
	if len(parameters) == 0 {
		return nil, fmt.Errorf("optimizer: created with empty parameters list")
	}

	validParams := []*tensor.Tensor{}
	for _, p := range parameters {
		if p != nil && p.RequiresGrad {
			validParams = append(validParams, p)
		}
	}
	if len(validParams) == 0 {
		return nil, fmt.Errorf("optimizer: no parameters requiring gradients provided")
	}

	// initialize velocity buffers only if momentum is being used.
	velocities := make(map[*tensor.Tensor][]float32)
	if momentum > 0.0 {
		for _, p := range validParams {
			velocities[p] = make([]float32, tensor.Numel(p))
		}
	}

	return &SGD{
		learningRate: learningRate,
		momentum:     momentum,
		parameters:   validParams,
		velocities:   velocities,
	}, nil
}

// updates the parameters based on their gradients using the SGD rule.
func (s *SGD) Step() error {
	for _, p := range s.parameters {
		if p.Grad == nil {
			continue
		}

		if !tensor.IsSameSize(p, p.Grad) {
			return fmt.Errorf("optimizer: gradient size mismatch for parameter shape %v: grad shape %v",
				p.GetShape(), p.Grad.GetShape())
		}

		paramData := p.GetData()
		gradData := p.Grad.GetData()

		if s.momentum > 0.0 {
			v, ok := s.velocities[p]
			if !ok {
				return fmt.Errorf("optimizer: velocity buffer not found for parameter; this should not happen")
			}
			for i := range paramData {
				// v = momentum * v + grad
				v[i] = s.momentum*v[i] + gradData[i]
				// p = p - lr * v
				paramData[i] -= s.learningRate * v[i]
			}
		} else {
			for i := range paramData {
				paramData[i] -= s.learningRate * gradData[i]
			}
		}
	}
	return nil
}

// sets the gradients of all managed parameters to zero.
func (s *SGD) ZeroGrad() {
	for _, p := range s.parameters {
		p.ZeroGrad()
	}
}
