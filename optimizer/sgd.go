package optimizer

import (
	"fmt"
	"go-torch/tensor"
)


// common method all optimizers must utilize
type Optimizer interface {
	Step() error
	ZeroGrad()
	Parameters() []*tensor.Tensor // return the parameters managed by the optimizer
}



// SGD : Stochastic Gradient Descent optimizer.
type SGD struct {
	learningRate float64
	parameters   []*tensor.Tensor // tensors whose gradients will be updated
}



// creates a new SGD and recieves list of parameters (tensors with RequiresGrad=true) and a learning rate.
func NewSGD(parameters []*tensor.Tensor, learningRate float64) (*SGD, error) {
	if learningRate <= 0 {
		return nil, fmt.Errorf("optimizer: learning rate must be positive, got %f", learningRate)
	}
	if len(parameters) == 0 {
		return nil, fmt.Errorf("optimizer: created with empty parameters list")
	}


	// filter out params that don't require grad - shouldn't happen if usage is correct, but safe
	validParams := []*tensor.Tensor{}
	for _, p := range parameters {
		if p != nil && p.RequiresGrad {
			validParams = append(validParams, p)
		} else if p != nil {
			// fmt.Printf("Warning: Optimizer skipping parameter (op='%s', shape=%v) as it does not require grad.\n", p.Operation, p.GetShape())
		}
	}

	if len(validParams) == 0 {
		return nil, fmt.Errorf("optimizer: no parameters requiring gradients provided")
	}

	return &SGD{
		learningRate: learningRate,
		parameters:   validParams,
	}, nil
}



// step updates the parameters based on their gradients using the SGD rule:
// parameter = parameter - learning_rate * gradient
func (s *SGD) Step() error {
	for _, p := range s.parameters {
		if p.Grad == nil {
			// this is often okay if a parameter didn't participate in the forward pass
			// leading to the loss, but it could also indicate an issue. Warn and skip.
			continue
		}

		if !tensor.IsSameSize(p, p.Grad) {
			return fmt.Errorf("optimizer: gradient size mismatch for parameter (op='%s', shape=%v): grad shape %v, parameter shape %v",
				p.Operation, p.GetShape(), p.Grad.GetShape(), p.GetShape())
		}

		// apply SGD element-wise
		paramData := p.GetData()
		gradData := p.Grad.GetData()

		if len(paramData) != len(gradData) {
			return fmt.Errorf("optimizer: internal error: data length mismatch for parameter (op='%s', shape=%v) and grad (shape=%v)",
				p.Operation, p.GetShape(), p.Grad.GetShape())
		}

		for i := range paramData {
			paramData[i] -= s.learningRate * gradData[i]
		}
	}
	return nil 
}



// sets all params managed by this to zero
func (s *SGD) ZeroGrad() {
	for _, p := range s.parameters {
		p.ZeroGrad()
	}
}



// returns the slice of params managed by this optimizer 
func (s *SGD) Parameters() []*tensor.Tensor {
    return s.parameters
}