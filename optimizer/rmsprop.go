package optimizer

import (
	"fmt"
	"go-torch/tensor"
	"math"
)

// RMSProp implements the RMSProp optimization algorithm: divide the
// learning rate by a running average of the magnitude of recent gradients
// for that parameter. Optionally supports classic momentum on top (set
// momentum to 0.0 for plain RMSProp).
type RMSProp struct {
	learningRate float64
	decay        float64 
	epsilon      float64
	momentum     float64
	parameters   []*tensor.Tensor
	sqAvg        map[*tensor.Tensor][]float64 // running average of g^2
	velocity     map[*tensor.Tensor][]float64 // momentum buffer, only used if momentum > 0
}


// creates a new RMSProp optimizer.
// Common defaults: lr=0.01 (SGD-comparable) or 0.001 (Adam-comparable),
// decay=0.99, epsilon=1e-8, momentum=0.0.
func NewRMSProp(parameters []*tensor.Tensor, learningRate, decay, epsilon, momentum float64) (*RMSProp, error) {
	if learningRate <= 0.0 {
		return nil, fmt.Errorf("optimizer: RMSProp learning rate must be positive")
	}
	if decay <= 0.0 || decay >= 1.0 {
		return nil, fmt.Errorf("optimizer: RMSProp decay must be in (0, 1)")
	}
	if momentum < 0.0 {
		return nil, fmt.Errorf("optimizer: RMSProp momentum must be non-negative, got %f", momentum)
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

	sqAvg := make(map[*tensor.Tensor][]float64)
	var velocity map[*tensor.Tensor][]float64
	if momentum > 0.0 {
		velocity = make(map[*tensor.Tensor][]float64)
	}
	for _, p := range validParams {
		sqAvg[p] = make([]float64, tensor.Numel(p))
		if momentum > 0.0 {
			velocity[p] = make([]float64, tensor.Numel(p))
		}
	}

	return &RMSProp{
		learningRate: learningRate,
		decay:        decay,
		epsilon:      epsilon,
		momentum:     momentum,
		parameters:   validParams,
		sqAvg:        sqAvg,
		velocity:     velocity,
	}, nil
}

// Step performs a single RMSProp update for all parameters.
func (r *RMSProp) Step() error {
	for _, p := range r.parameters {
		if p.Grad == nil {
			continue
		}

		paramData := p.GetData()
		gradData := p.Grad.GetData()
		sqAvg, ok := r.sqAvg[p]
		if !ok {
			return fmt.Errorf("optimizer: RMSProp state not initialized for a parameter")
		}

		if r.momentum > 0.0 {
			v := r.velocity[p]
			for i := range paramData {
				g := gradData[i]
				// running average of squared gradient: E[g^2]_t = decay*E[g^2]_{t-1} + (1-decay)*g^2
				sqAvg[i] = r.decay*sqAvg[i] + (1-r.decay)*g*g
				// v = momentum*v + g / (sqrt(E[g^2]) + eps)
				v[i] = r.momentum*v[i] + g/(math.Sqrt(sqAvg[i])+r.epsilon)
				paramData[i] -= r.learningRate * v[i]
			}
		} else {
			for i := range paramData {
				g := gradData[i]
				sqAvg[i] = r.decay*sqAvg[i] + (1-r.decay)*g*g
				paramData[i] -= r.learningRate * g / (math.Sqrt(sqAvg[i]) + r.epsilon)
			}
		}
	}
	return nil
}


func (r *RMSProp) ZeroGrad() {
	for _, p := range r.parameters {
		p.ZeroGrad()
	}
}

func (r *RMSProp) SetLR(lr float64) {
	r.learningRate = lr
}

func (r *RMSProp) GetLR() float64 {
	return r.learningRate
}