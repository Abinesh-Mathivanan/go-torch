package optimizer

import (
	"fmt"
	"go-torch/tensor"
	"math"
)

// AdamW implements Adam with decoupled weight decay.
// This is a variant of Adam that applies weight decay directly to the parameters
type AdamW struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	weightDecay  float64
	parameters   []*tensor.Tensor
	m            map[*tensor.Tensor][]float64
	v            map[*tensor.Tensor][]float64
	t            int
}

// creates a new AdamW optimizer.
// Common defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
// weightDecay=0.01.
func NewAdamW(parameters []*tensor.Tensor, learningRate, beta1, beta2, epsilon, weightDecay float64) (*AdamW, error) {
	if learningRate <= 0.0 {
		return nil, fmt.Errorf("optimizer: AdamW learning rate must be positive")
	}
	if beta1 <= 0.0 || beta1 >= 1.0 {
		return nil, fmt.Errorf("optimizer: AdamW beta1 must be in (0, 1)")
	}
	if beta2 <= 0.0 || beta2 >= 1.0 {
		return nil, fmt.Errorf("optimizer: AdamW beta2 must be in (0, 1)")
	}
	if weightDecay < 0.0 {
		return nil, fmt.Errorf("optimizer: AdamW weight decay must be non-negative, got %f", weightDecay)
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

	m := make(map[*tensor.Tensor][]float64)
	v := make(map[*tensor.Tensor][]float64)
	for _, p := range validParams {
		m[p] = make([]float64, tensor.Numel(p))
		v[p] = make([]float64, tensor.Numel(p))
	}

	return &AdamW{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		weightDecay:  weightDecay,
		parameters:   validParams,
		m:            m,
		v:            v,
		t:            0,
	}, nil
}

// Step performs a single optimization step for all parameters.
func (a *AdamW) Step() error {
	a.t++

	for _, p := range a.parameters {
		if p.Grad == nil {
			continue
		}

		paramData := p.GetData()
		gradData := p.Grad.GetData()
		m_t, ok_m := a.m[p]
		v_t, ok_v := a.v[p]
		if !ok_m || !ok_v {
			return fmt.Errorf("optimizer: AdamW moment vectors not initialized for a parameter")
		}

		biasCorrection1 := 1.0 - math.Pow(a.beta1, float64(a.t))
		biasCorrection2 := 1.0 - math.Pow(a.beta2, float64(a.t))

		for i := range paramData {
			g_i := gradData[i]

			m_t[i] = a.beta1*m_t[i] + (1-a.beta1)*g_i
			v_t[i] = a.beta2*v_t[i] + (1-a.beta2)*(g_i*g_i)

			m_hat := m_t[i] / biasCorrection1
			v_hat := v_t[i] / biasCorrection2

			// decoupled weight decay: subtract lr*weightDecay*param directly,
			// separate from the adaptive gradient term above.
			paramData[i] -= a.learningRate * (m_hat/(math.Sqrt(v_hat)+a.epsilon) + a.weightDecay*paramData[i])
		}
	}
	return nil
}


func (a *AdamW) ZeroGrad() {
	for _, p := range a.parameters {
		p.ZeroGrad()
	}
}

func (a *AdamW) SetLR(lr float64) {
	a.learningRate = lr
}

func (a *AdamW) GetLR() float64 {
	return a.learningRate
}