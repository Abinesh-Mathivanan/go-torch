package optimizer

import (
	"fmt"
	"go-torch/tensor"
	"math"
)


// implements the Adam optimization algorithm.
type Adam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	parameters   []*tensor.Tensor
	m            map[*tensor.Tensor][]float64 // 1st moment vector (mean)
	v            map[*tensor.Tensor][]float64 // 2nd moment vector (uncentered variance)
	t            int                          // timestep 
}


// creates a new Adam optimizer.
// Common default values are lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
func NewAdam(parameters []*tensor.Tensor, learningRate, beta1, beta2, epsilon float64) (*Adam, error) {
	if learningRate <= 0.0 {
		return nil, fmt.Errorf("optimizer: Adam learning rate must be positive")
	}
	if beta1 <= 0.0 || beta1 >= 1.0 {
		return nil, fmt.Errorf("optimizer: Adam beta1 must be in (0, 1)")
	}
	if beta2 <= 0.0 || beta2 >= 1.0 {
		return nil, fmt.Errorf("optimizer: Adam beta2 must be in (0, 1)")
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

	// pre-allocate buffers for the moment vectors for each valid parameter.
	for _, p := range validParams {
		m[p] = make([]float64, tensor.Numel(p))
		v[p] = make([]float64, tensor.Numel(p))
	}

	return &Adam{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		parameters:   validParams,
		m:            m,
		v:            v,
		t:            0,
	}, nil
}


// performs a single optimization step for all parameters.
func (a *Adam) Step() error {
	a.t++ 

	for _, p := range a.parameters {
		// skip params that lacks grad
		if p.Grad == nil {
			continue 
		}

		paramData := p.GetData()
		gradData := p.Grad.GetData()
		m_t, ok_m := a.m[p]
		v_t, ok_v := a.v[p]

		if !ok_m || !ok_v {
			return fmt.Errorf("optimizer: Adam moment vectors not initialized for a parameter")
		}

		// bias correction terms
		biasCorrection1 := 1.0 - math.Pow(a.beta1, float64(a.t))
		biasCorrection2 := 1.0 - math.Pow(a.beta2, float64(a.t))

		// apply the Adam update rule element-wise
		for i := range paramData {
			g_i := gradData[i]

			// update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
			m_t[i] = a.beta1*m_t[i] + (1-a.beta1)*g_i

			// update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
			v_t[i] = a.beta2*v_t[i] + (1-a.beta2)*(g_i*g_i)

			// compute bias-corrected first moment estimate: m_hat = m_t / (1 - beta1^t)
			m_hat := m_t[i] / biasCorrection1

			// compute bias-corrected second raw moment estimate: v_hat = v_t / (1 - beta2^t)
			v_hat := v_t[i] / biasCorrection2

			// update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
			paramData[i] -= a.learningRate * m_hat / (math.Sqrt(v_hat) + a.epsilon)
		}
	}
	return nil
}


// sets the gradients of all managed parameters to zero.
func (a *Adam) ZeroGrad() {
	for _, p := range a.parameters {
		p.ZeroGrad()
	}
}

// SetLR updates the learning rate used by subsequent Step() calls.
func (a *Adam) SetLR(lr float64) {
	a.learningRate = lr
}

// GetLR returns the current learning rate.
func (a *Adam) GetLR() float64 {
	return a.learningRate
}