package nn

import (
	"fmt"
	"go-torch/tensor"
	"math"
)

// you definitely know RELU if you're reading this: out = max(0, t)
func RELU(t *tensor.Tensor) (*tensor.Tensor, error) {
	tData := t.GetData()
	outData := make([]float64, len(tData))
	for i, v := range tData {
		if v > 0 {
			outData[i] = v
		} else {
			outData[i] = 0
		}
	}

	// use newTensor
	r, err := tensor.NewTensor(t.GetShape(), outData)
	if err != nil {
		return nil, fmt.Errorf("relu failed to create output tensor: %w", err)
	}

	if t.RequiresGrad {
		r.RequiresGrad = true
		r.Parents = []*tensor.Tensor{t}
		r.Operation = "relu"

		// backward function receives the gradient to the output tensor (r).
		// it then compute and propagate the gradient to the input tensor (t).
		r.BackwardFunc = func(grad *tensor.Tensor) {
			if t.RequiresGrad {
				if t.Grad == nil {
					t.ZeroGrad()
				}

				// gradient of RELU is 1 if input > 0, else 0
				// apply incoming gradient element-wise: dL/dx_i = dL/dy_i * dy_i/dx_i
				// dy_i/dx_i is 1 if x_i > 0, else 0.
				gradData := grad.GetData()
				parentGradData := t.Grad.GetData() // This is now safe
				tData := t.GetData()

				for i := range parentGradData {
					if tData[i] > 0 {
						parentGradData[i] += gradData[i]
					}
				}
			}
		}
	}
	return r, nil
}

// we apply element wise sigmoid : out = 1 / (1 + exp(-t))
func Sigmoid(t *tensor.Tensor) (*tensor.Tensor, error) {
	tData := t.GetData()
	outData := make([]float64, len(tData))
	for i, v := range tData {
		outData[i] = 1.0 / (1.0 + math.Exp(-v))
	}

	r, err := tensor.NewTensor(t.GetShape(), outData)
	if err != nil {
		return nil, fmt.Errorf("sigmoid failed to create output tensor: %w", err)
	}

	if t.RequiresGrad {
		r.RequiresGrad = true
		r.Parents = []*tensor.Tensor{t}
		r.Operation = "sigmoid"

		r.BackwardFunc = func(grad *tensor.Tensor) {
			if t.RequiresGrad {
				if t.Grad == nil {
					t.ZeroGrad()
				}

				// gradient of Sigmoid is y * (1 - y), where y is the output.
				gradData := grad.GetData()
				parentGradData := t.Grad.GetData() // This is now safe

				for i := range parentGradData {
					s := outData[i]
					parentGradData[i] += gradData[i] * s * (1 - s)
				}
			}
		}
	}
	return r, nil
}


// element wise hyperbolic tangent : out = tanh(t)
func Tanh(t *tensor.Tensor) (*tensor.Tensor, error) {
	tData := t.GetData()
	outData := make([]float64, len(tData))
	for i, v := range tData {
		outData[i] = math.Tanh(v)
	}

	r, err := tensor.NewTensor(t.GetShape(), outData)
	if err != nil {
		return nil, fmt.Errorf("tanh failed to create output tensor: %w", err)
	}

	if t.RequiresGrad {
		r.RequiresGrad = true
		r.Parents = []*tensor.Tensor{t}
		r.Operation = "tanh"

		r.BackwardFunc = func(grad *tensor.Tensor) {
			if t.RequiresGrad {
				// Gradient of Tanh is 1 - y^2, where y is the output.
				gradDataForT := make([]float64, len(grad.GetData()))
				gradData := grad.GetData()

				for i := range gradDataForT {
					tv := outData[i]
					gradDataForT[i] = gradData[i] * (1 - tv*tv)
				}

				gradTensorForT, err := tensor.NewTensor(t.GetShape(), gradDataForT)
				if err != nil {
					fmt.Printf("Warning: Failed to create gradient tensor for Tanh backward: %v\n", err)
					return
				}

				t.Backward(gradTensorForT)
			}
		}
	}
	return r, nil
}

// Softmax applies the Softmax function to the flattened tensor.
func Softmax(t *tensor.Tensor) (*tensor.Tensor, error) {
	tData := t.GetData()
	outData := make([]float64, len(tData))

	// max for numerical stability (log-sum-exp trick)
	maxv := tData[0]
	for _, v := range tData {
		if v > maxv {
			maxv = v
		}
	}

	// compute exp and sum
	var sum float64
	for i, v := range tData {
		exp := math.Exp(v - maxv)
		outData[i] = exp
		sum += exp
	}

	// normalize -> probabilities
	if sum == 0 {
		return nil, fmt.Errorf("softmax sum is zero, cannot normalize")
	}
	for i := range outData {
		outData[i] /= sum
	}

	r, err := tensor.NewTensor(t.GetShape(), outData)
	if err != nil {
		return nil, fmt.Errorf("softmax failed to create output tensor: %w", err)
	}

	if t.RequiresGrad {
		r.RequiresGrad = true
		r.Parents = []*tensor.Tensor{t}
		r.Operation = "softmax"

		r.BackwardFunc = func(grad *tensor.Tensor) {
			if t.RequiresGrad {
				gradDataForT := make([]float64, len(grad.GetData()))
				gradData := grad.GetData()
				softmaxOutputData := r.GetData()

				dotProduct := 0.0
				if len(gradData) != len(softmaxOutputData) {
					fmt.Printf("Warning: Softmax backward shape mismatch. grad len: %d, output len: %d\n", len(gradData), len(softmaxOutputData))
					return
				}
				for i := range gradData {
					dotProduct += gradData[i] * softmaxOutputData[i]
				}

				for i := range gradDataForT {
					gradDataForT[i] = softmaxOutputData[i] * (gradData[i] - dotProduct)
				}

				gradTensorForT, err := tensor.NewTensor(t.GetShape(), gradDataForT)
				if err != nil {
					fmt.Printf("Warning: Failed to create gradient tensor for Softmax backward: %v\n", err)
					return
				}

				t.Backward(gradTensorForT)
			}
		}
	}
	return r, nil
}