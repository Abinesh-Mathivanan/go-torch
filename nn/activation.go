package nn

import (
	"fmt" 
	"math"
	"go-torch/tensor" 
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
				// gradient of RELU is 1 if input > 0, else 0
				// apply incoming gradient element-wise: dL/dx_i = dL/dy_i * dy_i/dx_i
				// dy_i/dx_i is 1 if x_i > 0, else 0.
				gradDataForT := make([]float64, len(grad.GetData()))
				tData := t.GetData() 
				gradData := grad.GetData() 

				for i := range gradDataForT {
					if tData[i] > 0 {
						gradDataForT[i] = gradData[i] 
					} else {
						gradDataForT[i] = 0 
					}
				}

				// create gradient tensor for the parent (t)
				gradTensorForT, err := tensor.NewTensor(t.GetShape(), gradDataForT)
				if err != nil {
					fmt.Printf("Warning: Failed to create gradient tensor for RELU backward: %v\n", err)
					return 
				}

				// call t.Backward with the computed gradient tensor
				t.Backward(gradTensorForT)
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
				// gradient of Sigmoid is y * (1 - y), where y is the output.
				// apply incoming gradient element-wise: dL/dx_i = dL/dy_i * dy_i/dx_i
				// dy_i/dx_i is outData[i] * (1 - outData[i])
				gradDataForT := make([]float64, len(grad.GetData()))
				gradData := grad.GetData() 

				for i := range gradDataForT {
					s := outData[i]
					gradDataForT[i] = gradData[i] * s * (1 - s)
				}

				gradTensorForT, err := tensor.NewTensor(t.GetShape(), gradDataForT)
				if err != nil {
					fmt.Printf("Warning: Failed to create gradient tensor for Sigmoid backward: %v\n", err)
					return
				}

				t.Backward(gradTensorForT)
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
				// Apply incoming gradient element-wise: dL/dx_i = dL/dy_i * dy_i/dx_i
				// dy_i/dx_i is 1 - outData[i]^2
				gradDataForT := make([]float64, len(grad.GetData()))
				gradData := grad.GetData() // Access incoming gradient data

				for i := range gradDataForT {
					tv := outData[i] // Use the calculated output value (y)
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
// Note: A proper Softmax should operate along a specific dimension.
// This implementation treats the tensor as a flat vector.
// Softmax(x)_i = exp(x_i) / sum(exp(x_j))
//
// WARNING: The backward pass implemented here is for the *standalone* Softmax function.
// It computes dL/dx given dL/dy, where y=softmax(x).
// This is different from the common optimization where Softmax is combined with Cross-Entropy loss,
// in which case the gradient w.r.t. the input (logits) is simply y - target.
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
		// div by 0 error
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
				// compute dL/dx = dL/dy * dy/dx (Jacobian matrix multiplication)
				// For Softmax(x)->y, dy_i/dx_j = y_i * (delta_ij - y_j)
				// dL/dx_j = sum_i (dL/dy_i * dy_i/dx_j) = sum_i (dL/dy_i * y_i * (delta_ij - y_j))
				// dL/dx_j = sum_i (dL/dy_i * y_i * delta_ij) - sum_i (dL/dy_i * y_i * y_j)
				// dL/dx_j = dL/dy_j * y_j - y_j * sum_i (dL/dy_i * y_i)
				// dL/dx_j = y_j * (dL/dy_j - sum_i (dL/dy_i * y_i))
				// Let grad.GetData() be dL/dy, and r.GetData() be y.
				// We need to compute sum_i (dL/dy_i * y_i), [dot product of grad and y].

				gradDataForT := make([]float64, len(grad.GetData()))
				gradData := grad.GetData() // incoming gradient dL/dy
				softmaxOutputData := r.GetData() 

				// Compute dot product: sum_i (dL/dy_i * y_i)
				dotProduct := 0.0
				if len(gradData) != len(softmaxOutputData) {
					// indicates a mismatch in shapes, which shouldn't happen if the library is used correctly
					// and tensor.Backward check passes.
					fmt.Printf("Warning: Softmax backward shape mismatch. grad len: %d, output len: %d\n", len(gradData), len(softmaxOutputData))
					return
				}
				for i := range gradData {
					dotProduct += gradData[i] * softmaxOutputData[i]
				}

				// Compute dL/dx_j = y_j * (dL/dy_j - dotProduct)
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