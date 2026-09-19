package optimizer

import (
	"go-torch/tensor"
	"math"
)

// ClipGradNorm computes the global L2 norm of the gradients across all
// given parameters (as if they were one flat vector) and, if that norm
// exceeds maxNorm, scales every gradient down by the same factor so the
// global norm becomes exactly maxNorm. 
func ClipGradNorm(parameters []*tensor.Tensor, maxNorm float64) float64 {
	if maxNorm <= 0 {
		return 0
	}

	var sumSq float64
	for _, p := range parameters {
		if p == nil || p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.GetData() {
			sumSq += g * g
		}
	}
	totalNorm := math.Sqrt(sumSq)

	if totalNorm > maxNorm && totalNorm > 0 {
		scale := maxNorm / totalNorm
		for _, p := range parameters {
			if p == nil || p.Grad == nil {
				continue
			}
			gradData := p.Grad.GetData()
			for i := range gradData {
				gradData[i] *= scale
			}
		}
	}

	return totalNorm
}

// ClipGradValue clips each individual gradient element to [-clipValue,
// clipValue]. Simpler and cheaper than ClipGradNorm (no need to gather a
// global norm across every parameter first), but less principled: it can
// change the direction of the gradient vector, where norm clipping only
// changes its magnitude. 
func ClipGradValue(parameters []*tensor.Tensor, clipValue float64) {
	if clipValue <= 0 {
		return
	}
	for _, p := range parameters {
		if p == nil || p.Grad == nil {
			continue
		}
		gradData := p.Grad.GetData()
		for i, g := range gradData {
			if g > clipValue {
				gradData[i] = clipValue
			} else if g < -clipValue {
				gradData[i] = -clipValue
			}
		}
	}
}