package nn

import (
	"fmt"
	"go-torch/tensor"
	"math"
)

// LayerNorm normalizes each sample independently across its feature
// dimension: for input [B, F], each row's F values are normalized to zero
// mean / unit variance (then scaled by Weight and shifted by Bias, both
// [F]), with no dependency between rows.

type LayerNorm struct {
	Weight, Bias *tensor.Tensor // [F] (gamma, beta)
	epsilon      float64
	features     int
}

// NewLayerNorm creates a new LayerNorm for inputs with 'features' columns.
// A common default for epsilon is 1e-5.
func NewLayerNorm(features int, epsilon float64) (*LayerNorm, error) {
	if features <= 0 {
		return nil, fmt.Errorf("layernorm: features must be positive, got %d", features)
	}

	weight, err := tensor.NewTensor([]int{features}, nil)
	if err != nil {
		return nil, err
	}
	for i := range weight.GetData() {
		weight.GetData()[i] = 1.0
	}
	weight.RequiresGrad = true

	bias, err := tensor.NewTensor([]int{features}, nil)
	if err != nil {
		return nil, err
	}
	bias.RequiresGrad = true

	return &LayerNorm{
		Weight:   weight,
		Bias:     bias,
		epsilon:  epsilon,
		features: features,
	}, nil
}

// Forward normalizes a [B, F] input across F (per row).
func (ln *LayerNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.GetShape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("layernorm: expects a 2D [batch, features] input, got shape %v", shape)
	}
	B, F := shape[0], shape[1]
	if F != ln.features {
		return nil, fmt.Errorf("layernorm: input has %d features, layer configured for %d", F, ln.features)
	}

	inData := input.GetData()
	outData := make([]float64, B*F)
	// cache per-row mean/invStd and the normalized (x-hat) values - the
	// backward pass needs all three, and recomputing mean/var there would
	// mean looping over the data twice for no reason.
	rowMean := make([]float64, B)
	rowInvStd := make([]float64, B)
	normData := make([]float64, B*F)

	gammaData := ln.Weight.GetData()
	betaData := ln.Bias.GetData()

	for i := 0; i < B; i++ {
		rowStart := i * F
		row := inData[rowStart : rowStart+F]

		var sum float64
		for _, v := range row {
			sum += v
		}
		mean := sum / float64(F)

		var sumSq float64
		for _, v := range row {
			d := v - mean
			sumSq += d * d
		}
		variance := sumSq / float64(F)
		invStd := 1.0 / math.Sqrt(variance+ln.epsilon)

		rowMean[i] = mean
		rowInvStd[i] = invStd

		for j := 0; j < F; j++ {
			normalized := (row[j] - mean) * invStd
			normData[rowStart+j] = normalized
			outData[rowStart+j] = normalized*gammaData[j] + betaData[j]
		}
	}

	out, err := tensor.NewTensor(shape, outData)
	if err != nil {
		return nil, err
	}

	if input.RequiresGrad || ln.Weight.RequiresGrad || ln.Bias.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{input, ln.Weight, ln.Bias}
		out.Operation = "layernorm"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			gradData := grad.GetData()

			dGamma := make([]float64, F)
			dBeta := make([]float64, F)

			// dGamma/dBeta reduce over the BATCH dimension (same as
			// BatchNorm), even though the normalization itself reduces
			// over FEATURES per row - gamma/beta are still one value per
			// feature, shared across every row.
			for j := 0; j < F; j++ {
				var sumDGamma, sumDBeta float64
				for i := 0; i < B; i++ {
					idx := i*F + j
					dy := gradData[idx]
					sumDBeta += dy
					sumDGamma += dy * normData[idx]
				}
				dGamma[j] = sumDGamma
				dBeta[j] = sumDBeta
			}

			if ln.Weight.RequiresGrad {
				if ln.Weight.Grad == nil {
					ln.Weight.ZeroGrad()
				}
				wGrad := ln.Weight.Grad.GetData()
				for j := range wGrad {
					wGrad[j] += dGamma[j]
				}
			}
			if ln.Bias.RequiresGrad {
				if ln.Bias.Grad == nil {
					ln.Bias.ZeroGrad()
				}
				bGrad := ln.Bias.Grad.GetData()
				for j := range bGrad {
					bGrad[j] += dBeta[j]
				}
			}

			if input.RequiresGrad {
				// Same "invStd/N * (N*dXhat - sum(dXhat) - xhat*sum(dXhat*xhat))"
				// formula as BatchNorm's input gradient, but the reduction
				// (the "N" and the sums) is over FEATURES within a row,
				// not over the BATCH within a column - this is the mirror image of BatchNorm along the other axis.
				gradInputData := make([]float64, B*F)
				Ff := float64(F)

				for i := 0; i < B; i++ {
					rowStart := i * F
					invStd := rowInvStd[i]

					var sumDXhat, sumDXhatXhat float64
					for j := 0; j < F; j++ {
						idx := rowStart + j
						dXhat := gradData[idx] * gammaData[j]
						sumDXhat += dXhat
						sumDXhatXhat += dXhat * normData[idx]
					}

					for j := 0; j < F; j++ {
						idx := rowStart + j
						dXhat := gradData[idx] * gammaData[j]
						xhat := normData[idx]
						gradInputData[idx] = invStd / Ff * (Ff*dXhat - sumDXhat - xhat*sumDXhatXhat)
					}
				}

				gradForInput, err := tensor.NewTensor(input.GetShape(), gradInputData)
				if err != nil {
					fmt.Printf("Warning: failed to create gradient tensor in LayerNorm backward: %v\n", err)
					return
				}
				input.Backward(gradForInput)
			}
		}
	}

	return out, nil
}

func (ln *LayerNorm) Parameters() []*tensor.Tensor { return []*tensor.Tensor{ln.Weight, ln.Bias} }


func (ln *LayerNorm) NamedParameters(prefix string) map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		prefix + ".weight": ln.Weight,
		prefix + ".bias":   ln.Bias,
	}
}
func (ln *LayerNorm) ZeroGrad()                    { ln.Weight.ZeroGrad(); ln.Bias.ZeroGrad() }
func (ln *LayerNorm) Name() string                 { return "LayerNorm" }
func (ln *LayerNorm) Train()                       {} // Layernorm is same in train/eval mode, but we still implement this to satisfy the Layer interface.
func (ln *LayerNorm) Eval()                        {}