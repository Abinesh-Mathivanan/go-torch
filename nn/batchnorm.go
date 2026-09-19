package nn

import (
	"fmt"
	"go-torch/tensor"
	"math"
	"runtime"
	"sync"
)

// BatchNorm1d: For 2D Tensors [Batch, Features] from Linear layers
// BatchNorm1d applies Batch Normalization over a 2D input.
type BatchNorm1d struct {
	Weight, Bias *tensor.Tensor
	RunningMean  *tensor.Tensor
	RunningVar   *tensor.Tensor
	momentum     float64
	epsilon      float64
	training     bool
}

// creates a new BatchNorm1d layer.
func NewBatchNorm1d(numFeatures int, momentum, epsilon float64) (*BatchNorm1d, error) {
	weight, err := tensor.NewTensor([]int{numFeatures}, nil)
	if err != nil {
		return nil, err
	}
	for i := range weight.GetData() {
		weight.GetData()[i] = 1.0
	}
	weight.RequiresGrad = true

	bias, err := tensor.NewTensor([]int{numFeatures}, nil)
	if err != nil {
		return nil, err
	}
	bias.RequiresGrad = true

	runningMean, err := tensor.NewTensor([]int{numFeatures}, nil)
	if err != nil {
		return nil, err
	}
	runningVar, err := tensor.NewTensor([]int{numFeatures}, nil)
	if err != nil {
		return nil, err
	}
	for i := range runningVar.GetData() {
		runningVar.GetData()[i] = 1.0
	}

	return &BatchNorm1d{
		Weight: weight, Bias: bias, RunningMean: runningMean, RunningVar: runningVar,
		momentum: momentum, epsilon: epsilon, training: true,
	}, nil
}

func (bn *BatchNorm1d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.GetShape()
	B, F := shape[0], shape[1]

	out, _ := tensor.NewTensor(shape, nil)
	outData := out.GetData()
	inData := input.GetData()
	// cache the normalized (x-hat) values from the forward pass; the backward
	// pass needs them to compute dGamma and the input gradient.
	normData := make([]float64, B*F)

	var mean, variance *tensor.Tensor
	if bn.training {
		var err error
		mean, err = tensor.Mean(input, 0, false) // mean across batch dim
		if err != nil {
			return nil, fmt.Errorf("batchnorm1d: failed computing mean: %w", err)
		}
		variance, err = tensor.Var(input, 0, false) // variance across batch dim
		if err != nil {
			return nil, fmt.Errorf("batchnorm1d: failed computing variance: %w", err)
		}

		rmData, rvData := bn.RunningMean.GetData(), bn.RunningVar.GetData()
		meanData, varData := mean.GetData(), variance.GetData()
		for i := range rmData {
			rmData[i] = (1-bn.momentum)*rmData[i] + bn.momentum*meanData[i]
			rvData[i] = (1-bn.momentum)*rvData[i] + bn.momentum*varData[i]
		}
	} else {
		mean, variance = bn.RunningMean, bn.RunningVar
	}

	meanData := mean.GetData()
	varData := variance.GetData()
	gammaData := bn.Weight.GetData()
	betaData := bn.Bias.GetData()

	for j := 0; j < F; j++ {
		m := meanData[j]
		v := varData[j]
		gamma := gammaData[j]
		beta := betaData[j]

		invStd := 1.0 / math.Sqrt(v+bn.epsilon)

		for i := 0; i < B; i++ {
			idx := i*F + j
			normalized := (inData[idx] - m) * invStd
			normData[idx] = normalized
			outData[idx] = normalized*gamma + beta
		}
	}

	if input.RequiresGrad || bn.Weight.RequiresGrad || bn.Bias.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{input, bn.Weight, bn.Bias}
		out.Operation = "batchnorm1d"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			gradData := grad.GetData()

			dGamma := make([]float64, F)
			dBeta := make([]float64, F)
			dXhat := make([]float64, B*F)

			for j := 0; j < F; j++ {
				var sumDGamma, sumDBeta float64
				for i := 0; i < B; i++ {
					idx := i*F + j
					dy := gradData[idx]
					sumDBeta += dy
					sumDGamma += dy * normData[idx]
					dXhat[idx] = dy * gammaData[j]
				}
				dGamma[j] = sumDGamma
				dBeta[j] = sumDBeta
			}

			if bn.Weight.RequiresGrad {
				if bn.Weight.Grad == nil {
					bn.Weight.ZeroGrad()
				}
				wGrad := bn.Weight.Grad.GetData()
				for j := range wGrad {
					wGrad[j] += dGamma[j]
				}
			}
			if bn.Bias.RequiresGrad {
				if bn.Bias.Grad == nil {
					bn.Bias.ZeroGrad()
				}
				bGrad := bn.Bias.Grad.GetData()
				for j := range bGrad {
					bGrad[j] += dBeta[j]
				}
			}

			if input.RequiresGrad {
				// standard batchnorm input-gradient formula:
				// dx = invStd/N * (N*dXhat - sum(dXhat) - xhat*sum(dXhat*xhat))
				gradInputData := make([]float64, B*F)
				Nf := float64(B)
				for j := 0; j < F; j++ {
					v := varData[j]
					invStd := 1.0 / math.Sqrt(v+bn.epsilon)
					var sumDXhat, sumDXhatXmu float64
					for i := 0; i < B; i++ {
						idx := i*F + j
						xmu := inData[idx] - meanData[j]
						sumDXhat += dXhat[idx]
						sumDXhatXmu += dXhat[idx] * xmu
					}
					for i := 0; i < B; i++ {
						idx := i*F + j
						xmu := inData[idx] - meanData[j]
						gradInputData[idx] = invStd / Nf * (Nf*dXhat[idx] - sumDXhat - xmu*invStd*invStd*sumDXhatXmu)
					}
				}
				gradForInput, err := tensor.NewTensor(input.GetShape(), gradInputData)
				if err != nil {
					fmt.Printf("Warning: failed to create gradient tensor in BatchNorm1d backward: %v\n", err)
					return
				}
				input.Backward(gradForInput)
			}
		}
	}
	return out, nil
}

func (bn *BatchNorm1d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{bn.Weight, bn.Bias} }

// NamedParameters returns this layer's state keyed by name, prefixed (e.g.
// prefix="bn1" -> "bn1.weight", "bn1.bias", "bn1.running_mean","bn1.running_var"). 
// This is useful for saving and loading model state, since the running statistics are part of the model's state even though they are not learnable parameters.
func (bn *BatchNorm1d) NamedParameters(prefix string) map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		prefix + ".weight":       bn.Weight,
		prefix + ".bias":         bn.Bias,
		prefix + ".running_mean": bn.RunningMean,
		prefix + ".running_var":  bn.RunningVar,
	}
}
func (bn *BatchNorm1d) ZeroGrad()                    { bn.Weight.ZeroGrad(); bn.Bias.ZeroGrad() }
func (bn *BatchNorm1d) Name() string                 { return "BatchNorm1d" }
func (bn *BatchNorm1d) Train()                       { bn.training = true }
func (bn *BatchNorm1d) Eval()                        { bn.training = false }

// BatchNorm2d: For 4D Tensors [Batch, Channels, Height, Width] from Conv layers
type BatchNorm2d struct {
	Weight, Bias *tensor.Tensor
	RunningMean  *tensor.Tensor
	RunningVar   *tensor.Tensor
	momentum     float64
	epsilon      float64
	training     bool
}

func NewBatchNorm2d(numChannels int, momentum, epsilon float64) (*BatchNorm2d, error) {
	weight, err := tensor.NewTensor([]int{numChannels}, nil)
	if err != nil {
		return nil, err
	}

	for i := range weight.GetData() {
		weight.GetData()[i] = 1.0
	}

	weight.RequiresGrad = true
	bias, err := tensor.NewTensor([]int{numChannels}, nil)
	if err != nil {
		return nil, err
	}
	bias.RequiresGrad = true
	runningMean, err := tensor.NewTensor([]int{numChannels}, nil)
	if err != nil {
		return nil, err
	}
	runningVar, err := tensor.NewTensor([]int{numChannels}, nil)
	if err != nil {
		return nil, err
	}

	for i := range runningVar.GetData() {
		runningVar.GetData()[i] = 1.0
	}
	return &BatchNorm2d{
		Weight: weight, Bias: bias, RunningMean: runningMean, RunningVar: runningVar,
		momentum: momentum, epsilon: epsilon, training: true,
	}, nil
}

func (bn *BatchNorm2d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.GetShape()
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	out, _ := tensor.NewTensor(shape, nil)
	outData := out.GetData()
	inData := input.GetData()
	// cache the normalized (x-hat) values from the forward pass for use in backward.
	normData := make([]float64, len(inData))

	var mean, variance *tensor.Tensor

	if bn.training {
		meanData := make([]float64, C)
		varData := make([]float64, C)
		N := float64(B * H * W)

		for c := 0; c < C; c++ {
			var sum, sumSq float64
			for b := 0; b < B; b++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						idx := b*(C*H*W) + c*(H*W) + h*W + w
						val := inData[idx]
						sum += val
						sumSq += val * val
					}
				}
			}
			channelMean := sum / N
			channelVar := (sumSq / N) - (channelMean * channelMean)
			meanData[c] = channelMean
			varData[c] = channelVar
		}

		mean, _ = tensor.NewTensor([]int{C}, meanData)
		variance, _ = tensor.NewTensor([]int{C}, varData)
		rmData, rvData := bn.RunningMean.GetData(), bn.RunningVar.GetData()

		for i := range rmData {
			rmData[i] = (1-bn.momentum)*rmData[i] + bn.momentum*meanData[i]
			rvData[i] = (1-bn.momentum)*rvData[i] + bn.momentum*varData[i]
		}

	} else {

		mean = bn.RunningMean
		variance = bn.RunningVar

	}

	meanData := mean.GetData()
	varData := variance.GetData()
	gammaData := bn.Weight.GetData()
	betaData := bn.Bias.GetData()

	fillJob := func(job int) {
		b := job / C
		c := job % C
		m := meanData[c]
		v := varData[c]
		gamma := gammaData[c]
		beta := betaData[c]
		invStd := 1.0 / (math.Sqrt(v + bn.epsilon))
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				idx := b*(C*H*W) + c*(H*W) + h*W + w
				normalized := (inData[idx] - m) * invStd
				normData[idx] = normalized
				outData[idx] = normalized*gamma + beta
			}
		}
	}

	if !tensor.ShouldParallelize(len(inData)) {
		for job := 0; job < B*C; job++ {
			fillJob(job)
		}
	} else {
		var wg sync.WaitGroup
		numGoroutines := runtime.NumCPU()
		jobsPerGo := (B*C + numGoroutines - 1) / numGoroutines

		for i := 0; i < numGoroutines; i++ {
			startJob, endJob := i*jobsPerGo, (i+1)*jobsPerGo
			if endJob > B*C {
				endJob = B * C
			}
			if startJob >= endJob {
				continue
			}
			wg.Add(1)

			go func(start, end int) {
				defer wg.Done()
				for job := start; job < end; job++ {
					fillJob(job)
				}
			}(startJob, endJob)
		}
		wg.Wait()
	}

	if input.RequiresGrad || bn.Weight.RequiresGrad || bn.Bias.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{input, bn.Weight, bn.Bias}
		out.Operation = "batchnorm2d"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			gradData := grad.GetData()
			N := float64(B * H * W)

			dGamma := make([]float64, C)
			dBeta := make([]float64, C)
			dXhat := make([]float64, len(inData))

			for c := 0; c < C; c++ {
				var sumDGamma, sumDBeta float64
				for b := 0; b < B; b++ {
					for h := 0; h < H; h++ {
						for w := 0; w < W; w++ {
							idx := b*(C*H*W) + c*(H*W) + h*W + w
							dy := gradData[idx]
							sumDBeta += dy
							sumDGamma += dy * normData[idx]
							dXhat[idx] = dy * gammaData[c]
						}
					}
				}
				dGamma[c] = sumDGamma
				dBeta[c] = sumDBeta
			}

			if bn.Weight.RequiresGrad {
				if bn.Weight.Grad == nil {
					bn.Weight.ZeroGrad()
				}
				wGrad := bn.Weight.Grad.GetData()
				for c := range wGrad {
					wGrad[c] += dGamma[c]
				}
			}
			if bn.Bias.RequiresGrad {
				if bn.Bias.Grad == nil {
					bn.Bias.ZeroGrad()
				}
				bGrad := bn.Bias.Grad.GetData()
				for c := range bGrad {
					bGrad[c] += dBeta[c]
				}
			}

			if input.RequiresGrad {
				gradInputData := make([]float64, len(inData))
				for c := 0; c < C; c++ {
					v := varData[c]
					invStd := 1.0 / math.Sqrt(v+bn.epsilon)
					var sumDXhat, sumDXhatXmu float64
					for b := 0; b < B; b++ {
						for h := 0; h < H; h++ {
							for w := 0; w < W; w++ {
								idx := b*(C*H*W) + c*(H*W) + h*W + w
								xmu := inData[idx] - meanData[c]
								sumDXhat += dXhat[idx]
								sumDXhatXmu += dXhat[idx] * xmu
							}
						}
					}
					for b := 0; b < B; b++ {
						for h := 0; h < H; h++ {
							for w := 0; w < W; w++ {
								idx := b*(C*H*W) + c*(H*W) + h*W + w
								xmu := inData[idx] - meanData[c]
								gradInputData[idx] = invStd / N * (N*dXhat[idx] - sumDXhat - xmu*invStd*invStd*sumDXhatXmu)
							}
						}
					}
				}
				gradForInput, err := tensor.NewTensor(input.GetShape(), gradInputData)
				if err != nil {
					fmt.Printf("Warning: failed to create gradient tensor in BatchNorm2d backward: %v\n", err)
					return
				}
				input.Backward(gradForInput)
			}
		}
	}
	return out, nil
}

func (bn *BatchNorm2d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{bn.Weight, bn.Bias} }

// NamedParameters returns this layer's state keyed by name, including
// running statistics - see the comment on BatchNorm1d.NamedParameters.
func (bn *BatchNorm2d) NamedParameters(prefix string) map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		prefix + ".weight":       bn.Weight,
		prefix + ".bias":         bn.Bias,
		prefix + ".running_mean": bn.RunningMean,
		prefix + ".running_var":  bn.RunningVar,
	}
}
func (bn *BatchNorm2d) ZeroGrad()                    { bn.Weight.ZeroGrad(); bn.Bias.ZeroGrad() }
func (bn *BatchNorm2d) Name() string                 { return "BatchNorm2d" }
func (bn *BatchNorm2d) Train()                       { bn.training = true }
func (bn *BatchNorm2d) Eval()                        { bn.training = false }