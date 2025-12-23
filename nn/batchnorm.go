package nn

import (
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
	momentum     float32
	epsilon      float32
	training     bool
}

// creates a new BatchNorm1d layer.
func NewBatchNorm1d(numFeatures int, momentum, epsilon float32) (*BatchNorm1d, error) {
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

	var mean, variance *tensor.Tensor
	if bn.training {
		mean = tensor.Mean(input, 0, false)    // mean across batch dim
		variance = tensor.Var(input, 0, false) // variance across batch dim

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

		invStd := float32(1.0 / math.Sqrt(float64(v+bn.epsilon)))

		for i := 0; i < B; i++ {
			idx := i*F + j
			normalized := (inData[idx] - m) * invStd
			outData[idx] = normalized*gamma + beta
		}
	}

	if input.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{input, bn.Weight, bn.Bias}
		out.Operation = "batchnorm1d"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			zeroGrad, _ := tensor.NewTensor(input.GetShape(), nil)
			input.Backward(zeroGrad)
		}
	}
	return out, nil
}

func (bn *BatchNorm1d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{bn.Weight, bn.Bias} }
func (bn *BatchNorm1d) ZeroGrad()                    { bn.Weight.ZeroGrad(); bn.Bias.ZeroGrad() }
func (bn *BatchNorm1d) Name() string                 { return "BatchNorm1d" }
func (bn *BatchNorm1d) Train()                       { bn.training = true }
func (bn *BatchNorm1d) Eval()                        { bn.training = false }

// BatchNorm2d: For 4D Tensors [Batch, Channels, Height, Width] from Conv layers
type BatchNorm2d struct {
	Weight, Bias *tensor.Tensor
	RunningMean  *tensor.Tensor
	RunningVar   *tensor.Tensor
	momentum     float32
	epsilon      float32
	training     bool
	// caching in backward is disabled. try to add it.
}

func NewBatchNorm2d(numChannels int, momentum, epsilon float32) (*BatchNorm2d, error) {
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

	var mean, variance *tensor.Tensor

	if bn.training {
		meanData := make([]float32, C)
		varData := make([]float32, C)
		N := float32(B * H * W)

		for c := 0; c < C; c++ {
			var sum, sumSq float32
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

	var wg sync.WaitGroup

	numGoroutines := runtime.NumCPU()
	jobsPerGo := (B*C + numGoroutines - 1) / numGoroutines

	for i := 0; i < numGoroutines; i++ {
		startJob, endJob := i*jobsPerGo, (i+1)*jobsPerGo
		if endJob > B*C {
			endJob = B * C
		}
		wg.Add(1)

		go func(start, end int) {
			defer wg.Done()

			for job := start; job < end; job++ {
				b := job / C
				c := job % C
				m := meanData[c]
				v := varData[c]
				gamma := gammaData[c]
				beta := betaData[c]
				invStd := float32(1.0 / (math.Sqrt(float64(v + bn.epsilon))))
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						idx := b*(C*H*W) + c*(H*W) + h*W + w
						normalized := (inData[idx] - m) * invStd
						outData[idx] = normalized*gamma + beta
					}
				}
			}

		}(startJob, endJob)
	}

	wg.Wait()

	if input.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{input, bn.Weight, bn.Bias}
		out.Operation = "batchnorm2d"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			zeroGrad, _ := tensor.NewTensor(input.GetShape(), nil)
			input.Backward(zeroGrad)
		}
	}
	return out, nil
}

func (bn *BatchNorm2d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{bn.Weight, bn.Bias} }
func (bn *BatchNorm2d) ZeroGrad()                    { bn.Weight.ZeroGrad(); bn.Bias.ZeroGrad() }
func (bn *BatchNorm2d) Name() string                 { return "BatchNorm2d" }
func (bn *BatchNorm2d) Train()                       { bn.training = true }
func (bn *BatchNorm2d) Eval()                        { bn.training = false }
