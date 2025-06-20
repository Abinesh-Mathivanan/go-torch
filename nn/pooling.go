package nn

import (
	"fmt"
	"go-torch/tensor"
	"math"
	"runtime"
	"sync"
)

// MaxPooling2D implements a 2D max pooling layer.
type MaxPooling2D struct {
	KernelSize int
	Stride     int
}


// returns a new MaxPooling layer 
func NewMaxPooling2D(kernelSize, stride int) *MaxPooling2D {
	return &MaxPooling2D{KernelSize: kernelSize, Stride: stride}
}



func (p *MaxPooling2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	if len(inputShape) != 4 {
		return nil, fmt.Errorf("maxpool expects a 4D input, got %dD", len(inputShape))
	}
	b, c, h, w := inputShape[0], inputShape[1], inputShape[2], inputShape[3]

	outH := (h - p.KernelSize)/p.Stride + 1
	outW := (w - p.KernelSize)/p.Stride + 1
	outShape := []int{b, c, outH, outW}
	outData := make([]float64, b*c*outH*outW)
	maxIndices := make([]int, len(outData))
	inputData := input.GetData()

	numGoroutines := runtime.NumCPU()
	var wg sync.WaitGroup
	
	totalJobs := b * c
	jobsPerGo := (totalJobs + numGoroutines - 1) / numGoroutines

	for i := 0; i < numGoroutines; i++ {
		startJob := i * jobsPerGo
		endJob := (i + 1) * jobsPerGo
		if endJob > totalJobs {
			endJob = totalJobs
		}
		if startJob >= endJob {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for job := s; job < e; job++ {
				i := job / c
				j := job % c
				for k := 0; k < outH; k++ {
					for l := 0; l < outW; l++ {
						maxVal := -math.MaxFloat64
						maxIndex := -1
						hStart, wStart := k*p.Stride, l*p.Stride
						for y := 0; y < p.KernelSize; y++ {
							for x := 0; x < p.KernelSize; x++ {
								srcH, srcW := hStart+y, wStart+x
								srcIndex := i*(c*h*w) + j*(h*w) + srcH*w + srcW
								if inputData[srcIndex] > maxVal {
									maxVal = inputData[srcIndex]
									maxIndex = srcIndex
								}
							}
						}
						destIndex := i*(c*outH*outW) + j*(outH*outW) + k*outW + l
						outData[destIndex] = maxVal
						maxIndices[destIndex] = maxIndex
					}
				}
			}
		}(startJob, endJob)
	}
	wg.Wait()

	output, err := tensor.NewTensor(outShape, outData)
	if err != nil { return nil, err }

	if input.RequiresGrad {
		output.RequiresGrad = true
		output.Parents = []*tensor.Tensor{input}
		output.Operation = "maxpool"
		output.BackwardFunc = func(grad *tensor.Tensor) {
			if input.RequiresGrad {
				gradInputData := make([]float64, tensor.Numel(input))
				gradOutputData := grad.GetData()
				for i, g := range gradOutputData {
					idx := maxIndices[i]
					gradInputData[idx] += g
				}
				gradForInput, _ := tensor.NewTensor(inputShape, gradInputData)
				input.Backward(gradForInput)
			}
		}
	}
	return output, nil
}


// we don't need to zero out any grad, since MaxPooling doesn't need any learnable parameters 
func (p *MaxPooling2D) Parameters() []*tensor.Tensor { return []*tensor.Tensor{} }
func (p *MaxPooling2D) ZeroGrad() {}