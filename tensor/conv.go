package tensor

import (
	"fmt"
	"runtime"
	"sync"
)



// Im2Col converts image-like data into a column matrix.
// i used core-parallelization over the batch dimension for performance.
func Im2Col(input *Tensor, kernelHeight, kernelWidth, stride, padding int) (*Tensor, error) {
	if len(input.GetShape()) != 4 {
		return nil, fmt.Errorf("im2col expects a 4D input tensor, but got %dD", len(input.GetShape()))
	}
	shape := input.GetShape()
	batchSize, channels, height, width := shape[0], shape[1], shape[2], shape[3]

	outHeight := (height + 2*padding - kernelHeight)/stride + 1
	outWidth := (width + 2*padding - kernelWidth)/stride + 1
	if outHeight <= 0 || outWidth <= 0 {
		return nil, fmt.Errorf("convolution produces invalid output size: %dx%d", outHeight, outWidth)
	}

	kernelSize := channels * kernelHeight * kernelWidth
	outputCols := outHeight * outWidth
	colMatrixShape := []int{kernelSize, batchSize * outputCols}
	colMatrixData := make([]float64, colMatrixShape[0]*colMatrixShape[1])

	inputData := input.GetData()
	numGoroutines := runtime.NumCPU()
	var wg sync.WaitGroup

	// Parallelize over the batch dimension
	batchesPerGo := (batchSize + numGoroutines - 1) / numGoroutines
	for i := 0; i < numGoroutines; i++ {
		startBatch := i * batchesPerGo
		endBatch := (i + 1) * batchesPerGo
		if endBatch > batchSize {
			endBatch = batchSize
		}

		if startBatch >= endBatch {
			continue
		}

		wg.Add(1)
		go func(sB, eB int) {
			defer wg.Done()
			for b := sB; b < eB; b++ {
				for c := 0; c < channels; c++ {
					for kh := 0; kh < kernelHeight; kh++ {
						for kw := 0; kw < kernelWidth; kw++ {
							inputRowStart := kh - padding
							inputColStart := kw - padding
							for oh := 0; oh < outHeight; oh++ {
								for ow := 0; ow < outWidth; ow++ {
									inputRow := inputRowStart + oh*stride
									inputCol := inputColStart + ow*stride

									colRow := c*(kernelHeight*kernelWidth) + kh*kernelWidth + kw
									colCol := b*outputCols + oh*outWidth + ow
									destIndex := colRow*colMatrixShape[1] + colCol

									if inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width {
										srcIndex := b*(channels*height*width) + c*(height*width) + inputRow*width + inputCol
										colMatrixData[destIndex] = inputData[srcIndex]
									}
								}
							}
						}
					}
				}
			}
		}(startBatch, endBatch)
	}
	wg.Wait()

	colMatrix, err := NewTensor(colMatrixShape, colMatrixData)
	if err != nil {
		return nil, fmt.Errorf("im2col failed to create output tensor: %w", err)
	}
	return colMatrix, nil
}



// Col2Im converts a column matrix back into image-like data.
// It is used in the backward pass of a convolution.
func Col2Im(cols *Tensor, inputShape []int, kernelHeight, kernelWidth, stride, padding int) (*Tensor, error) {
	if len(inputShape) != 4 {
		return nil, fmt.Errorf("col2im requires a 4D target inputShape, but got %dD", len(inputShape))
	}
	batchSize, channels, height, width := inputShape[0], inputShape[1], inputShape[2], inputShape[3]

	outHeight := (height + 2*padding - kernelHeight)/stride + 1
	outWidth := (width + 2*padding - kernelWidth)/stride + 1

	imgData := make([]float64, batchSize*channels*height*width)
	colsData := cols.GetData()
	colsShape := cols.GetShape()

	numGoroutines := runtime.NumCPU()
	var wg sync.WaitGroup
	
	totalJobs := batchSize * channels
	jobsPerGo := (totalJobs + numGoroutines - 1) / numGoroutines

	for i := 0; i < numGoroutines; i++ {
		startJob := i * jobsPerGo
		endJob := startJob + jobsPerGo
		if endJob > totalJobs {
			endJob = totalJobs
		}

		if startJob >= endJob {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for job := start; job < end; job++ {
				b := job / channels
				c := job % channels
				// The inner loops are now executed within a safe parallel context,
				// as each goroutine works on a different batch/channel combo.
				for kh := 0; kh < kernelHeight; kh++ {
					for kw := 0; kw < kernelWidth; kw++ {
						inputRowStart := kh - padding
						inputColStart := kw - padding
						for oh := 0; oh < outHeight; oh++ {
							for ow := 0; ow < outWidth; ow++ {
								inputRow := inputRowStart + oh*stride
								inputCol := inputColStart + ow*stride
								if inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width {
									colRow := c*(kernelHeight*kernelWidth) + kh*kernelWidth + kw
									colCol := b*outHeight*outWidth + oh*outWidth + ow
									srcIndex := colRow*colsShape[1] + colCol
									destIndex := b*(channels*height*width) + c*(height*width) + inputRow*width + inputCol
									imgData[destIndex] += colsData[srcIndex]
								}
							}
						}
					}
				}
			}
		}(startJob, endJob)
	}
	wg.Wait()

	imgTensor, err := NewTensor(inputShape, imgData)
	if err != nil {
		return nil, fmt.Errorf("col2im failed to create output tensor: %w", err)
	}
	return imgTensor, nil
}