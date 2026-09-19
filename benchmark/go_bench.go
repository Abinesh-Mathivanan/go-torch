package main

import (
	"encoding/binary"
	"fmt"
	"go-torch/autograd"
	"go-torch/nn"
	"go-torch/optimizer"
	"go-torch/tensor"
	"go-torch/utility"
	"io"
	"math/rand"
	"os"
	"time"
)


// data loader
const mnistDir = "mnist_data"

const (
	maxMNISTItems = 10_000_000
	maxMNISTDim   = 1 << 16
)

func loadImages(filepath string) (*tensor.Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var reader io.Reader = file
	var magic, numImages, numRows, numCols int32
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number from %s: %w", filepath, err)
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number for images file: %d", magic)
	}
	if err := binary.Read(reader, binary.BigEndian, &numImages); err != nil {
		return nil, fmt.Errorf("failed to read image count from %s: %w", filepath, err)
	}
	if err := binary.Read(reader, binary.BigEndian, &numRows); err != nil {
		return nil, fmt.Errorf("failed to read row count from %s: %w", filepath, err)
	}
	if err := binary.Read(reader, binary.BigEndian, &numCols); err != nil {
		return nil, fmt.Errorf("failed to read col count from %s: %w", filepath, err)
	}

	// validate header values before trusting them for an allocation size 
	// (prevents maliciously crafted files from causing OOM or other issues)
	if numImages <= 0 || numImages > maxMNISTItems {
		return nil, fmt.Errorf("invalid image count in header: %d", numImages)
	}
	if numRows <= 0 || numRows > maxMNISTDim || numCols <= 0 || numCols > maxMNISTDim {
		return nil, fmt.Errorf("invalid image dimensions in header: %dx%d", numRows, numCols)
	}

	totalBytes := int64(numImages) * int64(numRows) * int64(numCols)
	imageData := make([]byte, totalBytes)
	_, err = io.ReadFull(reader, imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to read image data from %s (expected %d bytes): %w", filepath, totalBytes, err)
	}
	floatData := make([]float64, len(imageData))
	for i, v := range imageData {
		floatData[i] = float64(v) / 255.0
	}
	shape := []int{int(numImages), 1, int(numRows), int(numCols)}
	return tensor.NewTensor(shape, floatData)
}


// load train and test labels 
func loadLabels(filepath string) ([]int, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var reader io.Reader = file
	var magic, numLabels int32
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number from %s: %w", filepath, err)
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number for labels file: %d", magic)
	}
	if err := binary.Read(reader, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read label count from %s: %w", filepath, err)
	}
	if numLabels <= 0 || numLabels > maxMNISTItems {
		return nil, fmt.Errorf("invalid label count in header: %d", numLabels)
	}
	labelData := make([]byte, numLabels)
	_, err = io.ReadFull(reader, labelData)
	if err != nil {
		return nil, fmt.Errorf("failed to read label data from %s: %w", filepath, err)
	}
	intLabels := make([]int, len(labelData))
	for i, v := range labelData {
		intLabels[i] = int(v)
	}
	return intLabels, nil
}



// main 
func main() {
	rand.Seed(time.Now().UnixNano())

	learningRate := 0.001
	batchSize := 32
	epochs := 5
	numClasses := 10


	// removed the TUI - so just plain logging
	fmt.Println("--- Go-Torch MNIST Trainer ---")
	fmt.Println("Initializing model with BatchNorm and Dropout...")


	model := nn.NewSequential()
	var err error

	conv1, err := nn.NewConv2D(1, 16, 5, 1, 2)
	if err != nil { panic(err) }
	model.Add(conv1)

	model.Add(nn.NewRELU())
	model.Add(nn.NewMaxPooling2D(2, 2))

	conv2, err := nn.NewConv2D(16, 32, 5, 1, 2)
	if err != nil { panic(err) }
	model.Add(conv2)

	bn2, err := nn.NewBatchNorm2d(32, 0.9, 1e-5)
	if err != nil { panic(err) }
	model.Add(bn2)

	model.Add(nn.NewRELU())
	model.Add(nn.NewMaxPooling2D(2, 2))
	model.Add(nn.NewFlatten())

	linear1, err := nn.NewLinear(32*7*7, 128)
	if err != nil { panic(err) }
	model.Add(linear1)

	bn3, err := nn.NewBatchNorm1d(128, 0.9, 1e-5)
	if err != nil { panic(err) }
	model.Add(bn3)

	model.Add(nn.NewRELU())
	model.Add(nn.NewDropout(0.5))

	linear2, err := nn.NewLinear(128, numClasses)
	if err != nil { panic(err) }
	model.Add(linear2)


	// model summary
	inspector := utility.NewModelInspector(model)
	inspector.Summary()

	fmt.Println("Loading dataset...")
	trainImages, err := loadImages(fmt.Sprintf("%s/train-images-idx3-ubyte/train-images-idx3-ubyte", mnistDir)); if err != nil { panic(err) }
	trainLabels, err := loadLabels(fmt.Sprintf("%s/train-labels-idx1-ubyte/train-labels-idx1-ubyte", mnistDir)); if err != nil { panic(err) }
	testImages, err := loadImages(fmt.Sprintf("%s/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", mnistDir)); if err != nil { panic(err) }
	testLabels, err := loadLabels(fmt.Sprintf("%s/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", mnistDir)); if err != nil { panic(err) }
	fmt.Println("Dataset loaded. Starting training...")

	// training 
	logger := utility.NewTrainingLogger(learningRate, batchSize, epochs)
	defer logger.Close()

	adamOptimizer, _ := optimizer.NewAdam(model.Parameters(), learningRate, 0.9, 0.999, 1e-8)

	batchImagesDataBuffer := make([]float64, batchSize*28*28)
	batchLabelsBuffer := make([]int, batchSize)

	numTrainSamples := trainImages.GetShape()[0]
	indices := make([]int, numTrainSamples)
	for i := range indices {
		indices[i] = i
	}
	numBatches := (numTrainSamples + batchSize - 1) / batchSize

	totalStartTime := time.Now()
	logger.Log("Starting training run with Adam optimizer...")

	for epoch := 1; epoch <= epochs; epoch++ {
		model.Train()
		runningLoss := 0.0
		epochStartTime := time.Now()
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		for i := 1; i <= numBatches; i++ {
			start := (i - 1) * batchSize
			end := start + batchSize
			if end > numTrainSamples { end = numTrainSamples }
			batchIndices := indices[start:end]
			currentBatchSize := len(batchIndices)
			if currentBatchSize == 0 { continue }

			currentBatchImageData := batchImagesDataBuffer[:currentBatchSize*28*28]
			batchLabels := batchLabelsBuffer[:currentBatchSize]
			allImageData := trainImages.GetData()
			for j, idx := range batchIndices {
				imgStart := idx * 28 * 28
				copy(currentBatchImageData[j*28*28:(j+1)*28*28], allImageData[imgStart:imgStart+28*28])
				batchLabels[j] = trainLabels[idx]
			}
			batchTensor, _ := tensor.NewTensor([]int{currentBatchSize, 1, 28, 28}, currentBatchImageData)

			model.ZeroGrad()
			logits, _ := model.Forward(batchTensor)
			loss, _ := nn.CrossEntropyLoss(logits, batchLabels)
			autograd.Backward(loss)
			adamOptimizer.Step()

			currentLoss := loss.GetData()[0]
			runningLoss += currentLoss
			logger.AddLoss(currentLoss)
			logger.UpdateStats(epoch, epochs, i, numBatches, runningLoss/float64(i), epochStartTime, totalStartTime)
		}

		model.Eval()
		logger.Log(fmt.Sprintf("Epoch %d complete. Running evaluation...", epoch))
		correct := 0
		numTestSamples := testImages.GetShape()[0]
		for i := 0; i < numTestSamples; i++ {
			imgTensor, _ := testImages.Slice(i)
			logits, _ := model.Forward(imgTensor)
			prediction := tensor.ArgMax(logits)
			if prediction == testLabels[i] {
				correct++
			}
		}
		accuracy := (float64(correct) / float64(numTestSamples)) * 100.0
		logger.AddAccuracy(accuracy)
	}
	logger.Log("Training complete!")
}