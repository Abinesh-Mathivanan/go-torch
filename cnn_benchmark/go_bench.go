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

func loadImages(filepath string) (*tensor.Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var reader io.Reader = file
	var magic, numImages, numRows, numCols int32
	binary.Read(reader, binary.BigEndian, &magic)
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number for images file: %d", magic)
	}
	binary.Read(reader, binary.BigEndian, &numImages)
	binary.Read(reader, binary.BigEndian, &numRows)
	binary.Read(reader, binary.BigEndian, &numCols)
	imageData := make([]byte, numImages*numRows*numCols)
	_, err = io.ReadFull(reader, imageData)
	if err != nil {
		return nil, err
	}
	floatData := make([]float64, len(imageData))
	for i, v := range imageData {
		floatData[i] = float64(v) / 255.0
	}
	shape := []int{int(numImages), 1, int(numRows), int(numCols)}
	return tensor.NewTensor(shape, floatData)
}

func loadLabels(filepath string) ([]int, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var reader io.Reader = file
	var magic, numLabels int32
	binary.Read(reader, binary.BigEndian, &magic)
	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number for labels file: %d", magic)
	}
	binary.Read(reader, binary.BigEndian, &numLabels)
	labelData := make([]byte, numLabels)
	_, err = io.ReadFull(reader, labelData)
	if err != nil {
		return nil, err
	}
	intLabels := make([]int, len(labelData))
	for i, v := range labelData {
		intLabels[i] = int(v)
	}
	return intLabels, nil
}


func main() {
	rand.Seed(time.Now().UnixNano())

	learningRate := 0.01
	batchSize := 32
	epochs := 5
	numClasses := 10

	// PHASE 1: PRE-TUI SETUP AND CONSOLE OUTPUT
	fmt.Println("--- Go-Torch MNIST Trainer ---")
	fmt.Println("Initializing model...")

	model := nn.NewSequential()
	conv1, err := nn.NewConv2D(1, 16, 5, 1, 2)
	if err != nil { panic(err) }
	model.Add(conv1)
	model.Add(nn.NewRELU())
	model.Add(nn.NewMaxPooling2D(2, 2))
	conv2, err := nn.NewConv2D(16, 32, 5, 1, 2)
	if err != nil { panic(err) }
	model.Add(conv2)
	model.Add(nn.NewRELU())
	model.Add(nn.NewMaxPooling2D(2, 2))
	model.Add(nn.NewFlatten())
	linear1, err := nn.NewLinear(32*7*7, 128)
	if err != nil { panic(err) }
	model.Add(linear1)
	model.Add(nn.NewRELU())
	linear2, err := nn.NewLinear(128, numClasses)
	if err != nil { panic(err) }
	model.Add(linear2)

	inspector := utility.NewModelInspector(model)
	inspector.Summary()

	fmt.Println("Loading dataset (this may take a moment)...")
	trainImages, err := loadImages(fmt.Sprintf("%s/train-images-idx3-ubyte/train-images-idx3-ubyte", mnistDir))
	if err != nil { panic(err) }
	trainLabels, err := loadLabels(fmt.Sprintf("%s/train-labels-idx1-ubyte/train-labels-idx1-ubyte", mnistDir))
	if err != nil { panic(err) }
	testImages, err := loadImages(fmt.Sprintf("%s/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", mnistDir))
	if err != nil { panic(err) }
	testLabels, err := loadLabels(fmt.Sprintf("%s/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", mnistDir))
	if err != nil { panic(err) }
	fmt.Println("Dataset loaded. Initializing TUI...")
	time.Sleep(2 * time.Second)



	// PHASE 2: TUI-ONLY MODE
	dashboard := utility.NewTrainingDashboard(learningRate, batchSize, epochs)
	defer dashboard.Close()

	go func() {
		optimizer, _ := optimizer.NewSGD(model.Parameters(), learningRate)

		batchImagesDataBuffer := make([]float64, batchSize*28*28)
		batchLabelsBuffer := make([]int, batchSize)

		numTrainSamples := trainImages.GetShape()[0]
		indices := make([]int, numTrainSamples)
		for i := range indices {
			indices[i] = i
		}
		numBatches := (numTrainSamples + batchSize - 1) / batchSize
		
		totalStartTime := time.Now()

		for epoch := 1; epoch <= epochs; epoch++ {
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
				optimizer.Step()

				currentLoss := loss.GetData()[0]
				runningLoss += currentLoss
				dashboard.AddLoss(currentLoss)
				dashboard.UpdateStats(epoch, epochs, i, numBatches, runningLoss/float64(i), epochStartTime, totalStartTime)
			}

			dashboard.Log(fmt.Sprintf("Epoch %d complete. Running evaluation...", epoch))
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
			dashboard.AddAccuracy(accuracy)
		}
		dashboard.Log("Training complete! Press 'q' or <C-c> to exit.")
	}()

	dashboard.Loop()
}