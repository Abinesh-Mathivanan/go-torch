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
	"math"
	"math/rand"
	"os"
	"time"
)

const mnistDir = "mnist_data"


// data loader 
func loadImages(filepath string) (*tensor.Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil { return nil, err }
	defer file.Close()

	var magic, count, rows, cols int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &count)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	if magic != 2051 { return nil, fmt.Errorf("wrong magic: %d", magic) }

	pixelsPerImg := int(rows * cols)
	totalPixels := int(count) * pixelsPerImg
	data := make([]byte, totalPixels)
	if _, err := io.ReadFull(file, data); err != nil { return nil, err }

	floatData := make([]float32, totalPixels)
	for i, v := range data {
		floatData[i] = float32(v) / 255.0
	}
	return tensor.NewTensor([]int{int(count), 1, int(rows), int(cols)}, floatData)
}

func loadLabels(filepath string) ([]int, error) {
	file, err := os.Open(filepath)
	if err != nil { return nil, err }
	defer file.Close()

	var magic, count int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &count)

	if magic != 2049 { return nil, fmt.Errorf("wrong magic: %d", magic) }

	data := make([]byte, count)
	if _, err := io.ReadFull(file, data); err != nil { return nil, err }

	labels := make([]int, count)
	for i, v := range data {
		labels[i] = int(v)
	}
	return labels, nil
}


// use xavier initialization to prevent loss plateau 
func initializeModel(m *nn.Sequential) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for _, p := range m.Parameters() {
		shape := p.GetShape()
		data := p.GetData()
		
		fanIn := float32(1.0)
		if len(shape) > 1 {
			fanIn = float32(shape[0])
		}
		std := float32(math.Sqrt(2.0 / float64(fanIn)))
		
		for i := range data {
			data[i] = (r.Float32()*2 - 1) * std
		}
	}
}


func main() {
	rand.Seed(time.Now().UnixNano())

	learningRate := float32(0.001)
	batchSize := 32
	epochs := 5

	trainImg, err := loadImages(mnistDir + "/train-images-idx3-ubyte/train-images-idx3-ubyte")
	if err != nil { panic(err) }
	trainLbl, err := loadLabels(mnistDir + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
	if err != nil { panic(err) }
	testImg, _ := loadImages(mnistDir + "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
	testLbl, _ := loadLabels(mnistDir + "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

	model := nn.NewSequential()
	c1, _ := nn.NewConv2D(1, 16, 5, 1, 2); model.Add(c1); model.Add(nn.NewRELU()); model.Add(nn.NewMaxPooling2D(2, 2))
	c2, _ := nn.NewConv2D(16, 32, 5, 1, 2); model.Add(c2); model.Add(nn.NewRELU()); model.Add(nn.NewMaxPooling2D(2, 2))
	model.Add(nn.NewFlatten())
	l1, _ := nn.NewLinear(32*7*7, 128); model.Add(l1); model.Add(nn.NewRELU())
	l2, _ := nn.NewLinear(128, 10); model.Add(l2)

	initializeModel(model)
	opt, _ := optimizer.NewAdam(model.Parameters(), learningRate, 0.9, 0.999, 1e-8)

	dashboard := utility.NewTrainingDashboard(learningRate, batchSize, epochs)

	go func() {
		numSamples := trainImg.GetShape()[0]
		numBatches := numSamples / batchSize
		totalStart := time.Now()

		dashboard.Log("System: MNIST Loaded. Starting Autograd Engine...")

		for epoch := 1; epoch <= epochs; epoch++ {
			epochStart := time.Now()
			indices := rand.Perm(numSamples)
			var runningLoss float32

			model.Train()
			for b := 0; b < numBatches; b++ {
				// Prepare Batch
				batchData := make([]float32, batchSize*784)
				batchTargets := make([]int, batchSize)
				for i := 0; i < batchSize; i++ {
					idx := indices[b*batchSize+i]
					copy(batchData[i*784:(i+1)*784], trainImg.GetData()[idx*784:(idx+1)*784])
					batchTargets[i] = trainLbl[idx]
				}

				x, _ := tensor.NewTensor([]int{batchSize, 1, 28, 28}, batchData)
				
				model.ZeroGrad()
				logits, _ := model.Forward(x)
				loss, _ := nn.CrossEntropyLoss(logits, batchTargets)
				
				currLoss := loss.GetData()[0]
				runningLoss += currLoss

				autograd.Backward(loss)
				opt.Step()

				dashboard.AddLoss(currLoss)
				dashboard.UpdateStats(epoch, epochs, b+1, numBatches, runningLoss/float32(b+1), epochStart, totalStart)
			}

			model.Eval()
			correct := 0
			valCount := 1000 
			for i := 0; i < valCount; i++ {
				img, _ := testImg.Slice(i)
				out, _ := model.Forward(img)
				if tensor.ArgMax(out) == testLbl[i] {
					correct++
				}
			}

			accuracy := (float32(correct) / float32(valCount)) * 100.0
			duration := time.Since(epochStart)
			
			dashboard.AddAccuracy(accuracy, duration)
			dashboard.Log(fmt.Sprintf("Epoch %d: Acc %.2f%% | Time: %v", epoch, accuracy, duration.Round(time.Second)))
		}

		dashboard.Log("Final: Training finished. Press 'q' to quit.")
	}()

	dashboard.Loop()
}