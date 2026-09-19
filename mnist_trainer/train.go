package main

import (

	"encoding/binary"
	"fmt"
	"go-torch/autograd"
	"go-torch/nn"
	"go-torch/optimizer"
	"go-torch/safetensors"
	"go-torch/tensor"
	"io"
	"log"
	"math/rand"
	"os"
	"time"
	"encoding/gob"

)



const (
	mnistDir = "mnist_data"

	// upper bounds used to sanity-check MNIST header fields before trusting
	// them for an allocation size. Real MNIST files are ~60k images of
	// 28x28, so these are generous but still protect against a corrupt or
	// maliciously crafted file causing an int32 overflow or a huge alloc.
	maxMNISTItems = 10_000_000
	maxMNISTDim   = 1 << 16
)


func loadImages(filepath string) (*tensor.Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil { return nil, err }
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

	// Validate header values before trusting them for an allocation size.
	// Without this, a truncated/corrupt/crafted file can drive numImages *
	// numRows * numCols to overflow int32 (wrapping to a small positive or
	// negative number) and produce a mismatched allocation, or simply try
	// to allocate an enormous slice.
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


// loads label files
func loadLabels(filepath string) ([]int, error) {
	file, err := os.Open(filepath)
	if err != nil { return nil, err }
	defer file.Close()

	var reader io.Reader = file
	
	var magic, numLabels int32
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number from %s: %w", filepath, err)
	}
	// 2049 is a magic number used to distinguish image files from label files
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


// --- Model Definition -
type SimpleCNN struct {
	Conv1   *nn.Conv2D
	Pool1   *nn.MaxPooling2D
	Conv2   *nn.Conv2D
	Pool2   *nn.MaxPooling2D
	Flatten *nn.Flatten
	Linear1 *nn.Linear
	Linear2 *nn.Linear
}

func NewSimpleCNN(numClasses int) (*SimpleCNN, error) {
	conv1, err := nn.NewConv2D(1, 16, 5, 1, 2)
	if err != nil { return nil, err }
	pool1 := nn.NewMaxPooling2D(2, 2)

	conv2, err := nn.NewConv2D(16, 32, 5, 1, 2)
	if err != nil { return nil, err }
	pool2 := nn.NewMaxPooling2D(2, 2)

	flatten := nn.NewFlatten()
	linear1, err := nn.NewLinear(32*7*7, 128)
	if err != nil { return nil, err }
	linear2, err := nn.NewLinear(128, numClasses)
	if err != nil { return nil, err }

	return &SimpleCNN{
		Conv1: conv1, Pool1: pool1, Conv2: conv2, Pool2: pool2,
		Flatten: flatten, Linear1: linear1, Linear2: linear2,
	}, nil
}

func (m *SimpleCNN) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	x, err = m.Conv1.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }
	x, err = m.Pool1.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Conv2.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }
	x, err = m.Pool2.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Flatten.Forward(x)
	if err != nil { return nil, err }

	x, err = m.Linear1.Forward(x)
	if err != nil { return nil, err }
	x, err = nn.RELU(x)
	if err != nil { return nil, err }

	logits, err := m.Linear2.Forward(x)
	if err != nil { return nil, err }

	return logits, nil
}

func (m *SimpleCNN) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Conv1.Parameters()...)
	params = append(params, m.Conv2.Parameters()...)
	params = append(params, m.Linear1.Parameters()...)
	params = append(params, m.Linear2.Parameters()...)
	return params
}

func (m *SimpleCNN) ZeroGrad() {
	m.Conv1.ZeroGrad()
	m.Conv2.ZeroGrad()
	m.Linear1.ZeroGrad()
	m.Linear2.ZeroGrad()
}

// NamedParameters builds a full state dict for the model - each parameter tensor is keyed by a name that includes the layer name and parameter type (e.g. "conv1.weight", "linear2.bias"). used for safetensors save/load.
func (m *SimpleCNN) NamedParameters() map[string]*tensor.Tensor {
	named := make(map[string]*tensor.Tensor)
	for k, v := range m.Conv1.NamedParameters("conv1") {
		named[k] = v
	}
	for k, v := range m.Conv2.NamedParameters("conv2") {
		named[k] = v
	}
	for k, v := range m.Linear1.NamedParameters("linear1") {
		named[k] = v
	}
	for k, v := range m.Linear2.NamedParameters("linear2") {
		named[k] = v
	}
	return named
}

// writes the model's parameters to a safetensors file
func (m *SimpleCNN) SaveSafetensors(filepath string) error {
	metadata := map[string]string{"format": "go-torch-simplecnn-v1"}
	if err := safetensors.Save(filepath, m.NamedParameters(), metadata); err != nil {
		return fmt.Errorf("could not save safetensors checkpoint to %s: %w", filepath, err)
	}
	return nil
}

// reads a safetensors file and loads the model's parameters from it, checking that the shapes match
func (m *SimpleCNN) LoadSafetensors(filepath string) error {
	loaded, _, err := safetensors.Load(filepath)
	if err != nil {
		return fmt.Errorf("could not load safetensors checkpoint from %s: %w", filepath, err)
	}

	named := m.NamedParameters()
	for name, dest := range named {
		src, ok := loaded[name]
		if !ok {
			return fmt.Errorf("checkpoint %s is missing tensor %q required by this model", filepath, name)
		}
		if !tensor.IsSameSize(dest, src) {
			return fmt.Errorf("checkpoint %s: tensor %q has shape %v, model expects %v", filepath, name, src.GetShape(), dest.GetShape())
		}
		copy(dest.GetData(), src.GetData())
	}
	return nil
}

func (m *SimpleCNN) Save(filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("could not create file %s: %w", filepath, err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	// we are saving the slice of all parameter tensors.
	err = encoder.Encode(m.Parameters())
	if err != nil {
		return fmt.Errorf("could not encode parameters to file %s: %w", filepath, err)
	}
	return nil
}


// loads model parameters from a gob file, checking that the shapes match
func (m *SimpleCNN) Load(filepath string) (err error) {
	file, openErr := os.Open(filepath)
	if openErr != nil {
		return fmt.Errorf("could not open file %s: %w", filepath, openErr)
	}
	defer file.Close()

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic while decoding model file %s (file may be corrupt or untrusted): %v", filepath, r)
		}
	}()

	decoder := gob.NewDecoder(file)

	var savedParams []*tensor.Tensor
	if decodeErr := decoder.Decode(&savedParams); decodeErr != nil {
		return fmt.Errorf("could not decode parameters from file %s: %w", filepath, decodeErr)
	}

	modelParams := m.Parameters()
	if len(savedParams) != len(modelParams) {
		return fmt.Errorf("parameter count mismatch: saved model has %d, current model has %d", len(savedParams), len(modelParams))
	}

	for i, param := range modelParams {
		savedParam := savedParams[i]
		if !tensor.IsSameSize(param, savedParam) {
			return fmt.Errorf("shape mismatch for parameter %d: saved shape %v, model shape %v", i, savedParam.GetShape(), param.GetShape())
		}
		copy(param.GetData(), savedParam.GetData())
	}

	return nil
}


func main() {
	rand.Seed(time.Now().UnixNano())


	// -- Load Data --
	fmt.Println("Preparing MNIST dataset from local files...")

	trainImagesPath := fmt.Sprintf("%s/%s/%s", mnistDir, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
	trainLabelsPath := fmt.Sprintf("%s/%s/%s", mnistDir, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
	testImagesPath := fmt.Sprintf("%s/%s/%s", mnistDir, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
	testLabelsPath := fmt.Sprintf("%s/%s/%s", mnistDir, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")

	trainImages, err := loadImages(trainImagesPath)
	if err != nil { log.Fatalf("Failed to load train images from %s: %v", trainImagesPath, err) }
	trainLabels, err := loadLabels(trainLabelsPath)
	if err != nil { log.Fatalf("Failed to load train labels from %s: %v", trainLabelsPath, err) }

	testImages, err := loadImages(testImagesPath)
	if err != nil { log.Fatalf("Failed to load test images from %s: %v", testImagesPath, err) }
	testLabels, err := loadLabels(testLabelsPath)
	if err != nil { log.Fatalf("Failed to load test labels from %s: %v", testLabelsPath, err) }

	fmt.Printf("Loaded %d training images and %d test images.\n", trainImages.GetShape()[0], testImages.GetShape()[0])



	// -- Initialize Model and Optimizer --
	numClasses := 10
	learningRate := 0.01
	batchSize := 32
	epochs := 1
	modelSavePath := "mnist_cnn.safetensors" // named, shape-checked format 

	model, err := NewSimpleCNN(numClasses)
	if err != nil { log.Fatalf("Failed to create model: %v", err) }

	optimizer, err := optimizer.NewSGD(model.Parameters(), learningRate, 0.0)
	if err != nil { log.Fatalf("Failed to create optimizer: %v", err) }


	// -- Training Loop --
	numTrainSamples := trainImages.GetShape()[0]
	numBatches := (numTrainSamples + batchSize - 1) / batchSize

	fmt.Println("\n--- Starting Training ---")

	for epoch := 0; epoch < epochs; epoch++ {
		epochStartTime := time.Now()
		runningLoss := 0.0

		indices := rand.Perm(numTrainSamples)

		for i := 0; i < numBatches; i++ {
			start := i * batchSize
			end := start + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}
			batchIndices := indices[start:end]

			// create batches
			currentBatchSize := len(batchIndices)
			if currentBatchSize == 0 {
				continue
			}
			batchImagesData := make([]float64, currentBatchSize*28*28)
			batchLabels := make([]int, currentBatchSize)

			allImageData := trainImages.GetData()
			for j, idx := range batchIndices {
				imgStart := idx * 28 * 28
				copy(batchImagesData[j*28*28:(j+1)*28*28], allImageData[imgStart:imgStart+28*28])
				batchLabels[j] = trainLabels[idx]
			}

			batchShape := []int{currentBatchSize, 1, 28, 28}
			batchTensor, _ := tensor.NewTensor(batchShape, batchImagesData)

			// --- Training step ---
			model.ZeroGrad()
			logits, err := model.Forward(batchTensor)
			if err != nil { log.Fatalf("Epoch %d, batch %d: forward pass failed: %v", epoch, i, err) }

			loss, err := nn.CrossEntropyLoss(logits, batchLabels)
			if err != nil { log.Fatalf("Epoch %d, batch %d: loss calculation failed: %v", epoch, i, err) }
			runningLoss += loss.GetData()[0]

			autograd.Backward(loss)
			err = optimizer.Step()
			if err != nil { log.Fatalf("Epoch %d, batch %d: optimizer step failed: %v", epoch, i, err) }

			// logging
			percentComplete := float64(i+1) / float64(numBatches) * 100
			fmt.Printf("\rEpoch %d/%d [%-50s] %3.0f%% - Avg Loss: %.4f",
				epoch+1,
				epochs,
				buildProgressBar(percentComplete),
				percentComplete,
				runningLoss/float64(i+1))
		}
		
		fmt.Println() 
		epochDuration := time.Since(epochStartTime)
		fmt.Printf("Epoch %d completed in %v.\n", epoch+1, epochDuration)


		// eval 
		fmt.Println("Running evaluation on test set...")
		correct := 0
		numTestSamples := testImages.GetShape()[0]
		allTestImagesData := testImages.GetData()

		for i := 0; i < numTestSamples; i += batchSize {
			end := i + batchSize
			if end > numTestSamples {
				end = numTestSamples
			}
			currentBatchSize := end - i

			batchImagesData := make([]float64, currentBatchSize*28*28)
			batchLabels := testLabels[i:end]

			startData := i * 28 * 28
			endData := end * 28 * 28
			copy(batchImagesData, allTestImagesData[startData:endData])

			batchShape := []int{len(batchLabels), 1, 28, 28}
			batchTensor, _ := tensor.NewTensor(batchShape, batchImagesData)
			logits, _ := model.Forward(batchTensor)

			logitsData := logits.GetData()
			numClasses := logits.GetShape()[1]
			for j := 0; j < len(batchLabels); j++ {
				maxLogit := -1e9
				prediction := -1
				for k := 0; k < numClasses; k++ {
					logit := logitsData[j*numClasses+k]
					if logit > maxLogit {
						maxLogit = logit
						prediction = k
					}
				}
				if prediction == batchLabels[j] {
					correct++
				}
			}
		}
		accuracy := float64(correct) / float64(numTestSamples)
		fmt.Printf("Test Accuracy: %.2f%%\n\n", accuracy*100)
	}
	fmt.Println("--- Training Complete ---")


	// -- Save the Trained Model --
	fmt.Printf("\nSaving trained model to %s...\n", modelSavePath)
	err = model.SaveSafetensors(modelSavePath)
	if err != nil {
		log.Fatalf("Error saving model: %v", err)
	}
	fmt.Println("Model saved successfully.")


	// -- demonstrate Loading the Model --
	fmt.Println("\n--- Demonstrating Model Loading ---")


	newModel, err := NewSimpleCNN(numClasses)
	if err != nil {
		log.Fatalf("Error creating new model instance for loading: %v", err)
	}

	// load the saved parameters into the new model instance.
	fmt.Printf("Loading parameters from %s into new model instance...\n", modelSavePath)
	err = newModel.LoadSafetensors(modelSavePath)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}
	fmt.Println("Model loaded successfully.")


	fmt.Println("Evaluating loaded model...")
	correct := 0
	numTestSamples := testImages.GetShape()[0]
	allTestImagesData := testImages.GetData()

	for i := 0; i < numTestSamples; i += batchSize {
		end := i + batchSize
		if end > numTestSamples {
			end = numTestSamples
		}
		currentBatchSize := end - i
		batchImagesData := make([]float64, currentBatchSize*28*28)
		batchLabels := testLabels[i:end]
		startData := i * 28 * 28
		endData := end * 28 * 28
		copy(batchImagesData, allTestImagesData[startData:endData])
		batchShape := []int{len(batchLabels), 1, 28, 28}
		batchTensor, _ := tensor.NewTensor(batchShape, batchImagesData)
		logits, _ := newModel.Forward(batchTensor) // Use newModel here!
		
		logitsData := logits.GetData()
		for j := 0; j < len(batchLabels); j++ {
			maxLogit := -1e9
			prediction := -1
			for k := 0; k < numClasses; k++ {
				logit := logitsData[j*numClasses+k]
				if logit > maxLogit {
					maxLogit = logit
					prediction = k
				}
			}
			if prediction == batchLabels[j] {
				correct++
			}
		}
	}
	accuracy := float64(correct) / float64(numTestSamples)
	fmt.Printf("Accuracy of loaded model: %.2f%%\n", accuracy*100)
}


// buildProgressBar is a helper function to create the visual progress bar string.
func buildProgressBar(percent float64) string {
	barWidth := 50
	progress := int(percent / 100.0 * float64(barWidth))
	
	bar := ""
	for i := 0; i < progress; i++ {
		bar += "="
	}
	if progress < barWidth {
		bar += ">"
	}
	for i := progress + 1; i < barWidth; i++ {
		bar += " "
	}
	return bar
}