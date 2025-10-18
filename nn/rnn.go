package nn

import (
	"fmt"
	"go-torch/tensor"
	"math/rand"
	"time"
)

// RNNCell represents a single RNN cell that processes one timestep
type RNNCell struct {
	Wxh        *tensor.Tensor // input-to-hidden weights [input_size, hidden_size]
	Whh        *tensor.Tensor // hidden-to-hidden weights [hidden_size, hidden_size]
	Bias       *tensor.Tensor // bias [hidden_size]
	inputSize  int
	hiddenSize int
}

// NewRNNCell creates a new RNN cell with Xavier initialization
func NewRNNCell(inputSize, hiddenSize int) (*RNNCell, error) {
	if inputSize <= 0 || hiddenSize <= 0 {
		return nil, fmt.Errorf("RNN cell dimensions must be positive, got input %d, hidden %d", inputSize, hiddenSize)
	}

	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	// Xavier initialization for better gradient flow
	// stddev = sqrt(2.0 / (input_size + hidden_size))
	stddev := 0.1 // Starting with simple initialization, can improve later
	
	// Initialize Wxh weights
	wxhData := make([]float64, inputSize*hiddenSize)
	for i := range wxhData {
		wxhData[i] = (2*random.Float64() - 1) * stddev
	}
	wxh, err := tensor.NewTensor([]int{inputSize, hiddenSize}, wxhData)
	if err != nil {
		return nil, fmt.Errorf("failed to create Wxh tensor: %w", err)
	}
	wxh.RequiresGrad = true

	// Initialize Whh weights
	whhData := make([]float64, hiddenSize*hiddenSize)
	for i := range whhData {
		whhData[i] = (2*random.Float64() - 1) * stddev
	}
	whh, err := tensor.NewTensor([]int{hiddenSize, hiddenSize}, whhData)
	if err != nil {
		return nil, fmt.Errorf("failed to create Whh tensor: %w", err)
	}
	whh.RequiresGrad = true

	// Initialize bias to zero
	biasData := make([]float64, hiddenSize)
	bias, err := tensor.NewTensor([]int{hiddenSize}, biasData)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias tensor: %w", err)
	}
	bias.RequiresGrad = true

	return &RNNCell{
		Wxh:        wxh,
		Whh:        whh,
		Bias:       bias,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
	}, nil
}

// Forward performs one timestep of RNN computation
// input: [batch_size, input_size]
// hidden: [batch_size, hidden_size] (previous hidden state)
// returns: new hidden state [batch_size, hidden_size]
func (rnn *RNNCell) Forward(input, hidden *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	hiddenShape := hidden.GetShape()
	
	if len(inputShape) != 2 || len(hiddenShape) != 2 {
		return nil, fmt.Errorf("RNN cell expects 2D tensors, got input %v, hidden %v", inputShape, hiddenShape)
	}
	
	batchSize := inputShape[0]
	if hiddenShape[0] != batchSize {
		return nil, fmt.Errorf("batch size mismatch: input %d, hidden %d", inputShape[0], hiddenShape[0])
	}
	
	if inputShape[1] != rnn.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", rnn.inputSize, inputShape[1])
	}
	
	if hiddenShape[1] != rnn.hiddenSize {
		return nil, fmt.Errorf("hidden size mismatch: expected %d, got %d", rnn.hiddenSize, hiddenShape[1])
	}

	// RNN computation: h_new = tanh(input @ Wxh + hidden @ Whh + bias)
	
	// input @ Wxh
	inputLinear, err := tensor.MatMulTensor(input, rnn.Wxh)
	if err != nil {
		return nil, fmt.Errorf("RNN cell input linear failed: %w", err)
	}

	// hidden @ Whh
	hiddenLinear, err := tensor.MatMulTensor(hidden, rnn.Whh)
	if err != nil {
		return nil, fmt.Errorf("RNN cell hidden linear failed: %w", err)
	}

	// Add the two linear transformations
	combined, err := tensor.AddTensor(inputLinear, hiddenLinear)
	if err != nil {
		return nil, fmt.Errorf("RNN cell addition failed: %w", err)
	}

	// Add bias (broadcast)
	withBias, err := tensor.AddTensorBroadcast(combined, rnn.Bias)
	if err != nil {
		return nil, fmt.Errorf("RNN cell bias addition failed: %w", err)
	}

	// Apply tanh activation
	newHidden, err := Tanh(withBias)
	if err != nil {
		return nil, fmt.Errorf("RNN cell tanh activation failed: %w", err)
	}

	return newHidden, nil
}

// Parameters returns all learnable parameters
func (rnn *RNNCell) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{rnn.Wxh, rnn.Whh, rnn.Bias}
}

// ZeroGrad zeros gradients for all parameters
func (rnn *RNNCell) ZeroGrad() {
	rnn.Wxh.ZeroGrad()
	rnn.Whh.ZeroGrad()
	rnn.Bias.ZeroGrad()
}

// RNN layer that processes sequences
type RNN struct {
	cell        *RNNCell
	inputSize   int
	hiddenSize  int
	numLayers   int
	returnSeq   bool // whether to return full sequence or just last output
}

// NewRNN creates a new RNN layer
func NewRNN(inputSize, hiddenSize, numLayers int, returnSequence bool) (*RNN, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("number of layers must be positive, got %d", numLayers)
	}
	
	// For now, implement single layer RNN
	if numLayers > 1 {
		return nil, fmt.Errorf("multi-layer RNN not yet implemented, got %d layers", numLayers)
	}

	cell, err := NewRNNCell(inputSize, hiddenSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN cell: %w", err)
	}

	return &RNN{
		cell:       cell,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		numLayers:  numLayers,
		returnSeq:  returnSequence,
	}, nil
}

// Forward processes a sequence of inputs
// input: [batch_size, seq_length, input_size]
// returns: [batch_size, seq_length, hidden_size] if returnSeq=true
//          [batch_size, hidden_size] if returnSeq=false (last timestep only)
func (rnn *RNN) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	if len(inputShape) != 3 {
		return nil, fmt.Errorf("RNN expects 3D input [batch_size, seq_length, input_size], got %v", inputShape)
	}

	batchSize := inputShape[0]
	seqLength := inputShape[1]
	inputSize := inputShape[2]

	if inputSize != rnn.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", rnn.inputSize, inputSize)
	}

	// Initialize hidden state to zeros
	hiddenData := make([]float64, batchSize*rnn.hiddenSize)
	hidden, err := tensor.NewTensor([]int{batchSize, rnn.hiddenSize}, hiddenData)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial hidden state: %w", err)
	}

	var outputs []*tensor.Tensor
	inputData := input.GetData()

	// Process each timestep
	for t := 0; t < seqLength; t++ {
		// Extract input for current timestep
		timestepData := make([]float64, batchSize*inputSize)
		for b := 0; b < batchSize; b++ {
			srcStart := b*seqLength*inputSize + t*inputSize
			dstStart := b * inputSize
			copy(timestepData[dstStart:dstStart+inputSize], inputData[srcStart:srcStart+inputSize])
		}
		
		timestepInput, err := tensor.NewTensor([]int{batchSize, inputSize}, timestepData)
		if err != nil {
			return nil, fmt.Errorf("failed to create timestep input: %w", err)
		}

		// RNN cell forward pass
		hidden, err = rnn.cell.Forward(timestepInput, hidden)
		if err != nil {
			return nil, fmt.Errorf("RNN forward failed at timestep %d: %w", t, err)
		}

		if rnn.returnSeq {
			outputs = append(outputs, hidden)
		}
	}

	if rnn.returnSeq {
		// Stack outputs to create [batch_size, seq_length, hidden_size]
		outputData := make([]float64, batchSize*seqLength*rnn.hiddenSize)
		for t, output := range outputs {
			outputSlice := output.GetData()
			for b := 0; b < batchSize; b++ {
				srcStart := b * rnn.hiddenSize
				dstStart := b*seqLength*rnn.hiddenSize + t*rnn.hiddenSize
				copy(outputData[dstStart:dstStart+rnn.hiddenSize], outputSlice[srcStart:srcStart+rnn.hiddenSize])
			}
		}
		
		result, err := tensor.NewTensor([]int{batchSize, seqLength, rnn.hiddenSize}, outputData)
		if err != nil {
			return nil, fmt.Errorf("failed to create sequence output: %w", err)
		}
		return result, nil
	}

	// Return only the last hidden state
	return hidden, nil
}

// Parameters returns all learnable parameters
func (rnn *RNN) Parameters() []*tensor.Tensor {
	return rnn.cell.Parameters()
}

// ZeroGrad zeros gradients for all parameters
func (rnn *RNN) ZeroGrad() {
	rnn.cell.ZeroGrad()
}

// Name returns the layer name
func (rnn *RNN) Name() string {
	return "RNN"
}

// Train sets the layer to training mode
func (rnn *RNN) Train() {
	// RNN doesn't have different train/eval behavior for now
}

// Eval sets the layer to evaluation mode
func (rnn *RNN) Eval() {
	// RNN doesn't have different train/eval behavior for now
}

// LSTMCell represents a single LSTM cell that processes one timestep
type LSTMCell struct {
	Wf         *tensor.Tensor // forget gate weights [input_size + hidden_size, hidden_size]
	Wi         *tensor.Tensor // input gate weights [input_size + hidden_size, hidden_size]
	Wg         *tensor.Tensor // candidate gate weights [input_size + hidden_size, hidden_size]
	Wo         *tensor.Tensor // output gate weights [input_size + hidden_size, hidden_size]
	Bf         *tensor.Tensor // forget gate bias [hidden_size]
	Bi         *tensor.Tensor // input gate bias [hidden_size]
	Bg         *tensor.Tensor // candidate gate bias [hidden_size]
	Bo         *tensor.Tensor // output gate bias [hidden_size]
	inputSize  int
	hiddenSize int
}

// NewLSTMCell creates a new LSTM cell with Xavier initialization
func NewLSTMCell(inputSize, hiddenSize int) (*LSTMCell, error) {
	if inputSize <= 0 || hiddenSize <= 0 {
		return nil, fmt.Errorf("LSTM cell dimensions must be positive, got input %d, hidden %d", inputSize, hiddenSize)
	}

	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	// Combined input size (input + hidden)
	combinedSize := inputSize + hiddenSize
	stddev := 0.1 // Starting with simple initialization
	
	// Create weight matrices for all gates
	createWeights := func() (*tensor.Tensor, error) {
		data := make([]float64, combinedSize*hiddenSize)
		for i := range data {
			data[i] = (2*random.Float64() - 1) * stddev
		}
		w, err := tensor.NewTensor([]int{combinedSize, hiddenSize}, data)
		if err != nil {
			return nil, err
		}
		w.RequiresGrad = true
		return w, nil
	}

	// Create bias vectors for all gates
	createBias := func() (*tensor.Tensor, error) {
		data := make([]float64, hiddenSize)
		// Initialize forget gate bias to 1.0 for better gradient flow
		b, err := tensor.NewTensor([]int{hiddenSize}, data)
		if err != nil {
			return nil, err
		}
		b.RequiresGrad = true
		return b, nil
	}

	Wf, err := createWeights()
	if err != nil { return nil, fmt.Errorf("failed to create forget gate weights: %w", err) }
	
	Wi, err := createWeights()
	if err != nil { return nil, fmt.Errorf("failed to create input gate weights: %w", err) }
	
	Wg, err := createWeights()
	if err != nil { return nil, fmt.Errorf("failed to create candidate weights: %w", err) }
	
	Wo, err := createWeights()
	if err != nil { return nil, fmt.Errorf("failed to create output gate weights: %w", err) }

	Bf, err := createBias()
	if err != nil { return nil, fmt.Errorf("failed to create forget gate bias: %w", err) }
	// Initialize forget gate bias to 1.0 for better gradient flow
	for i := range Bf.GetData() {
		Bf.GetData()[i] = 1.0
	}
	
	Bi, err := createBias()
	if err != nil { return nil, fmt.Errorf("failed to create input gate bias: %w", err) }
	
	Bg, err := createBias()
	if err != nil { return nil, fmt.Errorf("failed to create candidate bias: %w", err) }
	
	Bo, err := createBias()
	if err != nil { return nil, fmt.Errorf("failed to create output gate bias: %w", err) }

	return &LSTMCell{
		Wf: Wf, Wi: Wi, Wg: Wg, Wo: Wo,
		Bf: Bf, Bi: Bi, Bg: Bg, Bo: Bo,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
	}, nil
}

// Forward performs one timestep of LSTM computation
// input: [batch_size, input_size]
// hidden: [batch_size, hidden_size] (previous hidden state)
// cellState: [batch_size, hidden_size] (previous cell state)
// returns: (newHidden, newCellState) both [batch_size, hidden_size]
func (lstm *LSTMCell) Forward(input, hidden, cellState *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	inputShape := input.GetShape()
	hiddenShape := hidden.GetShape()
	cellShape := cellState.GetShape()
	
	if len(inputShape) != 2 || len(hiddenShape) != 2 || len(cellShape) != 2 {
		return nil, nil, fmt.Errorf("LSTM cell expects 2D tensors")
	}
	
	batchSize := inputShape[0]
	if hiddenShape[0] != batchSize || cellShape[0] != batchSize {
		return nil, nil, fmt.Errorf("batch size mismatch")
	}

	// Concatenate input and hidden: [batch_size, input_size + hidden_size]
	inputData := input.GetData()
	hiddenData := hidden.GetData()
	combinedData := make([]float64, batchSize*(lstm.inputSize+lstm.hiddenSize))
	
	for b := 0; b < batchSize; b++ {
		// Copy input
		inputStart := b * lstm.inputSize
		hiddenStart := b * lstm.hiddenSize
		combinedStart := b * (lstm.inputSize + lstm.hiddenSize)
		
		copy(combinedData[combinedStart:combinedStart+lstm.inputSize], 
			 inputData[inputStart:inputStart+lstm.inputSize])
		copy(combinedData[combinedStart+lstm.inputSize:combinedStart+lstm.inputSize+lstm.hiddenSize],
			 hiddenData[hiddenStart:hiddenStart+lstm.hiddenSize])
	}
	
	combined, err := tensor.NewTensor([]int{batchSize, lstm.inputSize + lstm.hiddenSize}, combinedData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create combined input: %w", err)
	}

	// Compute gates
	// Forget gate: f_t = sigmoid(combined @ Wf + Bf)
	forgetLinear, err := tensor.MatMulTensor(combined, lstm.Wf)
	if err != nil { return nil, nil, fmt.Errorf("forget gate matmul failed: %w", err) }
	forgetBiased, err := tensor.AddTensorBroadcast(forgetLinear, lstm.Bf)
	if err != nil { return nil, nil, fmt.Errorf("forget gate bias failed: %w", err) }
	forgetGate, err := Sigmoid(forgetBiased)
	if err != nil { return nil, nil, fmt.Errorf("forget gate sigmoid failed: %w", err) }

	// Input gate: i_t = sigmoid(combined @ Wi + Bi)
	inputLinear, err := tensor.MatMulTensor(combined, lstm.Wi)
	if err != nil { return nil, nil, fmt.Errorf("input gate matmul failed: %w", err) }
	inputBiased, err := tensor.AddTensorBroadcast(inputLinear, lstm.Bi)
	if err != nil { return nil, nil, fmt.Errorf("input gate bias failed: %w", err) }
	inputGate, err := Sigmoid(inputBiased)
	if err != nil { return nil, nil, fmt.Errorf("input gate sigmoid failed: %w", err) }

	// Candidate values: g_t = tanh(combined @ Wg + Bg)
	candidateLinear, err := tensor.MatMulTensor(combined, lstm.Wg)
	if err != nil { return nil, nil, fmt.Errorf("candidate matmul failed: %w", err) }
	candidateBiased, err := tensor.AddTensorBroadcast(candidateLinear, lstm.Bg)
	if err != nil { return nil, nil, fmt.Errorf("candidate bias failed: %w", err) }
	candidateValues, err := Tanh(candidateBiased)
	if err != nil { return nil, nil, fmt.Errorf("candidate tanh failed: %w", err) }

	// Output gate: o_t = sigmoid(combined @ Wo + Bo)
	outputLinear, err := tensor.MatMulTensor(combined, lstm.Wo)
	if err != nil { return nil, nil, fmt.Errorf("output gate matmul failed: %w", err) }
	outputBiased, err := tensor.AddTensorBroadcast(outputLinear, lstm.Bo)
	if err != nil { return nil, nil, fmt.Errorf("output gate bias failed: %w", err) }
	outputGate, err := Sigmoid(outputBiased)
	if err != nil { return nil, nil, fmt.Errorf("output gate sigmoid failed: %w", err) }

	// Update cell state: C_t = f_t * C_{t-1} + i_t * g_t
	forgetTerm, err := tensor.MulTensor(forgetGate, cellState)
	if err != nil { return nil, nil, fmt.Errorf("forget term multiplication failed: %w", err) }
	
	inputTerm, err := tensor.MulTensor(inputGate, candidateValues)
	if err != nil { return nil, nil, fmt.Errorf("input term multiplication failed: %w", err) }
	
	newCellState, err := tensor.AddTensor(forgetTerm, inputTerm)
	if err != nil { return nil, nil, fmt.Errorf("cell state update failed: %w", err) }

	// Update hidden state: h_t = o_t * tanh(C_t)
	cellActivated, err := Tanh(newCellState)
	if err != nil { return nil, nil, fmt.Errorf("cell state tanh failed: %w", err) }
	
	newHidden, err := tensor.MulTensor(outputGate, cellActivated)
	if err != nil { return nil, nil, fmt.Errorf("hidden state update failed: %w", err) }

	return newHidden, newCellState, nil
}

// Parameters returns all learnable parameters
func (lstm *LSTMCell) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{
		lstm.Wf, lstm.Wi, lstm.Wg, lstm.Wo,
		lstm.Bf, lstm.Bi, lstm.Bg, lstm.Bo,
	}
}

// ZeroGrad zeros gradients for all parameters
func (lstm *LSTMCell) ZeroGrad() {
	for _, param := range lstm.Parameters() {
		param.ZeroGrad()
	}
}

// LSTM layer that processes sequences
type LSTM struct {
	cell        *LSTMCell
	inputSize   int
	hiddenSize  int
	numLayers   int
	returnSeq   bool // whether to return full sequence or just last output
}

// NewLSTM creates a new LSTM layer
func NewLSTM(inputSize, hiddenSize, numLayers int, returnSequence bool) (*LSTM, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("number of layers must be positive, got %d", numLayers)
	}
	
	// For now, implement single layer LSTM
	if numLayers > 1 {
		return nil, fmt.Errorf("multi-layer LSTM not yet implemented, got %d layers", numLayers)
	}

	cell, err := NewLSTMCell(inputSize, hiddenSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM cell: %w", err)
	}

	return &LSTM{
		cell:       cell,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		numLayers:  numLayers,
		returnSeq:  returnSequence,
	}, nil
}

// Forward processes a sequence of inputs
// input: [batch_size, seq_length, input_size]
// returns: [batch_size, seq_length, hidden_size] if returnSeq=true
//          [batch_size, hidden_size] if returnSeq=false (last timestep only)
func (lstm *LSTM) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	inputShape := input.GetShape()
	if len(inputShape) != 3 {
		return nil, fmt.Errorf("LSTM expects 3D input [batch_size, seq_length, input_size], got %v", inputShape)
	}

	batchSize := inputShape[0]
	seqLength := inputShape[1]
	inputSize := inputShape[2]

	if inputSize != lstm.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", lstm.inputSize, inputSize)
	}

	// Initialize hidden state and cell state to zeros
	hiddenData := make([]float64, batchSize*lstm.hiddenSize)
	hidden, err := tensor.NewTensor([]int{batchSize, lstm.hiddenSize}, hiddenData)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial hidden state: %w", err)
	}

	cellData := make([]float64, batchSize*lstm.hiddenSize)
	cellState, err := tensor.NewTensor([]int{batchSize, lstm.hiddenSize}, cellData)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial cell state: %w", err)
	}

	var outputs []*tensor.Tensor
	inputData := input.GetData()

	// Process each timestep
	for t := 0; t < seqLength; t++ {
		// Extract input for current timestep
		timestepData := make([]float64, batchSize*inputSize)
		for b := 0; b < batchSize; b++ {
			srcStart := b*seqLength*inputSize + t*inputSize
			dstStart := b * inputSize
			copy(timestepData[dstStart:dstStart+inputSize], inputData[srcStart:srcStart+inputSize])
		}
		
		timestepInput, err := tensor.NewTensor([]int{batchSize, inputSize}, timestepData)
		if err != nil {
			return nil, fmt.Errorf("failed to create timestep input: %w", err)
		}

		// LSTM cell forward pass
		hidden, cellState, err = lstm.cell.Forward(timestepInput, hidden, cellState)
		if err != nil {
			return nil, fmt.Errorf("LSTM forward failed at timestep %d: %w", t, err)
		}

		if lstm.returnSeq {
			outputs = append(outputs, hidden)
		}
	}

	if lstm.returnSeq {
		// Stack outputs to create [batch_size, seq_length, hidden_size]
		outputData := make([]float64, batchSize*seqLength*lstm.hiddenSize)
		for t, output := range outputs {
			outputSlice := output.GetData()
			for b := 0; b < batchSize; b++ {
				srcStart := b * lstm.hiddenSize
				dstStart := b*seqLength*lstm.hiddenSize + t*lstm.hiddenSize
				copy(outputData[dstStart:dstStart+lstm.hiddenSize], outputSlice[srcStart:srcStart+lstm.hiddenSize])
			}
		}
		
		result, err := tensor.NewTensor([]int{batchSize, seqLength, lstm.hiddenSize}, outputData)
		if err != nil {
			return nil, fmt.Errorf("failed to create sequence output: %w", err)
		}
		return result, nil
	}

	// Return only the last hidden state
	return hidden, nil
}

// Parameters returns all learnable parameters
func (lstm *LSTM) Parameters() []*tensor.Tensor {
	return lstm.cell.Parameters()
}

// ZeroGrad zeros gradients for all parameters
func (lstm *LSTM) ZeroGrad() {
	lstm.cell.ZeroGrad()
}

// Name returns the layer name
func (lstm *LSTM) Name() string {
	return "LSTM"
}

// Train sets the layer to training mode
func (lstm *LSTM) Train() {
	// LSTM doesn't have different train/eval behavior for now
}

// Eval sets the layer to evaluation mode
func (lstm *LSTM) Eval() {
	// LSTM doesn't have different train/eval behavior for now
}