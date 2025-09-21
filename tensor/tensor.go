package tensor

import (
	"fmt"
	"runtime"
	"sync"
	"bytes" 
	"encoding/gob"

	"gonum.org/v1/gonum/mat" // For Dense matrix type and operations To use system-installed C BLAS
)

// NOTE: Most of the functions are self-explanatory, doesn't need much comments / explaantion (except the auto-grad part)



// fully gob functionality - for model saving 
func init() {
    gob.Register(&Tensor{})
}


type tensorGob struct {
	Shape []int
	Data  []float64
}


func (t *Tensor) GobEncode() ([]byte, error) {
	// We will encode our data into a buffer.
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	// Create an instance of our temporary exported struct and copy the data to it.
	err := encoder.Encode(tensorGob{
		Shape: t.shape,
		Data:  t.data,
	})

	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (t *Tensor) GobDecode(data []byte) error {
	// create a buffer from the incoming byte slice.
	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)

	// decode the data into our temporary exported struct.
	var tg tensorGob
	err := decoder.Decode(&tg)
	if err != nil {
		return err
	}

	// copy the data from the temporary struct back into our Tensor's unexported fields.
	t.shape = tg.Shape
	t.data = tg.Data
	
	return nil
}


// simple Tensor struct
type Tensor struct {
	shape        []int
	data         []float64
	Grad         *Tensor
	RequiresGrad bool
	Parents      []*Tensor
	Operation    string
	BackwardFunc func(*Tensor)
}

// utility function to check if two tensors have the same shape
func IsSameSize(a, b *Tensor) bool {
	aShape := a.shape
	bShape := b.shape
	if len(aShape) != len(bShape) {
		return false
	}
	for i := range aShape {
		if aShape[i] != bShape[i] {
			return false
		}
	}
	return true
}

// builds a new tensor with the given shape and data
func NewTensor(shape []int, data []float64) (*Tensor, error) {
	total := 1
	if len(shape) == 0 { // Scalar case
		if len(data) == 0 {
			total = 1 // Default scalar has 1 element
		} else if len(data) == 1 {
			total = 1
		} else {
			return nil, fmt.Errorf("scalar shape [] implies 1 element but data has length %d", len(data))
		}
	} else {
		for _, dim := range shape {
			if dim <= 0 {
				return nil, fmt.Errorf("shape %v contains non-positive dimension", shape)
			}
			total *= dim
		}
	}

	if len(data) > 0 && total != len(data) {
		return nil, fmt.Errorf("shape %v implies %d elements but data has length %d", shape, total, len(data))
	}

	// Ensure data slice is allocated if input data is nil or empty and total > 0
	var finalData []float64
	if len(data) == 0 {
		finalData = make([]float64, total) // Initialize with zeros
	} else {
		finalData = make([]float64, total)
		copy(finalData, data)
	}

	return &Tensor{
		shape:        append([]int{}, shape...), // Defensive copy of shape
		data:         finalData,                 // Use the new/copied data
		Grad:         nil,
		RequiresGrad: false,
		Parents:      nil,
		Operation:    "",
		BackwardFunc: nil,
	}, nil
}

// clones a tensor
func CloneTensor(t *Tensor) *Tensor {
	clonedData := make([]float64, len(t.data))
	copy(clonedData, t.data)

	clonedShape := append([]int{}, t.shape...)

	return &Tensor{
		data:         clonedData,
		shape:        clonedShape,
		RequiresGrad: t.RequiresGrad,
		Grad:         nil,
		Parents:      nil,
		Operation:    "",
		BackwardFunc: nil,
	}
}

// adds two tensors
func AddTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	if !IsSameSize(t1, t2) {
		return nil, fmt.Errorf("tensors %v and %v have different sizes for addition", t1, t2)
	}

	outData := make([]float64, len(t1.data))
	for i := 0; i < len(t1.data); i++ {
		outData[i] = t1.data[i] + t2.data[i]
	}

	out, err := NewTensor(t1.shape, outData)
	if err != nil {
		return nil, err
	}

	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "add"

		out.BackwardFunc = func(grad *Tensor) {
			// The gradient of addition is 1, so we just pass the incoming
			// gradient to both parents.
			if t1.RequiresGrad {
				t1.Backward(grad) // Backward now accumulates
			}
			if t2.RequiresGrad {
				t2.Backward(grad) // Backward now accumulates
			}
		}
	}
	return out, nil
}

// multiplies two tensors
func MulTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	if !IsSameSize(t1, t2) {
		return nil, fmt.Errorf("tensors %v and %v have different sizes for multiplication", t1, t2)
	}

	outData := make([]float64, len(t1.data))
	for i := 0; i < len(t1.data); i++ {
		outData[i] = t1.data[i] * t2.data[i]
	}

	out, err := NewTensor(t1.shape, outData)
	if err != nil {
		return nil, err
	}

	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "mul"

		out.BackwardFunc = func(grad *Tensor) {
			var wg sync.WaitGroup
			numGoroutines := runtime.NumCPU()
			jobsPerGo := (len(grad.data) + numGoroutines - 1) / numGoroutines

			if t1.RequiresGrad {
				if t1.Grad == nil { t1.ZeroGrad() } // Ensure grad tensor exists
				wg.Add(1)
				go func() {
					defer wg.Done()
					t1GradData := t1.Grad.GetData()
					gradData := grad.GetData()
					t2Data := t2.GetData()
					for i := 0; i < numGoroutines; i++ {
						start, end := i*jobsPerGo, (i+1)*jobsPerGo
						if end > len(gradData) { end = len(gradData) }
						if start >= end { continue }
						// Launch inner goroutine for parallel accumulation
						wg.Add(1)
						go func(s, e int) {
							defer wg.Done()
							for j := s; j < e; j++ {
								t1GradData[j] += gradData[j] * t2Data[j]
							}
						}(start, end)
					}
				}()
			}

			if t2.RequiresGrad {
				if t2.Grad == nil { t2.ZeroGrad() } // Ensure grad tensor exists
				wg.Add(1)
				go func() {
					defer wg.Done()
					t2GradData := t2.Grad.GetData()
					gradData := grad.GetData()
					t1Data := t1.GetData()
					for i := 0; i < numGoroutines; i++ {
						start, end := i*jobsPerGo, (i+1)*jobsPerGo
						if end > len(gradData) { end = len(gradData) }
						if start >= end { continue }
						wg.Add(1)
						go func(s, e int) {
							defer wg.Done()
							for j := s; j < e; j++ {
								t2GradData[j] += gradData[j] * t1Data[j]
							}
						}(start, end)
					}
				}()
			}
			wg.Wait()
		}
	}
	return out, nil
}

// returns the number of elements in a tensor
func Numel(t *Tensor) int {
	if t == nil {
		return 0
	}
	if len(t.shape) == 0 {
		if len(t.data) == 1 {
			return 1
		}
		return 0
	}
	n := 1
	for _, s := range t.shape {
		if s <= 0 {
			return 0
		}
		n *= s
	}
	return n
}

// reshapes the given tensor to the given shape
func Reshape(t *Tensor, newShape []int) (*Tensor, error) {
	originalNumel := Numel(t)
	reshapedNumel := 1
	if len(newShape) == 0 {
		if originalNumel == 1 {
			reshapedNumel = 1
		} else {
			return nil, fmt.Errorf("cannot reshape tensor with %d elements to scalar shape %v unless it has 1 element", originalNumel, newShape)
		}
	} else {
		for _, dim := range newShape {
			if dim <= 0 {
				return nil, fmt.Errorf("newShape %v contains non-positive dimension", newShape)
			}
			reshapedNumel *= dim
		}
	}

	if originalNumel != reshapedNumel {
		return nil, fmt.Errorf("cannot reshape tensor with %d elements to shape %v (requires %d elements)", originalNumel, newShape, reshapedNumel)
	}

	outData := make([]float64, len(t.data))
	copy(outData, t.data)

	out, err := NewTensor(newShape, outData)
	if err != nil {
		return nil, err
	}

	if t.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t}
		out.Operation = "reshape"
		out.BackwardFunc = func(grad *Tensor) {
			if t.RequiresGrad {
				// The gradient must be reshaped back to the parent's original shape.
				// The data itself doesn't change, just the view.
				reshapedGradForT, _ := NewTensor(append([]int{}, t.shape...), grad.data)
				t.Backward(reshapedGradForT)
			}
		}
	}
	return out, nil
}

// this defines the GetData() and GetShape() accessors, used for testing & debugging
func (t *Tensor) GetData() []float64 {
	return t.data
}

func (t *Tensor) GetShape() []int {
	return t.shape
}

// returns a tensor with all elements set to 1
func OnesLike(t *Tensor) (*Tensor, error) {
	shape := append([]int{}, t.shape...)
	size := Numel(t)
	if size == 0 && len(t.shape) > 0 {
		validShape := true
		for _, dim := range t.shape {
			if dim <= 0 {
				validShape = false
				break
			}
		}
		if !validShape {
			return nil, fmt.Errorf("cannot create OnesLike for tensor with invalid shape %v resulting in 0 elements", t.shape)
		}
	}

	data := make([]float64, size)
	for i := range data {
		data[i] = 1
	}
	newT, err := NewTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return newT, nil
}

// sets the gradient of a tensor to zero
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad.data {
			t.Grad.data[i] = 0
		}
	} else if t.RequiresGrad {
		numElements := Numel(t)
		if numElements < 0 {
			numElements = 0
		}
		zeroData := make([]float64, numElements)
		gradTensor, err := NewTensor(append([]int{}, t.shape...), zeroData)
		if err == nil {
			gradTensor.RequiresGrad = false
			t.Grad = gradTensor
		}
	}
}

// computes the backward pass for a tensor
func (t *Tensor) Backward(grad *Tensor) {
	if !t.RequiresGrad {
		return
	}

	// This function is now ONLY a gradient accumulator.
	// The autograd engine is responsible for calling t.BackwardFunc.

	if grad == nil {
		// This should ideally not be hit if using the autograd engine,
		// which always starts with a gradient of 1.0 for the root.
		if Numel(t) == 1 {
			grad, _ = NewTensor(t.shape, []float64{1.0})
		} else {
			// Cannot proceed without a gradient for a non-scalar tensor.
			return
		}
	}

	if !IsSameSize(t, grad) {
		fmt.Printf("Error: Mismatch in shape during backward accumulation. Tensor shape: %v, Grad shape: %v\n", t.shape, grad.shape)
		return
	}

	if t.Grad == nil {
		// If grad tensor doesn't exist, create it by copying the incoming grad.
		gradDataCopy := make([]float64, len(grad.data))
		copy(gradDataCopy, grad.data)
		initializedGrad, err := NewTensor(append([]int{}, t.shape...), gradDataCopy)
		if err == nil {
			initializedGrad.RequiresGrad = false
			t.Grad = initializedGrad
		} else {
			fmt.Printf("Error initializing gradient tensor: %v\n", err)
		}
	} else {
		// If it exists, accumulate.
		for i := range t.Grad.data {
			t.Grad.data[i] += grad.data[i]
		}
	}
}

// Transpose function (no changes from your provided version)
func Transpose(t *Tensor) (*Tensor, error) {
	shape := t.GetShape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("transpose requires a tensor with at least 2 dimensions, got %v", shape)
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("transpose currently only supports 2D tensors, got %v", shape)
	}

	newShape := append([]int{}, shape...)
	newShape[0], newShape[1] = newShape[1], newShape[0] // Corrected for 2D transpose

	originalNumel := Numel(t)
	tData := t.GetData()
	outData := make([]float64, originalNumel)

	M, N := shape[0], shape[1]
	for r := 0; r < M; r++ {
		for c := 0; c < N; c++ {
			originalFlatIndex := r*N + c
			transposedFlatIndex := c*M + r // N_new * r_new + c_new => M * c + r
			outData[transposedFlatIndex] = tData[originalFlatIndex]
		}
	}

	out, err := NewTensor(newShape, outData)
	if err != nil {
		return nil, fmt.Errorf("transpose failed to create output tensor: %w", err)
	}

	if t.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t}
		out.Operation = "transpose"
		out.BackwardFunc = func(grad *Tensor) {
			if t.RequiresGrad {
				transposedGrad, err := Transpose(grad)
				if err != nil {
					fmt.Printf("Warning: Failed to transpose gradient in Transpose backward: %v\n", err)
					return
				}
				t.Backward(transposedGrad)
			}
		}
	}
	return out, nil
}

const blasThreshold = 64 


func matMulWithTranspose(t1, t2 *Tensor, transposeT1, transposeT2 bool) (*Tensor, error) {
	shape1 := t1.GetShape()
	shape2 := t2.GetShape()

	gonumT1 := mat.NewDense(shape1[0], shape1[1], t1.GetData())
	gonumT2 := mat.NewDense(shape2[0], shape2[1], t2.GetData())

	var a, b mat.Matrix = gonumT1, gonumT2
	if transposeT1 {
		a = a.T()
	}
	if transposeT2 {
		b = b.T()
	}

	r, c := a.Dims()
	br, bc := b.Dims()
	if c != br {
		return nil, fmt.Errorf("matmul incompatible shapes after transpose")
	}

	gonumOut := mat.NewDense(r, bc, nil)
	gonumOut.Mul(a, b)

	outShape := []int{r, bc}
	rawData := gonumOut.RawMatrix().Data
	outData := make([]float64, len(rawData))
	copy(outData, rawData)
	
	return NewTensor(outShape, outData)
}



// MatMulTensor performs matrix multiplication between two tensors.
// Uses BLAS (via gonum) for larger matrices, falls back to parallel Go for smaller ones.
func MatMulTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	shape1 := t1.GetShape()
	shape2 := t2.GetShape()

	if len(shape1) != 2 || len(shape2) != 2 {
		return nil, fmt.Errorf("matmul supports 2D tensors only, got %v and %v", shape1, shape2)
	}

	M := shape1[0]
	K1 := shape1[1]
	K2 := shape2[0]
	N := shape2[1]

	if K1 != K2 {
		return nil, fmt.Errorf("matmul incompatible shapes: inner dimensions mismatch %v (%d) and %v (%d)", shape1, K1, shape2, K2)
	}
	K := K1

	outShape := []int{M, N}
	var outData []float64
	var err error

	// Decide whether to use BLAS or pure Go parallel version
	// You can tune blasThreshold based on benchmarking
	useBLAS := M > blasThreshold && K > blasThreshold && N > blasThreshold && M*K*N > blasThreshold*blasThreshold*blasThreshold/10 // Heuristic

	if useBLAS {
		// --- BLAS Version using gonum ---
		// Convert t1 and t2 data to gonum.Dense matrices
		// gonum.Dense expects data in row-major order if a slice is provided
		gonumT1 := mat.NewDense(M, K, t1.GetData()) // Assumes t1.GetData() returns a fresh slice or is safe to use
		gonumT2 := mat.NewDense(K, N, t2.GetData()) // Assumes t2.GetData() returns a fresh slice or is safe to use

		gonumOut := mat.NewDense(M, N, nil) // nil data means it will allocate its own

		// Perform matrix multiplication: C = A * B
		gonumOut.Mul(gonumT1, gonumT2)

		// Get the data back from gonumOut. RawMatrix().Data returns the backing slice.
		// It's important to copy this data if gonumOut might be modified elsewhere,
		// or if our Tensor expects to own its data slice.
		// For now, we assume we can copy it into a new slice for our Tensor.
		rawData := gonumOut.RawMatrix().Data
		outData = make([]float64, len(rawData))
		copy(outData, rawData)

	} else {
		// if not blas, fall back to parallel Go
		outData = make([]float64, M*N)
		t1Data := t1.GetData()
		t2Transposed, transposeErr := Transpose(t2) // Cache locality
		if transposeErr != nil {
			return nil, fmt.Errorf("matmul (go version) failed to transpose t2: %w", transposeErr)
		}
		t2TransposedData := t2Transposed.GetData()

		numGoroutines := runtime.NumCPU()
		rowsPerGoroutine := (M + numGoroutines - 1) / numGoroutines

		var wg sync.WaitGroup
		for i := 0; i < numGoroutines; i++ {
			startRow := i * rowsPerGoroutine
			endRow := (i + 1) * rowsPerGoroutine
			if endRow > M {
				endRow = M
			}
			if startRow >= endRow {
				continue
			}

			wg.Add(1)
			go func(sR, eR int) {
				defer wg.Done()
				for rIdx := sR; rIdx < eR; rIdx++ {
					for cIdx := 0; cIdx < N; cIdx++ {
						sum := 0.0
						t1RowOffset := rIdx * K
						t2TRowOffset := cIdx * K // t2Transposed is N x K, so cIdx is row index
						for kIdx := 0; kIdx < K; kIdx++ {
							sum += t1Data[t1RowOffset+kIdx] * t2TransposedData[t2TRowOffset+kIdx]
						}
						outData[rIdx*N+cIdx] = sum
					}
				}
			}(startRow, endRow)
		}
		wg.Wait()
	}

	out, err := NewTensor(outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("matmul failed to create output tensor: %w", err)
	}

	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "matmul"

		out.BackwardFunc = func(grad *Tensor) {
			// dL/dX = dL/dO @ W.T  => grad @ t2.T
			// dL/dW = X.T @ dL/dO  => t1.T @ grad
			
			if t1.RequiresGrad {
				gradForT1, err := matMulWithTranspose(grad, t2, false, true) // grad * t2.T
				if err != nil {
					fmt.Printf("Warning: MatMul backward failed to compute grad for t1: %v\n", err)
				} else {
					t1.Backward(gradForT1)
				}
			}

			if t2.RequiresGrad {
				gradForT2, err := matMulWithTranspose(t1, grad, true, false) // t1.T * grad
				if err != nil {
					fmt.Printf("Warning: MatMul backward failed to compute grad for t2: %v\n", err)
				} else {
					t2.Backward(gradForT2)
				}
			}
		}
	}

	return out, nil
}


// Permute reorders the dimensions of a tensor according to the given axes.
func Permute(t *Tensor, axes []int) (*Tensor, error) {
	if len(t.shape) != len(axes) {
		return nil, fmt.Errorf("permute: number of axes (%d) must match tensor rank (%d)", len(axes), len(t.shape))
	}
	
	newShape := make([]int, len(t.shape))
	for i, axis := range axes {
		newShape[i] = t.shape[axis]
	}
	
	outData := make([]float64, Numel(t))
	
	// A fast path for the common Conv2D case: [C, B, H, W] -> [B, C, H, W] (axes {1, 0, 2, 3})
	if len(t.shape) == 4 && len(axes) == 4 && axes[0] == 1 && axes[1] == 0 && axes[2] == 2 && axes[3] == 3 {
		C, B, H, W := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
		tData := t.GetData()
		for b := 0; b < B; b++ {
			for c := 0; c < C; c++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						srcIndex := c*(B*H*W) + b*(H*W) + h*W + w
						destIndex := b*(C*H*W) + c*(H*W) + h*W + w
						outData[destIndex] = tData[srcIndex]
					}
				}
			}
		}
	} else {
		// A generic N-D implementation is complex and not included here for brevity.
		return nil, fmt.Errorf("permute currently only supports the specific [1, 0, 2, 3] permutation for 4D tensors")
	}

	out, err := NewTensor(newShape, outData)
	if err != nil {
		return nil, err
	}

	if t.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t}
		out.Operation = "permute"
		out.BackwardFunc = func(grad *Tensor) {
			if t.RequiresGrad {
				inverseAxes := make([]int, len(axes))
				for i, axis := range axes {
					inverseAxes[axis] = i
				}
				gradForT, err := Permute(grad, inverseAxes)
				if err != nil {
					panic(fmt.Sprintf("error permuting gradient: %v", err))
				}
				t.Backward(gradForT)
			}
		}
	}

	return out, nil
}


// AddTensorBroadcast performs element-wise addition with broadcasting.
// It currently supports adding a 1D tensor (b) to a 4D tensor (a).
// Shape a: [Batch, Channels, H, W], Shape b: [Channels]
func AddTensorBroadcast(a *Tensor, b *Tensor) (*Tensor, error) {
	aShape := a.GetShape()
	bShape := b.GetShape()

	outData := make([]float64, Numel(a))
	aData := a.GetData()
	bData := b.GetData()

	// 4D Conv2D case
	if len(aShape) == 4 && len(bShape) == 1 && aShape[1] == bShape[0] {
		B, C, H, W := aShape[0], aShape[1], aShape[2], aShape[3]
		numGoroutines := runtime.NumCPU()
		var wg sync.WaitGroup
		batchesPerGo := (B + numGoroutines - 1) / numGoroutines
		for i := 0; i < numGoroutines; i++ {
			startBatch := i * batchesPerGo
			endBatch := (i + 1) * batchesPerGo
			if endBatch > B {
				endBatch = B
			}
			if startBatch >= endBatch {
				continue
			}

			wg.Add(1)
			go func(sB, eB int) {
				defer wg.Done()
				for b_idx := sB; b_idx < eB; b_idx++ {
					for c_idx := 0; c_idx < C; c_idx++ {
						biasVal := bData[c_idx]
						offset := b_idx*(C*H*W) + c_idx*(H*W)
						for i := 0; i < H*W; i++ {
							idx := offset + i
							outData[idx] = aData[idx] + biasVal
						}
					}
				}
			}(startBatch, endBatch)
		}
		wg.Wait()

		// 2D Linear layer case
	} else if len(aShape) == 2 && len(bShape) == 1 && aShape[1] == bShape[0] {
		B, F := aShape[0], aShape[1]
		numGoroutines := runtime.NumCPU()
		var wg sync.WaitGroup
		rowsPerGo := (B + numGoroutines - 1) / numGoroutines
		for i := 0; i < numGoroutines; i++ {
			startRow := i * rowsPerGo
			endRow := (i + 1) * rowsPerGo
			if endRow > B {
				endRow = B
			}
			if startRow >= endRow {
				continue
			}

			wg.Add(1)
			go func(sR, eR int) {
				defer wg.Done()
				for r := sR; r < eR; r++ {
					offset := r * F
					for c := 0; c < F; c++ {
						outData[offset+c] = aData[offset+c] + bData[c]
					}
				}
			}(startRow, endRow) 
		}
		wg.Wait()
	} else {
		return nil, fmt.Errorf("unsupported broadcast: a=%v, b=%v. Only 4D+1D and 2D+1D supported", aShape, bShape)
	}

	out, err := NewTensor(aShape, outData)
	if err != nil {
		return nil, err
	}

	if a.RequiresGrad || b.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{a, b}
		out.Operation = "add_broadcast"
		out.BackwardFunc = func(grad *Tensor) {
			if a.RequiresGrad {
				a.Backward(grad) // For a, grad flows straight through
			}
			if b.RequiresGrad {
				if b.Grad == nil {
					b.ZeroGrad()
				}
				gradData := grad.GetData()
				bGradData := b.Grad.GetData()

				// Sum gradients across the broadcasted dimension(s)
				if len(aShape) == 4 { // 4D case
					B, C, H, W := aShape[0], aShape[1], aShape[2], aShape[3]
					for i := 0; i < B; i++ {
						for j := 0; j < C; j++ {
							offset := i*(C*H*W) + j*(H*W)
							sum := 0.0
							for k := 0; k < H*W; k++ {
								sum += gradData[offset+k]
							}
							bGradData[j] += sum
						}
					}
				} else if len(aShape) == 2 { // 2D case
					B, F := aShape[0], aShape[1]
					for c := 0; c < F; c++ {
						sum := 0.0
						for r := 0; r < B; r++ {
							sum += gradData[r*F+c]
						}
						bGradData[c] += sum
					}
				}
			}
		}
	}
	return out, nil
}


// returns the index of the maximum value in a tensor.
// effectively flattens the tensor before finding the index.
func ArgMax(t *Tensor) int {
	data := t.GetData()
	if len(data) == 0 {
		return -1 
	}
	maxIdx := 0
	maxVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
			maxIdx = i
		}
	}
	return maxIdx
}


// extracts a single item from a 4D tensor along the batch dimension (axis 0).
func (t *Tensor) Slice(index int) (*Tensor, error) {
	shape := t.GetShape() // Expects [B, C, H, W]
	if len(shape) != 4 {
		return nil, fmt.Errorf("slice is only supported for 4D tensors, got %dD", len(shape))
	}
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	if index < 0 || index >= B {
		return nil, fmt.Errorf("slice index %d out of bounds for batch size %d", index, B)
	}
	
	newShape := []int{1, C, H, W}
	numelPerItem := C * H * W
	start := index * numelPerItem
	end := start + numelPerItem
	
	newData := make([]float64, numelPerItem)
	copy(newData, t.data[start:end])
	
	return NewTensor(newShape, newData)
}


// prints the tensor in readable format
func PrintTensor(t *Tensor) {
	if t == nil {
		fmt.Println("<nil tensor>")
		return
	}
	fmt.Printf("Tensor(shape=%v, data=%v, requires_grad=%v", t.shape, t.data, t.RequiresGrad)
	if t.Grad != nil {
		fmt.Printf(", grad_data=%v (shape=%v)", t.Grad.data, t.Grad.shape)
	}
	if t.Operation != "" {
		fmt.Printf(", op=%s", t.Operation)
	}
	fmt.Println(")")
}