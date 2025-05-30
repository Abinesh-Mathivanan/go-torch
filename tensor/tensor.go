package tensor

import (
	"fmt"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/mat" // For Dense matrix type and operations To use system-installed C BLAS
)

// NOTE: Most of the functions are self-explanatory, doesn't need much comments / explaantion (except the auto-grad part)

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
			if t1.RequiresGrad {
				gradDataForT1 := make([]float64, len(grad.data))
				copy(gradDataForT1, grad.data)
				t1Grad, _ := NewTensor(t1.shape, gradDataForT1)
				t1.Backward(t1Grad)
			}
			if t2.RequiresGrad {
				gradDataForT2 := make([]float64, len(grad.data))
				copy(gradDataForT2, grad.data)
				t2Grad, _ := NewTensor(t2.shape, gradDataForT2)
				t2.Backward(t2Grad)
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
			if t1.RequiresGrad {
				gradDataForT1 := make([]float64, len(grad.data))
				for i := range gradDataForT1 {
					gradDataForT1[i] = grad.data[i] * t2.data[i]
				}
				t1Grad, _ := NewTensor(t1.shape, gradDataForT1)
				t1.Backward(t1Grad)
			}
			if t2.RequiresGrad {
				gradDataForT2 := make([]float64, len(grad.data))
				for i := range gradDataForT2 {
					gradDataForT2[i] = grad.data[i] * t1.data[i]
				}
				t2Grad, _ := NewTensor(t2.shape, gradDataForT2)
				t2.Backward(t2Grad)
			}
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
				gradDataForT := make([]float64, len(grad.data))
				copy(gradDataForT, grad.data)
				reshapedGradForT, _ := NewTensor(append([]int{}, t.shape...), gradDataForT)
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

	currentNumel := Numel(t)

	if grad == nil {
		if currentNumel == 1 {
			tempGradData := []float64{1.0}
			grad, _ = NewTensor(append([]int{}, t.shape...), tempGradData)
		} else {
			fmt.Printf("Warning: Tensor.Backward called with nil grad on non-scalar tensor %v. This might be an issue.\n", t.shape)
			return
		}
	} else if !IsSameSize(t, grad) {
		fmt.Printf("Error: Mismatch in shape during backward. Tensor shape: %v, Grad shape: %v\n", t.shape, grad.shape)
		return
	}

	if t.Grad == nil {
		gradDataCopy := make([]float64, len(grad.data))
		copy(gradDataCopy, grad.data)
		initializedGrad, err := NewTensor(append([]int{}, t.shape...), gradDataCopy)
		if err == nil {
			initializedGrad.RequiresGrad = false
			t.Grad = initializedGrad
		} else {
			fmt.Printf("Error initializing gradient tensor: %v\n", err)
			return
		}
	} else {
		for i := range t.Grad.data {
			t.Grad.data[i] += grad.data[i]
		}
	}

	if t.BackwardFunc != nil {
		t.BackwardFunc(grad)
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
			// These will themselves use the (potentially BLAS-accelerated) MatMulTensor

			if t1.RequiresGrad {
				t2T_for_grad, err_t2t := Transpose(t2)
				if err_t2t != nil {
					fmt.Printf("Warning: MatMul backward failed to transpose t2: %v\n", err_t2t)
				} else {
					gradForT1, err_gt1 := MatMulTensor(grad, t2T_for_grad) // Recursive call
					if err_gt1 != nil {
						fmt.Printf("Warning: MatMul backward failed to compute grad for t1: %v\n", err_gt1)
					} else {
						t1.Backward(gradForT1)
					}
				}
			}

			if t2.RequiresGrad {
				t1T_for_grad, err_t1t := Transpose(t1)
				if err_t1t != nil {
					fmt.Printf("Warning: MatMul backward failed to transpose t1: %v\n", err_t1t)
				} else {
					gradForT2, err_gt2 := MatMulTensor(t1T_for_grad, grad) // Recursive call
					if err_gt2 != nil {
						fmt.Printf("Warning: MatMul backward failed to compute grad for t2: %v\n", err_gt2)
					} else {
						t2.Backward(gradForT2)
					}
				}
			}
		}
	}

	return out, nil
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