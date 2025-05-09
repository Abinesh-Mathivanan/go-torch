package tensor

import (
	"fmt"
)


// NOTE: Most of the functions are self-explanatory, doesn't need much comments / explaantion (except the auto-grad part)


// simple Tensor struct
type Tensor struct {
	shape         []int
	data          []float64
	Grad          *Tensor
	RequiresGrad  bool
	Parents       []*Tensor
	Operation     string
	BackwardFunc  func(*Tensor)
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
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("shape %v contains non-positive dimension", shape)
		}
		total *= dim
	}
	if len(data) > 0 && total != len(data) {
		return nil, fmt.Errorf("shape %v implies %d elements but data has length %d", shape, total, len(data))
	}
	if len(data) == 0 && total > 0 {
		data = make([]float64, total)
	}

	return &Tensor{
		shape:        append([]int{}, shape...),
		data:         append([]float64{}, data...),
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
		data:          clonedData,
		shape:         clonedShape,
		RequiresGrad:  t.RequiresGrad,
		Grad:          nil,
		Parents:       nil,
		Operation:     "",
		BackwardFunc:  nil,
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
		if numElements < 0 { numElements = 0} 
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



// tranposes the last two dimensions of a tensor, currently supports 2D [M, N] -> [N, M].
// for higher dimensions, it would typically transpose the last two.
// TODO: implement a robust transpose system if necessary 
func Transpose(t *Tensor) (*Tensor, error) {
	shape := t.GetShape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("transpose requires a tensor with at least 2 dimensions, got %v", shape)
	}

	newShape := append([]int{}, shape...)
	lastDimIdx := len(newShape) - 1
	secondLastDimIdx := len(newShape) - 2
	newShape[lastDimIdx], newShape[secondLastDimIdx] = newShape[secondLastDimIdx], newShape[lastDimIdx]

	originalNumel := Numel(t)
	newNumel := Numel(&Tensor{shape: newShape}) 
	if originalNumel != newNumel {
		return nil, fmt.Errorf("transpose error: element count mismatch %d != %d", originalNumel, newNumel)
	}

	outData := make([]float64, originalNumel)
	tData := t.GetData()


	// calculate strides for original and transposed tensors
	// stride for original [..., M, N] is [..., N, 1]
	originalStrides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		originalStrides[i] = stride
		stride *= shape[i]
	}


	// stride for transposed [..., N, M] is [..., M, 1]
	transposedStrides := make([]int, len(newShape))
	stride = 1
	for i := len(newShape) - 1; i >= 0; i-- {
		transposedStrides[i] = stride
		stride *= newShape[i]
	}


	// Map original flat index to multi-dimensional index, then to transposed multi-dimensional index, then to transposed flat index.
	// This is complex for generic N-D. Let's simplify for 2D [M, N] -> [N, M]
	if len(shape) == 2 {
		M, N := shape[0], shape[1]
		// Access element at original index (row, col) -> flat index row*N + col
		// Map to transposed index (col, row) -> flat index col*M + row
		for r := 0; r < M; r++ {
			for c := 0; c < N; c++ {
				originalFlatIndex := r*N + c
				transposedFlatIndex := c*M + r
				outData[transposedFlatIndex] = tData[originalFlatIndex]
			}
		}
	} else {
		// TODO: as i mentioned before, a proper N-D transpose would involve mapping linear index back to N-D index.
		return nil, fmt.Errorf("transpose only supports 2D tensors currently, got %v", shape)
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
				// grad(transpose) = transpose(grad)
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


// MatMulTensor performs matrix multiplication between two tensors.
// currently assumes t1 is [..., M, K] and t2 is [..., K, N] and performs
// broadcasted matrix multiplication on the last two dimensions.
// For the Linear layer context, this expects input [B, I] and weight [I, O] (batch dimension B is optional)

// TODO: robust broadcasting technique is needed 
func MatMulTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	shape1 := t1.GetShape()
	shape2 := t2.GetShape()

	if len(shape1) < 2 || len(shape2) < 2 {
        return nil, fmt.Errorf("matmul requires tensors with at least 2 dimensions, got %v and %v", shape1, shape2)
    }

	// assume t1 is [..., M, K] and t2 is [..., K, N].
	// core multiplication is on the last two dimensions.
	// dimensions before the last two must be broadcastable or match.
	// For Linear layer input [B, I] and weight [I, O]:
	// t1: [B, I], t2: [I, O]
	// M=B, K=I, N=O. Inner dimensions match: I.
	// result shape: [B, O]. Batch dimension B must match or be broadcastable.
	// our current tensor structure doesn't explicitly handle batching,
	// so let's assume t1 is [Batch*M, K] and t2 is [K, N] or we only support 2D [M, K] @ [K, N].
	// since linear layer applies batching, t1 is [B, K] and t2 is [K, N].

	// Input: t1 shape [B, K], t2 shape [K, N]
	
	k1 := shape1[len(shape1)-1] 
	k2 := shape2[len(shape2)-2] 
	if k1 != k2 {
		return nil, fmt.Errorf("matmul incompatible shapes: inner dimensions mismatch %v and %v (%d != %d)", shape1, shape2, k1, k2)
	}

	// Output shape: [B, N]
	
	// leading dimensions of t1 and t2 (excluding the last two) must match or be broadcastable.
	// for input [B, I] and weight [I, O], t1 [B, K], t2 [K, N], the leading dimensions are [] and [].
	// if t1 was [A, B, K] and t2 [K, N], result is [A, B, N].
	// if t1 was [B, K] and t2 [A, K, N], broadcasting rules are more complex.
	// let's strictly require t1 to be [B, K] and t2 to be [K, N] for now.
	if len(shape1) != 2 || len(shape2) != 2 {
         return nil, fmt.Errorf("matmul only supports 2D tensors ([B, K] @ [K, N]) currently, got %v and %v", shape1, shape2)
    }
	
	// rename variables to match standard matmul notation [M, K] @ [K, N] -> [M, N]
	// input: [M, K], weight: [K, N]
	
	M := shape1[0] // batch size
	K := shape1[1] // input dimensions

	// shape2 is weight: [In, Out] -> [K_weight, N_weight]
	K_weight := shape2[0]
	N_weight := shape2[1] 

	if K != K_weight {
		return nil, fmt.Errorf("matmul incompatible shapes: inner dimensions mismatch %v and %v (%d != %d)", shape1, shape2, K, K_weight)
	}
	N := N_weight 

	outShape := []int{M, N} // [batch, output]
	outNumel := M * N
	outData := make([]float64, outNumel)

	t1Data := t1.GetData()
	t2Data := t2.GetData()

	// performs matrix multiplication C[i][j] = sum_k(A[i][k] * B[k][j])
	// A is t1 ([M, K]), B is t2 ([K, N])
	// Result C is outData ([M, N])
	for i := 0; i < M; i++ { 
		for j := 0; j < N; j++ { 
			sum := 0.0
			for k_idx := 0; k_idx < K; k_idx++ { 

				t1Val := t1Data[i*K + k_idx]
				t2Val := t2Data[k_idx*N + j]

				sum += t1Val * t2Val
			}
			outData[i*N + j] = sum
		}
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
			// chain rule: dL/dX = dL/dO @ W.T
			// chain rule: dL/dW = X.T @ dL/dO
			// O is the output of matmul, X is t1, W is t2, dL/dO is grad

			gradShape := grad.GetShape()
             if len(gradShape) != 2 || gradShape[0] != M || gradShape[1] != N {
                fmt.Printf("Warning: MatMul backward received incorrect gradient shape %v, expected %v\n", gradShape, outShape)
                return 
            }


			if t1.RequiresGrad {
				// compute dL/dX = dL/dO @ W.T
				// grad shape [M, N], t2 shape [K, N]
				// W.T shape [N, K] (Transpose of t2)
				// dL/dX shape [M, K] (Same as t1)
				t2Transposed, err := Transpose(t2)
				if err != nil {
					fmt.Printf("Warning: Failed to transpose t2 in MatMul backward: %v\n", err)
				} else {
					gradForT1, err := MatMulTensor(grad, t2Transposed) // MatMul: grad [M, N] @ t2Transposed [N, K] -> result [M, K]
					if err != nil {
						fmt.Printf("Warning: Failed to compute grad for t1 in MatMul backward: %v\n", err)
					} else {
						t1.Backward(gradForT1)
					}
				}
			}

			if t2.RequiresGrad {
				// compute dL/dW = X.T @ dL/dO
				// t1 shape [M, K], grad shape [M, N]
				// X.T shape [K, M] (Transpose of t1)
				// dL/dW shape [K, N] (Same as t2)
				t1Transposed, err := Transpose(t1)
				if err != nil {
					fmt.Printf("Warning: Failed to transpose t1 in MatMul backward: %v\n", err)
				} else {
					gradForT2, err := MatMulTensor(t1Transposed, grad) // MatMul: t1Transposed [K, M] @ grad [M, N] -> result [K, N]
					if err != nil {
						fmt.Printf("Warning: Failed to compute grad for t2 in MatMul backward: %v\n", err)
					} else {
						// * if t2 (weight) was involved in multiple forward ops (e.g., shared weights),
						// * its gradient needs to be accumulated. t.Backward() handles this accumulation.
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