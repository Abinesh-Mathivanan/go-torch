package tensor

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)


// slice pool - reduce GC from 1900 to 363
var pool = sync.Pool{
	New: func() interface{} {
		return nil
	},
}

func getSlice(size int) []float32 {
	if val := pool.Get(); val != nil {
		s := val.([]float32)
		if cap(s) >= size {
			return s[:size]
		}
	}
	return make([]float32, size)
}

func putSlice(s []float32) {
	pool.Put(s)
}

func init() {
	gob.Register(&Tensor{})
}

type tensorGob struct {
	Shape []int
	Data  []float32
}

func (t *Tensor) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(tensorGob{Shape: t.shape, Data: t.data})
	return buf.Bytes(), err
}

func (t *Tensor) GobDecode(data []byte) error {
	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)
	var tg tensorGob
	if err := decoder.Decode(&tg); err != nil {
		return err
	}
	t.shape = tg.Shape
	t.data = tg.Data
	return nil
}

type Tensor struct {
	shape        []int
	data         []float32
	Grad         *Tensor
	RequiresGrad bool
	Parents      []*Tensor
	Operation    string
	BackwardFunc func(*Tensor)
}

func IsSameSize(a, b *Tensor) bool {
	if len(a.shape) != len(b.shape) {
		return false
	}
	for i := range a.shape {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	return true
}

func NewTensor(shape []int, data []float32) (*Tensor, error) {
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape %v", shape)
		}
		total *= dim
	}

	var finalData []float32
	if len(data) == 0 {
		finalData = getSlice(total)
		for i := range finalData {
			finalData[i] = 0
		}
	} else {
		if total != len(data) {
			return nil, fmt.Errorf("shape/data mismatch")
		}
		finalData = getSlice(total)
		copy(finalData, data)
	}

	return &Tensor{
		shape: append([]int{}, shape...),
		data:  finalData,
	}, nil
}

// AddTensor with BCE (Bounds Check Elimination) optimization
func AddTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	if !IsSameSize(t1, t2) {
		return nil, fmt.Errorf("shape mismatch")
	}

	n := len(t1.data)
	outData := getSlice(n)
	d1, d2 := t1.data, t2.data

	// check bounds once before the loop
	_ = d1[n-1]
	_ = d2[n-1]
	_ = outData[n-1]
	for i := 0; i < n; i++ {
		outData[i] = d1[i] + d2[i]
	}

	out, _ := NewTensor(t1.shape, outData)
	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "add"
		out.BackwardFunc = func(grad *Tensor) {
			if t1.RequiresGrad { t1.Backward(grad) }
			if t2.RequiresGrad { t2.Backward(grad) }
		}
	}
	return out, nil
}

// MulTensor (Element-wise)
func MulTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	if !IsSameSize(t1, t2) {
		return nil, fmt.Errorf("shape mismatch")
	}

	n := len(t1.data)
	outData := getSlice(n)
	d1, d2 := t1.data, t2.data

	_ = d1[n-1]
	_ = d2[n-1]
	_ = outData[n-1]
	for i := 0; i < n; i++ {
		outData[i] = d1[i] * d2[i]
	}

	out, _ := NewTensor(t1.shape, outData)
	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "mul"
		out.BackwardFunc = func(grad *Tensor) {
			if t1.RequiresGrad {
				g1, _ := MulTensor(grad, t2)
				t1.Backward(g1)
			}
			if t2.RequiresGrad {
				g2, _ := MulTensor(grad, t1)
				t2.Backward(g2)
			}
		}
	}
	return out, nil
}

// MatMulTensor optimized with blas32 
func MatMulTensor(t1 *Tensor, t2 *Tensor) (*Tensor, error) {
	s1, s2 := t1.shape, t2.shape
	if len(s1) != 2 || len(s2) != 2 || s1[1] != s2[0] {
		return nil, fmt.Errorf("matmul shape mismatch: %v, %v", s1, s2)
	}

	M, K, N := s1[0], s1[1], s2[1]
	outData := getSlice(M * N)

	// blas32.Gemm for high performance float32 matmul
	// C = alpha*A*B + beta*C
	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0,
		blas32.General{Rows: M, Cols: K, Stride: K, Data: t1.data},
		blas32.General{Rows: K, Cols: N, Stride: N, Data: t2.data},
		0.0,
		blas32.General{Rows: M, Cols: N, Stride: N, Data: outData},
	)

	out, _ := NewTensor([]int{M, N}, outData)
	if t1.RequiresGrad || t2.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t1, t2}
		out.Operation = "matmul"
		out.BackwardFunc = func(grad *Tensor) {
			if t1.RequiresGrad {
				// grad [M,N] @ t2.T [N,K]
				gData1 := getSlice(M * K)
				blas32.Gemm(blas.NoTrans, blas.Trans, 1.0,
					blas32.General{Rows: M, Cols: N, Stride: N, Data: grad.data},
					blas32.General{Rows: K, Cols: N, Stride: N, Data: t2.data},
					0.0,
					blas32.General{Rows: M, Cols: K, Stride: K, Data: gData1},
				)
				gt1, _ := NewTensor([]int{M, K}, gData1)
				t1.Backward(gt1)
			}
			if t2.RequiresGrad {
				// t1.T [K,M] @ grad [M,N]
				gData2 := getSlice(K * N)
				blas32.Gemm(blas.Trans, blas.NoTrans, 1.0,
					blas32.General{Rows: M, Cols: K, Stride: K, Data: t1.data},
					blas32.General{Rows: M, Cols: N, Stride: N, Data: grad.data},
					0.0,
					blas32.General{Rows: K, Cols: N, Stride: N, Data: gData2},
				)
				gt2, _ := NewTensor([]int{K, N}, gData2)
				t2.Backward(gt2)
			}
		}
	}
	return out, nil
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		d := t.Grad.data
		for i := range d { d[i] = 0 }
	} else if t.RequiresGrad {
		t.Grad, _ = NewTensor(t.shape, nil)
	}
}

func (t *Tensor) Backward(grad *Tensor) {
	if !t.RequiresGrad { return }
	if grad == nil {
		if Numel(t) == 1 {
			grad, _ = NewTensor(t.shape, []float32{1.0})
		} else { return }
	}

	if t.Grad == nil {
		t.Grad = CloneTensor(grad)
		t.Grad.RequiresGrad = false
	} else {
		d, gd := t.Grad.data, grad.data
		_ = d[len(d)-1]
		_ = gd[len(d)-1]
		for i := range d {
			d[i] += gd[i]
		}
	}
}

func AddTensorBroadcast(a *Tensor, b *Tensor) (*Tensor, error) {
	aShape, bShape := a.shape, b.shape
	outData := getSlice(Numel(a))
	aData, bData := a.data, b.data

	// Implementation for 4D (Conv) and 2D (Linear) Bias Addition
	if len(aShape) == 4 && len(bShape) == 1 {
		B, C, HW := aShape[0], aShape[1], aShape[2]*aShape[3]
		for b_idx := 0; b_idx < B; b_idx++ {
			for c_idx := 0; c_idx < C; c_idx++ {
				bias := bData[c_idx]
				offset := b_idx*C*HW + c_idx*HW
				for i := 0; i < HW; i++ {
					outData[offset+i] = aData[offset+i] + bias
				}
			}
		}
	} else if len(aShape) == 2 && len(bShape) == 1 {
		B, F := aShape[0], aShape[1]
		for r := 0; r < B; r++ {
			offset := r * F
			for c := 0; c < F; c++ {
				outData[offset+c] = aData[offset+c] + bData[c]
			}
		}
	}

	out, _ := NewTensor(aShape, outData)
	if a.RequiresGrad || b.RequiresGrad {
		out.RequiresGrad, out.Parents, out.Operation = true, []*Tensor{a, b}, "add_broadcast"
		out.BackwardFunc = func(grad *Tensor) {
			if a.RequiresGrad { a.Backward(grad) }
			if b.RequiresGrad {
				b.ZeroGrad()
				gD, bGD := grad.data, b.Grad.data
				if len(aShape) == 4 {
					B, C, HW := aShape[0], aShape[1], aShape[2]*aShape[3]
					for i := 0; i < B; i++ {
						for j := 0; j < C; j++ {
							off, sum := i*C*HW+j*HW, float32(0)
							for k := 0; k < HW; k++ { sum += gD[off+k] }
							bGD[j] += sum
						}
					}
				} else {
					B, F := aShape[0], aShape[1]
					for c := 0; c < F; c++ {
						sum := float32(0)
						for r := 0; r < B; r++ { sum += gD[r*F+c] }
						bGD[c] += sum
					}
				}
			}
		}
	}
	return out, nil
}

// --- Basic Utilities ---

func Numel(t *Tensor) int {
	res := 1
	for _, v := range t.shape { res *= v }
	return res
}

func (t *Tensor) GetData() []float32 { return t.data }
func (t *Tensor) GetShape() []int    { return t.shape }

func CloneTensor(t *Tensor) *Tensor {
	d := getSlice(len(t.data))
	copy(d, t.data)
	return &Tensor{shape: append([]int{}, t.shape...), data: d, RequiresGrad: t.RequiresGrad}
}

func (t *Tensor) Pow(scalar float32) *Tensor {
	outData := getSlice(len(t.data))
	for i, v := range t.data {
		outData[i] = float32(math.Pow(float64(v), float64(scalar)))
	}
	out, _ := NewTensor(t.shape, outData)
	return out
}

func Sub(t1, t2 *Tensor) *Tensor {
	outData := getSlice(Numel(t1))
	d1, d2 := t1.data, t2.data
	if IsSameSize(t1, t2) {
		for i := range d1 { outData[i] = d1[i] - d2[i] }
	} else if len(t1.shape) == 2 && t2.shape[0] == 1 { // Broadcast sub
		for i := 0; i < t1.shape[0]; i++ {
			for j := 0; j < t1.shape[1]; j++ {
				outData[i*t1.shape[1]+j] = d1[i*t1.shape[1]+j] - d2[j]
			}
		}
	}
	out, _ := NewTensor(t1.shape, outData)
	return out
}

func (t *Tensor) MulScalar(s float32) *Tensor {
	d := getSlice(len(t.data))
	for i, v := range t.data { d[i] = v * s }
	out, _ := NewTensor(t.shape, d)
	return out
}

func Sum(t *Tensor, axis int, keepDims bool) *Tensor {
	s := t.shape
	var nS []int
	var d []float32
	if axis == 0 {
		d = getSlice(s[1])
		for j := 0; j < s[1]; j++ {
			var sum float32
			for i := 0; i < s[0]; i++ { sum += t.data[i*s[1]+j] }
			d[j] = sum
		}
		if keepDims { nS = []int{1, s[1]} } else { nS = []int{s[1]} }
	}
	out, _ := NewTensor(nS, d)
	return out
}

func Mean(t *Tensor, axis int, keepDims bool) *Tensor {
	return Sum(t, axis, keepDims).MulScalar(1.0 / float32(t.shape[axis]))
}

func Var(t *Tensor, axis int, keepDims bool) *Tensor {
	mean := Mean(t, axis, true)
	diff := Sub(t, mean)
	return Mean(diff.Pow(2), axis, keepDims)
}

func Reshape(t *Tensor, newShape []int) (*Tensor, error) {
	if Numel(t) != 1 {
		total := 1
		for _, v := range newShape { total *= v }
		if total != Numel(t) { return nil, fmt.Errorf("reshape mismatch") }
	}
	out, _ := NewTensor(newShape, t.data)
	if t.RequiresGrad {
		out.RequiresGrad, out.Parents, out.Operation = true, []*Tensor{t}, "reshape"
		out.BackwardFunc = func(grad *Tensor) {
			g, _ := NewTensor(t.shape, grad.data)
			t.Backward(g)
		}
	}
	return out, nil
}

func ArgMax(t *Tensor) int {
	idx, max := 0, t.data[0]
	for i, v := range t.data {
		if v > max { max, idx = v, i }
	}
	return idx
}

func (t *Tensor) Slice(index int) (*Tensor, error) {
	s := t.shape
	numel := s[1] * s[2] * s[3]
	newData := getSlice(numel)
	copy(newData, t.data[index*numel:(index+1)*numel])
	return NewTensor([]int{1, s[1], s[2], s[3]}, newData)
}

// OnesLike returns a tensor of the same shape as t but filled with 1.0
func OnesLike(t *Tensor) (*Tensor, error) {
	size := Numel(t)
	data := getSlice(size)
	for i := 0; i < size; i++ {
		data[i] = 1.0
	}
	return NewTensor(t.shape, data)
}

// AddScalar adds a constant value to every element
func (t *Tensor) AddScalar(scalar float32) *Tensor {
	n := len(t.data)
	outData := getSlice(n)
	d := t.data
	_ = d[n-1]
	_ = outData[n-1]
	for i := 0; i < n; i++ {
		outData[i] = d[i] + scalar
	}
	out, _ := NewTensor(t.shape, outData)
	return out
}

func Transpose(t *Tensor) (*Tensor, error) {
	shape := t.shape
	if len(shape) != 2 {
		return nil, fmt.Errorf("transpose currently only supports 2D tensors, got %v", shape)
	}

	newShape := []int{shape[1], shape[0]}
	n := len(t.data)
	outData := getSlice(n)
	
	M, N := shape[0], shape[1]
	tData := t.data
	
	for r := 0; r < M; r++ {
		for c := 0; c < N; c++ {
			// Row-major to Column-major mapping
			outData[c*M+r] = tData[r*N+c]
		}
	}

	out, _ := NewTensor(newShape, outData)
	if t.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*Tensor{t}
		out.Operation = "transpose"
		out.BackwardFunc = func(grad *Tensor) {
			transposedGrad, _ := Transpose(grad)
			t.Backward(transposedGrad)
		}
	}
	return out, nil
}

func Permute(t *Tensor, axes []int) (*Tensor, error) {
	if len(t.shape) != len(axes) {
		return nil, fmt.Errorf("permute: axes mismatch")
	}
	
	newShape := make([]int, len(t.shape))
	for i, axis := range axes {
		newShape[i] = t.shape[axis]
	}
	
	outData := getSlice(Numel(t))

	// Optimized Fast Path for Conv2D: [1, 0, 2, 3]
	if len(t.shape) == 4 && axes[0] == 1 && axes[1] == 0 {
		C, B, H, W := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
		tData := t.data
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
		return nil, fmt.Errorf("permute: only axes [1, 0, 2, 3] supported for 4D")
	}

	out, _ := NewTensor(newShape, outData)
	if t.RequiresGrad {
		out.RequiresGrad, out.Parents, out.Operation = true, []*Tensor{t}, "permute"
		out.BackwardFunc = func(grad *Tensor) {
			// The inverse of [1,0,2,3] is [1,0,2,3]
			invAxes := []int{1, 0, 2, 3}
			g, _ := Permute(grad, invAxes)
			t.Backward(g)
		}
	}
	return out, nil
}

func PrintTensor(t *Tensor) {
	fmt.Printf("Tensor(shape=%v, data_len=%d, req_grad=%v)\n", t.shape, len(t.data), t.RequiresGrad)
}