// go run ./verify
package main

import (
	"fmt"
	"go-torch/autograd"
	"go-torch/nn"
	"go-torch/tensor"
	"math/rand"
)

const (
	epsilon    = 1e-5
	tolerance  = 1e-4 // relative error tolerance for the check to pass
)

// sumSquares computes a scalar loss L = sum(x_i^2), the simplest loss that
// depends on every element and has a clean, hand-verifiable gradient
// (dL/dx_i = 2*x_i), so it doesn't introduce any risk of ITS OWN backward
// being wrong and confusing the check.
func sumSquares(t *tensor.Tensor) (*tensor.Tensor, error) {
	data := t.GetData()
	sum := 0.0
	for _, v := range data {
		sum += v * v
	}
	out, err := tensor.NewTensor([]int{1}, []float64{sum})
	if err != nil {
		return nil, err
	}
	if t.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{t}
		out.BackwardFunc = func(grad *tensor.Tensor) {
			g := grad.GetData()[0]
			gradData := make([]float64, len(data))
			for i, v := range data {
				gradData[i] = 2 * v * g
			}
			gradTensor, _ := tensor.NewTensor(t.GetShape(), gradData)
			t.Backward(gradTensor)
		}
	}
	return out, nil
}

// forwardLoss recomputes the whole forward pass (no grad tracking needed)
// and returns the scalar loss value - used for the +/- epsilon probes.
func forwardLoss1d(bn *nn.BatchNorm1d, input *tensor.Tensor) float64 {
	out, err := bn.Forward(input)
	if err != nil {
		panic(err)
	}
	sum := 0.0
	for _, v := range out.GetData() {
		sum += v * v
	}
	return sum
}

func forwardLoss2d(bn *nn.BatchNorm2d, input *tensor.Tensor) float64 {
	out, err := bn.Forward(input)
	if err != nil {
		panic(err)
	}
	sum := 0.0
	for _, v := range out.GetData() {
		sum += v * v
	}
	return sum
}

// checkGradient perturbs each element of `data` by +/- eps, measures the
// central-difference derivative via `loss`, and compares it against
// `analytic`. Returns the max relative error found and how many elements
// were checked.
func checkGradient(name string, data []float64, analytic []float64, eps float64, loss func() float64) (maxRelErr float64, n int) {
	if len(data) != len(analytic) {
		panic(fmt.Sprintf("%s: length mismatch data=%d analytic=%d", name, len(data), len(analytic)))
	}
	for i := range data {
		orig := data[i]

		data[i] = orig + eps
		lossPlus := loss()

		data[i] = orig - eps
		lossMinus := loss()

		data[i] = orig // restore

		numeric := (lossPlus - lossMinus) / (2 * eps)
		denom := max(absf(numeric), absf(analytic[i]), 1e-8)
		relErr := absf(numeric-analytic[i]) / denom

		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		n++
	}
	return maxRelErr, n
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func max(vals ...float64) float64 {
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func diagnoseEpsilonScaling(name string, data []float64, analytic []float64, loss func() float64) {
	fmt.Printf("\n--- epsilon scaling diagnostic: %s ---\n", name)
	for _, eps := range []float64{1e-3, 1e-4, 1e-5, 1e-6, 1e-7} {
		maxErr, _ := checkGradient(name, data, analytic, eps, loss)
		fmt.Printf("  eps=%.0e  max relative error = %.3e\n", eps, maxErr)
	}
}

const batchNormEpsilon = 1e-3

func runBatchNorm1dCheck() bool {
	fmt.Println("=== BatchNorm1d gradient check ===")
	rand.Seed(42)

	B, F := 8, 4
	data := make([]float64, B*F)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	input, _ := tensor.NewTensor([]int{B, F}, data)
	input.RequiresGrad = true

	bn, err := nn.NewBatchNorm1d(F, 0.9, 1e-5)
	if err != nil {
		panic(err)
	}
	// give gamma/beta non-trivial values so their gradients are non-degenerate
	for i, v := range []float64{1.5, 0.5, 2.0, -1.0} {
		bn.Weight.GetData()[i] = v
	}
	for i, v := range []float64{0.1, -0.2, 0.3, 0.0} {
		bn.Bias.GetData()[i] = v
	}

	out, err := bn.Forward(input)
	if err != nil {
		panic(err)
	}
	loss, err := sumSquares(out)
	if err != nil {
		panic(err)
	}
	autograd.Backward(loss)

	inputGradCopy := append([]float64{}, input.Grad.GetData()...)
	weightGradCopy := append([]float64{}, bn.Weight.Grad.GetData()...)
	biasGradCopy := append([]float64{}, bn.Bias.Grad.GetData()...)

	ok := true

	maxErr, n := checkGradient("input", input.GetData(), inputGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss1d(bn, input)
	})
	fmt.Printf("  input  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))
	if maxErr > tolerance {
		diagnoseEpsilonScaling("BatchNorm1d input", input.GetData(), inputGradCopy, func() float64 {
			return forwardLoss1d(bn, input)
		})
	}

	maxErr, n = checkGradient("weight (gamma)", bn.Weight.GetData(), weightGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss1d(bn, input)
	})
	fmt.Printf("  gamma  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	maxErr, n = checkGradient("bias (beta)", bn.Bias.GetData(), biasGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss1d(bn, input)
	})
	fmt.Printf("  beta   grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	return ok
}

func runBatchNorm2dCheck() bool {
	fmt.Println("\n=== BatchNorm2d gradient check ===")
	rand.Seed(7)

	B, C, H, W := 2, 3, 4, 4
	data := make([]float64, B*C*H*W)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	input, _ := tensor.NewTensor([]int{B, C, H, W}, data)
	input.RequiresGrad = true

	bn, err := nn.NewBatchNorm2d(C, 0.9, 1e-5)
	if err != nil {
		panic(err)
	}
	for i, v := range []float64{1.2, 0.8, -1.5} {
		bn.Weight.GetData()[i] = v
	}
	for i, v := range []float64{0.05, -0.1, 0.2} {
		bn.Bias.GetData()[i] = v
	}

	out, err := bn.Forward(input)
	if err != nil {
		panic(err)
	}
	loss, err := sumSquares(out)
	if err != nil {
		panic(err)
	}
	autograd.Backward(loss)

	inputGradCopy := append([]float64{}, input.Grad.GetData()...)
	weightGradCopy := append([]float64{}, bn.Weight.Grad.GetData()...)
	biasGradCopy := append([]float64{}, bn.Bias.Grad.GetData()...)

	ok := true

	maxErr, n := checkGradient("input", input.GetData(), inputGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss2d(bn, input)
	})
	fmt.Printf("  input  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))
	if maxErr > tolerance {
		diagnoseEpsilonScaling("BatchNorm2d input", input.GetData(), inputGradCopy, func() float64 {
			return forwardLoss2d(bn, input)
		})
	}

	maxErr, n = checkGradient("weight (gamma)", bn.Weight.GetData(), weightGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss2d(bn, input)
	})
	fmt.Printf("  gamma  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	maxErr, n = checkGradient("bias (beta)", bn.Bias.GetData(), biasGradCopy, batchNormEpsilon, func() float64 {
		return forwardLoss2d(bn, input)
	})
	fmt.Printf("  beta   grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	return ok
}

func verdict(maxErr float64, ok *bool) string {
	if maxErr > tolerance {
		*ok = false
		return "FAIL"
	}
	return "PASS"
}

func runLayerNormCheck() bool {
	fmt.Println("\n=== LayerNorm gradient check ===")
	rand.Seed(99)

	B, F := 6, 5
	data := make([]float64, B*F)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	input, _ := tensor.NewTensor([]int{B, F}, data)
	input.RequiresGrad = true

	ln, err := nn.NewLayerNorm(F, 1e-5)
	if err != nil {
		panic(err)
	}
	for i, v := range []float64{1.3, 0.7, -0.9, 2.0, 0.4} {
		ln.Weight.GetData()[i] = v
	}
	for i, v := range []float64{0.2, -0.1, 0.0, 0.3, -0.2} {
		ln.Bias.GetData()[i] = v
	}

	forwardLoss := func() float64 {
		out, err := ln.Forward(input)
		if err != nil {
			panic(err)
		}
		sum := 0.0
		for _, v := range out.GetData() {
			sum += v * v
		}
		return sum
	}

	out, err := ln.Forward(input)
	if err != nil {
		panic(err)
	}
	loss, err := sumSquares(out)
	if err != nil {
		panic(err)
	}
	autograd.Backward(loss)

	inputGradCopy := append([]float64{}, input.Grad.GetData()...)
	weightGradCopy := append([]float64{}, ln.Weight.Grad.GetData()...)
	biasGradCopy := append([]float64{}, ln.Bias.Grad.GetData()...)

	ok := true

	maxErr, n := checkGradient("input", input.GetData(), inputGradCopy, epsilon, forwardLoss)
	fmt.Printf("  input  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	maxErr, n = checkGradient("weight (gamma)", ln.Weight.GetData(), weightGradCopy, epsilon, forwardLoss)
	fmt.Printf("  gamma  grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	maxErr, n = checkGradient("bias (beta)", ln.Bias.GetData(), biasGradCopy, epsilon, forwardLoss)
	fmt.Printf("  beta   grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))

	return ok
}

func runActivationCheck(name string, fn func(*tensor.Tensor) (*tensor.Tensor, error)) bool {
	fmt.Printf("\n=== %s gradient check ===\n", name)
	rand.Seed(123)

	n := 20
	data := make([]float64, n)
	for i := range data {
		data[i] = rand.NormFloat64() * 2 // spread across +/- a few, exercises both branches of piecewise functions
	}
	input, _ := tensor.NewTensor([]int{n}, data)
	input.RequiresGrad = true

	forwardLoss := func() float64 {
		out, err := fn(input)
		if err != nil {
			panic(err)
		}
		sum := 0.0
		for _, v := range out.GetData() {
			sum += v * v
		}
		return sum
	}

	out, err := fn(input)
	if err != nil {
		panic(err)
	}
	loss, err := sumSquares(out)
	if err != nil {
		panic(err)
	}
	autograd.Backward(loss)

	inputGradCopy := append([]float64{}, input.Grad.GetData()...)

	ok := true
	maxErr, checked := checkGradient(name, input.GetData(), inputGradCopy, epsilon, forwardLoss)
	fmt.Printf("  input grad: checked %d elements, max relative error = %.3e %s\n", checked, maxErr, verdict(maxErr, &ok))
	return ok
}

func runEmbeddingCheck() bool {
	fmt.Println("\n=== Embedding gradient check (repeated-index scatter-add) ===")
	rand.Seed(55)

	vocabSize, embedDim := 6, 4
	emb, err := nn.NewEmbedding(vocabSize, embedDim)
	if err != nil {
		panic(err)
	}
	// deliberately repeat id 2 three times - this is exactly the case that
	// breaks if backward overwrites instead of accumulating into a row.
	ids := []int{2, 0, 2, 4, 2, 1}

	forwardLoss := func() float64 {
		out, err := emb.Forward(ids)
		if err != nil {
			panic(err)
		}
		sum := 0.0
		for _, v := range out.GetData() {
			sum += v * v
		}
		return sum
	}

	out, err := emb.Forward(ids)
	if err != nil {
		panic(err)
	}
	loss, err := sumSquares(out)
	if err != nil {
		panic(err)
	}
	autograd.Backward(loss)

	weightGradCopy := append([]float64{}, emb.Weight.Grad.GetData()...)

	ok := true
	maxErr, n := checkGradient("embedding weight", emb.Weight.GetData(), weightGradCopy, epsilon, forwardLoss)
	fmt.Printf("  weight grad: checked %d elements, max relative error = %.3e %s\n", n, maxErr, verdict(maxErr, &ok))
	return ok
}

func main() {
	ok1 := runBatchNorm1dCheck()
	ok2 := runBatchNorm2dCheck()
	ok3 := runLayerNormCheck()
	ok4 := runActivationCheck("LeakyReLU", func(t *tensor.Tensor) (*tensor.Tensor, error) { return nn.LeakyReLU(t, 0.1) })
	ok5 := runActivationCheck("GELU", nn.GELU)
	ok6 := runActivationCheck("SiLU", nn.SiLU)
	ok7 := runEmbeddingCheck()

	fmt.Println()
	if ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 {
		fmt.Println("All gradient checks passed :)")
	} else {
		fmt.Println("Gradient checks failed: analytic backward does not match numerical estimate :(")
	}
}