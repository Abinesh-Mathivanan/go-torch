package nn

import (
	"fmt"
	"go-torch/tensor"
	"math/rand"
)

// Embedding is a lookup table mapping integer token ids to dense vectors.
// Weight has shape [vocabSize, embedDim]; Forward looks up one row per id.
type Embedding struct {
	Weight    *tensor.Tensor // [vocabSize, embedDim]
	vocabSize int
	embedDim  int
}

// NewEmbedding creates a new Embedding layer with small random initialization (uniform in [-0.1, 0.1], 
// a common default for embedding tables - large initial embedding magnitudes tend to dominate whatever
// they're summed/concatenated with downstream).
func NewEmbedding(vocabSize, embedDim int) (*Embedding, error) {
	if vocabSize <= 0 || embedDim <= 0 {
		return nil, fmt.Errorf("embedding: vocabSize and embedDim must be positive, got %d, %d", vocabSize, embedDim)
	}

	weightData := make([]float64, vocabSize*embedDim)
	for i := range weightData {
		weightData[i] = (2*rand.Float64() - 1) * 0.1
	}
	weight, err := tensor.NewTensor([]int{vocabSize, embedDim}, weightData)
	if err != nil {
		return nil, fmt.Errorf("embedding: failed to create weight tensor: %w", err)
	}
	weight.RequiresGrad = true

	return &Embedding{
		Weight:    weight,
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}, nil
}

// Forward looks up the embedding vector for each id in ids (a flat slice -
// for a [batch, seqLen] token grid, flatten row-major and Reshape the
// result to [batch, seqLen, embedDim] afterward). 

// Returns a tensor of shape [len(ids), embedDim].
func (e *Embedding) Forward(ids []int) (*tensor.Tensor, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("embedding: ids must be non-empty")
	}

	weightData := e.Weight.GetData()
	outData := make([]float64, len(ids)*e.embedDim)

	for i, id := range ids {
		if id < 0 || id >= e.vocabSize {
			return nil, fmt.Errorf("embedding: id %d at position %d out of range [0, %d)", id, i, e.vocabSize)
		}
		copy(outData[i*e.embedDim:(i+1)*e.embedDim], weightData[id*e.embedDim:(id+1)*e.embedDim])
	}

	out, err := tensor.NewTensor([]int{len(ids), e.embedDim}, outData)
	if err != nil {
		return nil, fmt.Errorf("embedding: failed to create output tensor: %w", err)
	}

	if e.Weight.RequiresGrad {
		out.RequiresGrad = true
		out.Parents = []*tensor.Tensor{e.Weight}
		out.Operation = "embedding"
		out.BackwardFunc = func(grad *tensor.Tensor) {
			if !e.Weight.RequiresGrad {
				return
			}
			if e.Weight.Grad == nil {
				e.Weight.ZeroGrad()
			}
			weightGradData := e.Weight.Grad.GetData()
			gradData := grad.GetData()

			for i, id := range ids {
				destStart := id * e.embedDim
				srcStart := i * e.embedDim
				for j := 0; j < e.embedDim; j++ {
					weightGradData[destStart+j] += gradData[srcStart+j]
				}
			}
		}
	}

	return out, nil
}

// Parameters returns the embedding table so it can be registered with an optimizer.
func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.Weight}
}

// NamedParameters returns this layer's parameters keyed by name, prefixed.
func (e *Embedding) NamedParameters(prefix string) map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		prefix + ".weight": e.Weight,
	}
}


func (e *Embedding) ZeroGrad() {
	e.Weight.ZeroGrad()
}

func (e *Embedding) Name() string {
	return "Embedding"
}