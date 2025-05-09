package autograd

import (
	"fmt"
	"go-torch/tensor"
)

// backward performs the backward pass starting from the root tensor.
// it computes gradients for all tensors in the computation graph that lead to the root
// and have RequiresGrad set to true.
// it uses a topological sort of the graph defined by tensor.Tensor.Parents.

// currently, the main.go uses the backward pass from tensor/tensor.go and works fine
// i implemented this to rewrite in future and to experiment some things 
// TODO: rewrite tensor/tensor.go and implement the autograd functionality here. add new ideas. 

func Backward(root *tensor.Tensor) {
	if !root.RequiresGrad {
		return
	}

	if root.Grad == nil {
		ones, err := tensor.OnesLike(root)
		if err != nil {
			panic(fmt.Sprintf("Error creating initial gradient for root tensor: %v", err))
		}
		ones.RequiresGrad = false
		root.Grad = ones
	} else {
        root.Grad.RequiresGrad = false
    }

	visited := make(map[*tensor.Tensor]bool)
	var topo []*tensor.Tensor

	var dfs func(*tensor.Tensor)
	dfs = func(t *tensor.Tensor) {
		if t == nil || visited[t] {
			return
		}
		visited[t] = true
		if t.RequiresGrad {
			for _, parentTensor := range t.Parents {
				dfs(parentTensor)
			}
		}
		topo = append(topo, t)
	}

	dfs(root)

	for i := len(topo) - 1; i >= 0; i-- {
		currentTensor := topo[i]
		if currentTensor.BackwardFunc != nil {
			if currentTensor.Grad == nil {
				// this might indicate an issue if an intermediate node that should have
				// received a gradient didn't. Or it's a leaf node that is not the root
				// and has RequiresGrad but no consumers yet in this specific backward pass.
				// for now, if Grad is nil, calling BackwardFunc would likely be problematic.
				continue
			}
			currentTensor.BackwardFunc(currentTensor.Grad)
		}
	}
}