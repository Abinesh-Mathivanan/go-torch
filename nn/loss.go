package nn

import (
	"fmt"
	"math"
	"go-torch/tensor"
)



// computes the Cross-Entropy Loss between logits and target labels.
// it assumes logits is a tensor that can be treated as a flattened 2D structure
// [batch_size, num_classes], where the last dimension contains class scores.
// targets are expected to be class indices (0-indexed).
func CrossEntropyLoss(logits *tensor.Tensor, targets []int) (*tensor.Tensor, error) {
	
	logitsSize := tensor.Numel(logits)
	batchSize := len(targets)

	if batchSize == 0 {
		// loss, gradient = 0, 0
		zeroLossTensor, err := tensor.NewTensor([]int{1}, []float64{0.0})
		if err != nil {
			return nil, fmt.Errorf("cross_entropy_loss: failed to create zero loss tensor for empty batch: %w", err)
		}
		zeroLossTensor.RequiresGrad = false
		return zeroLossTensor, nil 
	}

	// Infer numClasses from logits size and batch size.
	if logitsSize%batchSize != 0 {
		return nil, fmt.Errorf("cross_entropy_loss: logits size (%d) is not divisible by batch size (%d)", logitsSize, batchSize)
	}
	numClasses := logitsSize / batchSize

	if numClasses == 0 {
		return nil, fmt.Errorf("cross_entropy_loss: inferred number of classes is zero")
	}

	logitsData := logits.GetData()
	probsData := make([]float64, logitsSize) // store probabilities flattened like logits
	lossSum := 0.0


	// batch-wise softmax and loss is computed here
	for i := 0; i < batchSize; i++ {
		startIdx := i * numClasses
		endIdx := startIdx + numClasses
		itemLogits := logitsData[startIdx:endIdx] 
		itemProbs := probsData[startIdx:endIdx]   

		// apply Softmax to itemLogits for numerical stability (log-sum-exp)
		maxv := itemLogits[0]
		for _, v := range itemLogits {
			if v > maxv {
				maxv = v
			}
		}

		// math.Exp(v - maxv) is used stable exp calculation. also used in nn/linear.go
		var sumExp float64
		for k, v := range itemLogits {
			expv := math.Exp(v - maxv)
			itemProbs[k] = expv        
			sumExp += expv
		}

		if sumExp == 0 {
			// This can happen if all logits for an item are extremely negative.
			// log(0) is -Inf. If sum is zero, probs will be NaN or Inf.
			return nil, fmt.Errorf("cross_entropy_loss: sum of exponentiated logits is zero for batch item %d, cannot normalize", i)
		}

		// probabilities are normalized
		targetIndex := targets[i]
		if targetIndex < 0 || targetIndex >= numClasses {
			return nil, fmt.Errorf("cross_entropy_loss: target index %d out of bounds for batch item %d with %d classes", targetIndex, i, numClasses)
		}

		for k := range itemProbs {
			itemProbs[k] /= sumExp
		}

		targetProb := itemProbs[targetIndex]
		// add a small epsilon to prevent log(0) which is -Inf.
		// using Max(targetProb, 1e-10) is another option, but adding epsilon is also common.
		// i think the mathematically more robust way is to compute LogSumExp and combine it with the target logit.
        if targetProb <= 0 { 
            targetProb = 1e-10 
        }

		lossSum -= math.Log(targetProb)
	}

	meanLossValue := lossSum / float64(batchSize)

	lossTensor, err := tensor.NewTensor([]int{1}, []float64{meanLossValue})
	if err != nil {
		return nil, fmt.Errorf("cross_entropy_loss: failed to create output tensor for mean loss: %w", err)
	}

	// if lossTensor requires grad, then logits also requires grad
	lossTensor.RequiresGrad = logits.RequiresGrad
	lossTensor.Parents = []*tensor.Tensor{logits}
	lossTensor.Operation = "cross_entropy_loss"

	// called during back-propog if lossTensor requires grad
	if lossTensor.RequiresGrad {
		lossTensor.BackwardFunc = func(grad *tensor.Tensor) {
			// 'grad' is dL_total/dL_this, where L_this is the mean loss here.
			// for the final loss node, 'grad' is 1.0.
			// in this Softmax+CrossEntropy gradient, the 'grad' value isn't used directly,
			// but would scale dL/d(logits) if part of a larger graph. For a final node, scaling is 1.

			if logits.RequiresGrad {
				// gradient of mean loss with respect to logits is (y - target_one_hot) / batchSize.
				gradDataForLogits := make([]float64, logitsSize) 

				// compute gradient (probs - target_one_hot) per batch item
				for i := 0; i < batchSize; i++ {
					startIdx := i * numClasses
					targetIndex := targets[i]

					// The gradient for batch item 'i' is probs_i - target_one_hot_i
					// if confused, target_one_hot_i is a vector with 1 at targetIndex and 0 elsewhere.
					for j := 0; j < numClasses; j++ {
						gradDataForLogits[startIdx+j] = probsData[startIdx+j]
						if j == targetIndex {
							gradDataForLogits[startIdx+j] -= 1.0 
						}
					}
				}

				// scale the gradient by 1/batchSize 
				scale := 1.0 / float64(batchSize)
				for i := range gradDataForLogits {
					gradDataForLogits[i] *= scale
				}

				// gradient tensor must have the same shape as the parent (logits), else err.
				gradTensorForLogits, err := tensor.NewTensor(logits.GetShape(), gradDataForLogits)
				if err != nil {
					fmt.Printf("Warning: Failed to create gradient tensor for logits in CrossEntropyLoss backward: %v\n", err)
					return 
				}

				// Call backward on the parent (logits) with the computed gradient tensor
				logits.Backward(gradTensorForLogits)
			}
		}
	}

	return lossTensor, nil
}