//go run ./verify/safetensors_check
package main

import (
	"fmt"
	"go-torch/safetensors"
	"go-torch/tensor"
	"math"
	"os"
)

func mustTensor(shape []int, data []float64) *tensor.Tensor {
	t, err := tensor.NewTensor(shape, data)
	if err != nil {
		panic(err)
	}
	return t
}

func roundTripCheck() bool {
	fmt.Println("=== safetensors round-trip check ===")

	original := map[string]*tensor.Tensor{
		"conv1.weight": mustTensor([]int{2, 1, 3, 3}, []float64{
			0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9,
			-1.1, 1.2, -1.3, 1.4, -1.5, 1.6, -1.7, 1.8, -1.9,
		}),
		"conv1.bias": mustTensor([]int{2}, []float64{0.01, -0.02}),
		"edge_cases": mustTensor([]int{6}, []float64{
			0.0, math.Copysign(0, -1), 1e-300, 1e300, 1.0 / 3.0, math.Pi,
		}),
		"bn.running_mean": mustTensor([]int{4}, []float64{0, 0, 0, 0}),
	}
	metadata := map[string]string{"format": "go-torch-test-v1", "note": "round-trip check"}

	path := "/tmp/safetensors_roundtrip_check.safetensors"
	defer os.Remove(path)

	if err := safetensors.Save(path, original, metadata); err != nil {
		fmt.Printf("  FAIL: Save returned error: %v\n", err)
		return false
	}

	loaded, loadedMeta, err := safetensors.Load(path)
	if err != nil {
		fmt.Printf("  FAIL: Load returned error: %v\n", err)
		return false
	}

	ok := true

	if len(loaded) != len(original) {
		fmt.Printf("  FAIL: loaded %d tensors, expected %d\n", len(loaded), len(original))
		ok = false
	}

	for name, orig := range original {
		got, present := loaded[name]
		if !present {
			fmt.Printf("  FAIL: %q missing after load\n", name)
			ok = false
			continue
		}

		if fmt.Sprint(got.GetShape()) != fmt.Sprint(orig.GetShape()) {
			fmt.Printf("  FAIL: %q shape mismatch: got %v, want %v\n", name, got.GetShape(), orig.GetShape())
			ok = false
			continue
		}

		origData, gotData := orig.GetData(), got.GetData()
		exact := true
		for i := range origData {
			// bit-exact comparison, not float equality-with-tolerance
			if math.Float64bits(origData[i]) != math.Float64bits(gotData[i]) {
				exact = false
				fmt.Printf("  FAIL: %q element %d: got %v (bits %x), want %v (bits %x)\n",
					name, i, gotData[i], math.Float64bits(gotData[i]), origData[i], math.Float64bits(origData[i]))
			}
		}
		if exact {
			fmt.Printf("  PASS: %q (%d elements, bit-exact)\n", name, len(origData))
		} else {
			ok = false
		}
	}

	if loadedMeta["format"] != metadata["format"] || loadedMeta["note"] != metadata["note"] {
		fmt.Printf("  FAIL: metadata mismatch: got %v, want %v\n", loadedMeta, metadata)
		ok = false
	} else {
		fmt.Println("  PASS: metadata round-tripped correctly")
	}

	return ok
}

func errorPathChecks() bool {
	fmt.Println("\n=== safetensors error-path checks ===")
	ok := true

	// A file that's just a few random bytes - not even a valid header (length - should error cleanly, not panic.)
	truncatedPath := "/tmp/safetensors_truncated_check.safetensors"
	defer os.Remove(truncatedPath)
	if err := os.WriteFile(truncatedPath, []byte{1, 2, 3}, 0644); err != nil {
		panic(err)
	}
	if _, _, err := safetensors.Load(truncatedPath); err == nil {
		fmt.Println("  FAIL: Load on a truncated file returned no error")
		ok = false
	} else {
		fmt.Printf("  PASS: truncated file rejected cleanly: %v\n", err)
	}

	// A header length field claiming an implausibly large header should be
	// rejected before any large allocation is attempted.
	hugeHeaderPath := "/tmp/safetensors_huge_header_check.safetensors"
	defer os.Remove(hugeHeaderPath)
	hugeLenBytes := make([]byte, 8)
	for i := range hugeLenBytes {
		hugeLenBytes[i] = 0xFF // maximal uint64 header length claim
	}
	if err := os.WriteFile(hugeHeaderPath, hugeLenBytes, 0644); err != nil {
		panic(err)
	}
	if _, _, err := safetensors.Load(hugeHeaderPath); err == nil {
		fmt.Println("  FAIL: Load with an implausible header length returned no error")
		ok = false
	} else {
		fmt.Printf("  PASS: implausible header length rejected cleanly: %v\n", err)
	}

	// Saving an empty tensor map should be rejected rather than silently
	// producing a useless/empty checkpoint file.
	if err := safetensors.Save("/tmp/safetensors_empty_check.safetensors", map[string]*tensor.Tensor{}, nil); err == nil {
		fmt.Println("  FAIL: Save with an empty tensor map returned no error")
		ok = false
		os.Remove("/tmp/safetensors_empty_check.safetensors")
	} else {
		fmt.Printf("  PASS: empty tensor map rejected cleanly: %v\n", err)
	}

	return ok
}

func main() {
	ok1 := roundTripCheck()
	ok2 := errorPathChecks()

	fmt.Println()
	if ok1 && ok2 {
		fmt.Println("All safetensors checks passed :)")
	} else {
		fmt.Println("Safetensors checks failed :(")
		os.Exit(1)
	}
}