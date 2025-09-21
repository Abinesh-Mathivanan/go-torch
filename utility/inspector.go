package utility

import (
	"fmt"
	"go-torch/nn"
	"go-torch/tensor"
	"os"
	"text/tabwriter"
)

// provides utility functions to analyze and log details of a model.
type ModelInspector struct {
	model *nn.Sequential
}

// creates a new inspector for the given sequential model.
func NewModelInspector(model *nn.Sequential) *ModelInspector {
	return &ModelInspector{model: model}
}

// prints summary of the model
func (mi *ModelInspector) Summary() {
	fmt.Println("\n--- Model Summary ---")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Layer (Type)\tParameters\tShape\tParam #")
	fmt.Fprintln(w, "--------------\t----------\t-----\t-------")

	for _, layer := range mi.model.Layers() {
		params := layer.Parameters()
		layerName := layer.Name()

		if len(params) == 0 {
			fmt.Fprintf(w, "%s\t-\t-\t0\n", layerName)
			continue
		}

		names := []string{"Weight", "Bias"}
		for i, p := range params {
			paramName := names[i]
			shape := p.GetShape()
			numel := tensor.Numel(p)

			layerDisplayName := layerName
			if i > 0 {
				layerDisplayName = "" 
			}
			fmt.Fprintf(w, "%s\t%s\t%v\t%d\n", layerDisplayName, paramName, shape, numel)
		}
	}

	w.Flush() 

	total, trainable := mi.CountParameters()

	fmt.Println("----------------------------------")
	fmt.Printf("Total Parameters: %d\n", total)
	fmt.Printf("Trainable Parameters: %d\n", trainable)
	fmt.Println("----------------------------------")
}


// parameter counts for the model.
func (mi *ModelInspector) CountParameters() (total int64, trainable int64) {
	for _, layer := range mi.model.Layers() {
		for _, p := range layer.Parameters() {
			numel := int64(tensor.Numel(p))
			total += numel
			if p.RequiresGrad {
				trainable += numel
			}
		}
	}
	return total, trainable
}