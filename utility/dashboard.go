package utility

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	ui "github.com/gizak/termui/v3"
	"github.com/gizak/termui/v3/widgets"
)


// manages the TUI for real-time training monitoring.
type TrainingDashboard struct {
	grid *ui.Grid

	lossPlot     *widgets.Plot
	accuracyPlot *widgets.Plot

	progressGauge *widgets.Gauge
	progressList  *widgets.List
	systemList    *widgets.List
	logParagraph  *widgets.Paragraph

	fullLossData     []float64
	fullAccuracyData []float64
	renderMutex      sync.Mutex
}


func NewTrainingDashboard(learningRate float64, batchSize int, epochs int) *TrainingDashboard {
	if err := ui.Init(); err != nil {
		log.Fatalf("failed to initialize termui: %v", err)
	}

	d := &TrainingDashboard{
		fullLossData:     []float64{0, 0},
		fullAccuracyData: []float64{0, 0},
	}

	d.lossPlot = widgets.NewPlot()
	d.lossPlot.Title = "Full Training Loss"
	d.lossPlot.Data = [][]float64{d.fullLossData} 
	d.lossPlot.LineColors[0] = ui.ColorRed

	d.accuracyPlot = widgets.NewPlot()
	d.accuracyPlot.Title = "Full Validation Accuracy (%)"
	d.accuracyPlot.Data = [][]float64{d.fullAccuracyData}
	d.accuracyPlot.LineColors[0] = ui.ColorGreen

	d.progressGauge = widgets.NewGauge()
	d.progressGauge.Title = "Epoch Progress"
	d.progressGauge.BarColor = ui.ColorBlue
	d.systemList = widgets.NewList()
	d.systemList.Title = "System & Timing"
	d.progressList = widgets.NewList()
	d.progressList.Title = "Training Status"
	hyperParamList := widgets.NewList()
	hyperParamList.Title = "Hyperparameters"
	hyperParamList.Rows = []string{
		fmt.Sprintf("Epochs: %d", epochs),
		fmt.Sprintf("Batch Size: %d", batchSize),
		fmt.Sprintf("Learn Rate: %.4f", learningRate),
	}
	d.logParagraph = widgets.NewParagraph()
	d.logParagraph.Title = "Event Log"

	d.grid = ui.NewGrid()
	termWidth, termHeight := ui.TerminalDimensions()
	d.grid.SetRect(0, 0, termWidth, termHeight)
	d.grid.Set(
		ui.NewRow(0.4, ui.NewCol(0.5, d.lossPlot), ui.NewCol(0.5, d.accuracyPlot)),
		ui.NewRow(0.3, ui.NewCol(0.34, d.progressList), ui.NewCol(0.33, d.systemList), ui.NewCol(0.33, hyperParamList)),
		ui.NewRow(0.3, ui.NewCol(1.0, ui.NewRow(0.4, d.progressGauge), ui.NewRow(0.6, d.logParagraph))),
	)

	return d
}


// downsample averages a slice of data to fit a target width [as i wanted the graph to be restricted within the grid]
func (d *TrainingDashboard) downsample(data []float64, targetWidth int) []float64 {
	if targetWidth <= 0 || len(data) <= targetWidth {
		return data 
	}

	downsampled := make([]float64, targetWidth)
	binSize := float64(len(data)) / float64(targetWidth)

	for i := 0; i < targetWidth; i++ {
		start := int(float64(i) * binSize)
		end := int(float64(i+1) * binSize)
		if end > len(data) {
			end = len(data)
		}

		bin := data[start:end]
		if len(bin) == 0 {
			if i > 0 {
				downsampled[i] = downsampled[i-1] 
			} else {
				downsampled[i] = 0
			}
			continue
		}

		var sum float64
		for _, v := range bin {
			sum += v
		}
		downsampled[i] = sum / float64(len(bin))
	}
	return downsampled
}


// update the dashboard.
func (d *TrainingDashboard) UpdateStats(epoch, totalEpochs, batch, totalBatches int, avgLoss float64, epochStartTime, totalStartTime time.Time) {
	d.renderMutex.Lock()
	defer d.renderMutex.Unlock()

	d.progressList.Rows = []string{
		fmt.Sprintf("Epoch: %d / %d", epoch, totalEpochs),
		fmt.Sprintf("Batch: %d / %d", batch, totalBatches),
		fmt.Sprintf("Avg Loss: %.4f", avgLoss),
	}
	
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	epochElapsed := time.Since(epochStartTime).Round(time.Second)
	totalElapsed := time.Since(totalStartTime).Round(time.Second)
	var eta time.Duration
	if batch > 0 {
		timePerBatch := epochElapsed.Seconds() / float64(batch)
		eta = time.Duration(timePerBatch*float64(totalBatches-batch)) * time.Second
	}
	d.systemList.Rows = []string{
		fmt.Sprintf("Epoch Time: %v", epochElapsed),
		fmt.Sprintf("Total Time: %v", totalElapsed),
		fmt.Sprintf("ETA (Epoch): %v", eta),
		"---",
		fmt.Sprintf("Heap Alloc: %d MiB", memStats.Alloc/1024/1024),
		fmt.Sprintf("Goroutines: %d", runtime.NumGoroutine()),
	}
	d.progressGauge.Percent = int(float64(batch) / float64(totalBatches) * 100)
	
	d.lossPlot.Data[0] = d.downsample(d.fullLossData, d.lossPlot.Inner.Dx())
	
	ui.Render(d.grid)
}


// appends a new loss value to the full history.
func (d *TrainingDashboard) AddLoss(loss float64) {
	d.renderMutex.Lock()
	defer d.renderMutex.Unlock()
	d.fullLossData = append(d.fullLossData, loss)
}


// appends a new accuracy value and triggers a re-render.
func (d *TrainingDashboard) AddAccuracy(accuracy float64) {
	d.renderMutex.Lock()
	defer d.renderMutex.Unlock()

	d.fullAccuracyData = append(d.fullAccuracyData, accuracy)
	d.lossPlot.Data[0] = d.downsample(d.fullLossData, d.lossPlot.Inner.Dx())
	d.accuracyPlot.Data[0] = d.downsample(d.fullAccuracyData, d.accuracyPlot.Inner.Dx())

	ui.Render(d.grid)
}



// prints a message to the event log panel.
func (d *TrainingDashboard) Log(message string) {
	d.renderMutex.Lock()
	defer d.renderMutex.Unlock()
	d.logParagraph.Text = message
	ui.Render(d.grid)
}


// utility functions - close and loop
func (d *TrainingDashboard) Close() { ui.Close() }
func (d *TrainingDashboard) Loop() {
	uiEvents := ui.PollEvents()
	for {
		e := <-uiEvents
		if e.ID == "q" || e.ID == "<C-c>" {
			return
		}
	}
}