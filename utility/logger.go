package utility

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// minimal logging utility for training runs, with a simple in-terminal progress bar and a sparkline of recent loss values. 
type TrainingLogger struct {
	learningRate float64
	batchSize    int
	epochs       int

	mu              sync.Mutex
	lossHistory     []float64
	accuracyHistory []float64
}


func NewTrainingLogger(learningRate float64, batchSize, epochs int) *TrainingLogger {
	fmt.Printf("Training config: epochs=%d batch_size=%d learning_rate=%.4f\n", epochs, batchSize, learningRate)
	return &TrainingLogger{
		learningRate: learningRate,
		batchSize:    batchSize,
		epochs:       epochs,
	}
}


func (l *TrainingLogger) UpdateStats(epoch, totalEpochs, batch, totalBatches int, avgLoss float64, epochStart, totalStart time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()

	percent := float64(batch) / float64(totalBatches) * 100
	elapsed := time.Since(epochStart).Round(time.Second)

	var eta time.Duration
	if batch > 0 {
		perBatch := elapsed.Seconds() / float64(batch)
		eta = time.Duration(perBatch*float64(totalBatches-batch)) * time.Second
	}

	fmt.Printf("\rEpoch %d/%d [%s] %3.0f%% loss=%.4f elapsed=%v eta=%v   ",
		epoch, totalEpochs, progressBar(percent, 30), percent, avgLoss, elapsed, eta)
}

func progressBar(percent float64, width int) string {
	filled := int(percent / 100 * float64(width))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	return strings.Repeat("=", filled) + strings.Repeat(" ", width-filled)
}


func (l *TrainingLogger) AddLoss(loss float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lossHistory = append(l.lossHistory, loss)
}


func (l *TrainingLogger) AddAccuracy(accuracy float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.accuracyHistory = append(l.accuracyHistory, accuracy)
	fmt.Println() // end the in-place progress line before printing a summary
	fmt.Printf("  -> validation accuracy: %.2f%%  loss trend: %s\n", accuracy, sparkline(l.lossHistory, 40))
}


func (l *TrainingLogger) Log(message string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	fmt.Printf("[%s] %s\n", time.Now().Format("15:04:05"), message)
}


func (l *TrainingLogger) Close() {}

var sparkChars = []rune("▁▂▃▄▅▆▇█")


func sparkline(data []float64, width int) string {
	if len(data) == 0 {
		return ""
	}
	if len(data) > width {
		data = data[len(data)-width:]
	}
	lo, hi := data[0], data[0]
	for _, v := range data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	span := hi - lo

	var b strings.Builder
	for _, v := range data {
		idx := 0
		if span != 0 {
			idx = int((v - lo) / span * float64(len(sparkChars)-1))
		}
		b.WriteRune(sparkChars[idx])
	}
	return b.String()
}