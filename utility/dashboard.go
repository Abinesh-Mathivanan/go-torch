package utility

import (
	"fmt"
	"runtime"
	"strings"
	"time"

	tslc "github.com/NimbleMarkets/ntcharts/linechart/timeserieslinechart"
	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	zone "github.com/lrstanley/bubblezone"
)

// --- Messages ---
type statsMsg struct {
	epoch, tEpoch, batch, tBatch int
	avgLoss                      float32
	throughput                   float64
	heap, sys                    uint64
	gcCount                      uint32
	goroutines                   int
	eta                          time.Duration
}

type lossPointMsg float32
type epochDoneMsg struct {
	duration time.Duration
	accuracy float32
}
type logMsg string

// --- Model ---
type dashboardModel struct {
	lr           float32
	bs, epochs   int
	stats        statsMsg
	epochTimings []string
	logs         []string

	// UI Components
	lossChart   tslc.Model
	progressBar progress.Model
	zoneManager *zone.Manager

	width, height int
}

func (m dashboardModel) Init() tea.Cmd { return nil }

func (m dashboardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.progressBar.Width = m.width - 10
		// Dynamically resize the chart
		m.lossChart.Resize(m.width-6, 12)
		return m, nil

	case lossPointMsg:
		// Push new training loss point to ntchart
		m.lossChart.Push(tslc.TimePoint{
			Time:  time.Now(),
			Value: float64(msg),
		})
		m.lossChart.DrawBrailleAll()
		return m, nil

	case statsMsg:
		m.stats = msg
		return m, m.progressBar.SetPercent(float64(msg.batch) / float64(msg.tBatch))

	case epochDoneMsg:
		entry := fmt.Sprintf("E%d: Acc %.2f%% in %v", len(m.epochTimings)+1, msg.accuracy, msg.duration.Round(time.Second))
		m.epochTimings = append(m.epochTimings, entry)
		return m, nil

	case logMsg:
		m.logs = append(m.logs, string(msg))
		if len(m.logs) > 10 {
			m.logs = m.logs[1:]
		}
		return m, nil

	case tea.KeyMsg:
		if msg.String() == "q" || msg.String() == "ctrl+c" {
			return m, tea.Quit
		}

	case progress.FrameMsg:
		newP, cmd := m.progressBar.Update(msg)
		m.progressBar = newP.(progress.Model)
		return m, cmd
	}

	// Forward mouse/keyboard events to chart (allows zooming/panning)
	m.lossChart, _ = m.lossChart.Update(msg)
	m.lossChart.DrawBrailleAll()

	return m, nil
}

// --- View ---
var (
	titleStyle = lipgloss.NewStyle().Background(lipgloss.Color("62")).Foreground(lipgloss.Color("230")).Padding(0, 1).Bold(true)
	boxStyle   = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("62")).Padding(0, 1)
	dimStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
)

func (m dashboardModel) View() string {
	if m.width == 0 {
		return "Initializing High-Res Dashboard..."
	}

	// 1. NTCharts Loss Window
	chartHeader := lipgloss.JoinHorizontal(lipgloss.Center,
		titleStyle.Render("LIVE LOSS (BRAILLE)"),
		dimStyle.Render(" [Scroll to Pan, +/- to Zoom]"),
	)
	chartBox := boxStyle.Width(m.width - 2).Render(
		lipgloss.JoinVertical(lipgloss.Left, chartHeader, m.lossChart.View()),
	)

	// 2. Info Grid
	trainInfo := lipgloss.JoinVertical(lipgloss.Left,
		titleStyle.Render("TRAINING"),
		fmt.Sprintf("%s %d/%d", dimStyle.Render("Epoch:"), m.stats.epoch, m.stats.tEpoch),
		fmt.Sprintf("%s %d/%d", dimStyle.Render("Batch:"), m.stats.batch, m.stats.tBatch),
		fmt.Sprintf("%s %.2f/s", dimStyle.Render("Speed:"), m.stats.throughput),
		fmt.Sprintf("%s %.4f", dimStyle.Render("Loss: "), m.stats.avgLoss),
		fmt.Sprintf("%s %v", dimStyle.Render("ETA:  "), m.stats.eta),
	)

	sysInfo := lipgloss.JoinVertical(lipgloss.Left,
		titleStyle.Render("SYSTEM"),
		fmt.Sprintf("%s %d MiB", dimStyle.Render("Heap:"), m.stats.heap),
		fmt.Sprintf("%s %d MiB", dimStyle.Render("Total:"), m.stats.sys/1024/1024),
		fmt.Sprintf("%s %d", dimStyle.Render("GCs:  "), m.stats.gcCount),
		fmt.Sprintf("%s %d", dimStyle.Render("Go:   "), m.stats.goroutines),
	)

	epochInfo := lipgloss.JoinVertical(lipgloss.Left,
		titleStyle.Render("HISTORY"),
		strings.Join(m.epochTimings, "\n"),
	)

	midRow := lipgloss.JoinHorizontal(lipgloss.Top,
		boxStyle.Width(m.width/3-1).Height(8).Render(trainInfo),
		boxStyle.Width(m.width/3-1).Height(8).Render(epochInfo),
		boxStyle.Width(m.width/3-1).Height(8).Render(sysInfo),
	)

	// 3. Progress and Logs
	logBox := boxStyle.Width(m.width-2).Height(10).Render(
		lipgloss.JoinVertical(lipgloss.Left, titleStyle.Render("LOGS"), strings.Join(m.logs, "\n")),
	)

	ui := lipgloss.JoinVertical(lipgloss.Left,
		chartBox,
		midRow,
		"\n "+m.progressBar.View(),
		logBox,
	)

	// Wrap in BubbleZone for Mouse Support
	return m.zoneManager.Scan(ui)
}

// --- Public API ---

type TrainingDashboard struct {
	p *tea.Program
}

func NewTrainingDashboard(lr float32, bs, e int) *TrainingDashboard {
	zm := zone.New()
	chart := tslc.New(80, 12)
	chart.SetZoneManager(zm)
	chart.Focus()
	chart.SetStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("204")))

	m := dashboardModel{
		lr:          lr,
		bs:          bs,
		epochs:      e,
		lossChart:   chart,
		progressBar: progress.New(progress.WithDefaultGradient()),
		zoneManager: zm,
	}

	return &TrainingDashboard{
		p: tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion()),
	}
}

func (d *TrainingDashboard) UpdateStats(e, te, b, tb int, loss float32, start, total time.Time) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	elapsed := time.Since(start)
	throughput := float64(b*32) / elapsed.Seconds()

	eta := time.Duration(0)
	if b > 0 {
		eta = time.Duration(float64(elapsed.Seconds())/float64(b)*float64(tb-b)) * time.Second
	}

	d.p.Send(statsMsg{
		epoch: e, tEpoch: te, batch: b, tBatch: tb, avgLoss: loss,
		throughput: throughput,
		heap:       m.HeapInuse / 1024 / 1024,
		sys:        m.Sys,
		gcCount:    m.NumGC,
		goroutines: runtime.NumGoroutine(),
		eta:        eta,
	})
}

func (d *TrainingDashboard) AddLoss(v float32) { d.p.Send(lossPointMsg(v)) }

func (d *TrainingDashboard) AddAccuracy(acc float32, duration time.Duration) {
	d.p.Send(epochDoneMsg{duration: duration, accuracy: acc})
}

func (d *TrainingDashboard) Log(msg string) { d.p.Send(logMsg(msg)) }
func (d *TrainingDashboard) Loop()          { d.p.Run() }