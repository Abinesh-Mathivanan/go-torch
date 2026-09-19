package optimizer

import "math"

// computes the learning rate to use at a given training step
type Scheduler interface {
	LRAt(step int) float64
}

// sets opt's learning rate to sched.LRAt(step). 
func ApplyLR(opt LRSetter, sched Scheduler, step int) {
	opt.SetLR(sched.LRAt(step))
}

// decays the learning rate by a factor of Gamma every StepSize
// steps: lr = BaseLR * Gamma^floor(step / StepSize).
type StepLR struct {
	BaseLR   float64
	StepSize int
	Gamma    float64
}

func (s *StepLR) LRAt(step int) float64 {
	if s.StepSize <= 0 {
		return s.BaseLR
	}
	return s.BaseLR * math.Pow(s.Gamma, float64(step/s.StepSize))
}

// smoothly decays the learning rate from BaseLR to MinLR
// following a half-cosine curve over TotalSteps, then holds at MinLR.
type CosineAnnealingLR struct {
	BaseLR     float64
	MinLR      float64
	TotalSteps int
}

func (s *CosineAnnealingLR) LRAt(step int) float64 {
	if s.TotalSteps <= 0 || step >= s.TotalSteps {
		return s.MinLR
	}
	progress := float64(step) / float64(s.TotalSteps)
	return s.MinLR + 0.5*(s.BaseLR-s.MinLR)*(1+math.Cos(math.Pi*progress))
}

// linearly ramps the learning rate from 0 up to the target learning rate over WarmupSteps, then 
// delegates to Inner for subsequent steps.
type WarmupScheduler struct {
	Inner       Scheduler
	WarmupSteps int
}

func (s *WarmupScheduler) LRAt(step int) float64 {
	if s.WarmupSteps <= 0 {
		return s.Inner.LRAt(step)
	}
	if step < s.WarmupSteps {
		target := s.Inner.LRAt(0)
		return target * float64(step+1) / float64(s.WarmupSteps)
	}
	return s.Inner.LRAt(step - s.WarmupSteps)
}