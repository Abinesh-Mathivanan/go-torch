package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"
)

func main() {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("dashboard in a new native window...")

	var cmd *exec.Cmd

	switch runtime.GOOS {

	case "windows":
		cmd = exec.Command("cmd", "/C", "start", "cmd", "/K", "go run ./mnist_trainer/train.go")

	case "darwin": 
		script := fmt.Sprintf("tell application \"Terminal\" to do script \"cd %s && go run ./mnist_trainer/train.go\"", wd)
		cmd = exec.Command("osascript", "-e", script)

	case "linux":
		terminals := []string{"gnome-terminal", "konsole", "xterm"}
		found := false
		for _, term := range terminals {
			path, err := exec.LookPath(term)
			if err == nil {
				if term == "gnome-terminal" {
					cmd = exec.Command(path, "--", "go", "run", "./mnist_trainer/train.go")
				} else {
					cmd = exec.Command(path, "-e", "go run ./mnist_trainer/train.go")
				}
				found = true
				break
			}
		}
		if !found {
			log.Fatal("Could not find a terminal emulator (gnome-terminal, konsole, xterm)")
		}
	default:
		log.Fatalf("Unsupported OS: %s", runtime.GOOS)
	}

	err = cmd.Run()
	if err != nil {
		log.Fatalf("Failed to launch terminal: %v", err)
	}

	fmt.Println("Terminal launched")
}