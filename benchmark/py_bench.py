import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import gzip
import os
import psutil 


def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    # rss: Resident Set Size, the non-swapped physical memory a process has used.
    # this is a good proxy for 'Alloc' in Go.
    mem_info = process.memory_info()
    print(f"\n--- Memory Stats after {stage} ---")
    print(f"Process Memory (RSS): {mem_info.rss / 1024 / 1024:.2f} MiB")
    print("---------------------------------")



MNIST_DIR = "mnist_data"

def load_mnist_images(filepath):
    try:
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    except gzip.BadGzipFile:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)

    images = data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    return torch.from_numpy(images)

def load_mnist_labels(filepath):
    try:
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    except gzip.BadGzipFile:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    
    return torch.from_numpy(data).long()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print("--- PyTorch Benchmark ---")
    print_memory_usage("Program Start")

    
    print("Preparing MNIST dataset from local files...")
    
    def get_path(folder_name, file_name):
        return os.path.join(MNIST_DIR, folder_name, file_name)

    try:
        train_images = load_mnist_images(get_path("train-images-idx3-ubyte", "train-images-idx3-ubyte"))
        train_labels = load_mnist_labels(get_path("train-labels-idx1-ubyte", "train-labels-idx1-ubyte"))
    except FileNotFoundError:
        print("Could not find uncompressed files in subdirectories.")
        exit(1)
        
    print(f"Loaded {len(train_images)} training images.")
    print_memory_usage("Data Loading")

    

    num_classes = 10
    learning_rate = 0.01
    batch_size = 32
    epochs = 3

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print_memory_usage("Model Initialization")
    

    # training loop
    num_train_samples = len(train_images)
    indices = np.arange(num_train_samples)

    print(f"\nStarting training: {epochs} epochs, batch size {batch_size}...")
    
    model.train()
    total_training_time = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        np.random.seed(42)
        np.random.shuffle(indices)
        
        for i in range(0, num_train_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            images = train_images[batch_indices].to(DEVICE)
            labels = train_labels[batch_indices].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.4f} seconds.")
        print_memory_usage(f"End of Epoch {epoch + 1}")

    print("\n--- Benchmark Results ---")
    print(f"Total training time for {epochs} epochs: {total_training_time:.4f} seconds.")
    print(f"Average time per epoch: {total_training_time / epochs:.4f} seconds.")