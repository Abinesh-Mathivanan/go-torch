import os
from tabulate import tabulate

skip_files = {"benchmark.py", "benchmark.go", "code_analysis.py", "README.md"}  # filenames to skip (case-sensitive)
exts = {".go"}  # File extensions to consider

stats = []

def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        total = blank = comment = 0
        for line in f:
            total += 1
            stripped = line.strip()
            if not stripped:
                blank += 1
            elif stripped.startswith("//"):
                comment += 1
        return total, comment, blank

def walk_and_count(root):
    for dirpath, _, files in os.walk(root):
        for file in files:
            if file in skip_files or not any(file.endswith(ext) for ext in exts):
                continue
            full_path = os.path.join(dirpath, file)
            total, comment, blank = count_lines(full_path)
            stats.append([file, total, comment, blank, total - comment - blank])

walk_and_count(".")

headers = ["File", "Total", "Comments", "Blanks", "Code"]
print(tabulate(stats, headers=headers, tablefmt="grid"))

total_code = sum(row[4] for row in stats)
print(f"\nTotal Code Lines: {total_code}")
