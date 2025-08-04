import os
import json
import csv
import argparse
from tqdm import tqdm
from datetime import datetime

# Example command:
# python evaluation_classification.py --folder /home/ylin/NEW_RS_RAG/---------GITHUB/TASKS/answer/model_answers_classification/RSWK

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate classification task outputs.")
parser.add_argument("--folder", type=str, required=True, help="Path to folder with JSONL files.")
args = parser.parse_args()

# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Truncate float without rounding
def truncate_float(value, digits):
    str_val = f"{value:.10f}"
    integer_part, _, decimal_part = str_val.partition(".")
    truncated_decimal = decimal_part[:digits].ljust(digits, "0")
    return f"{integer_part}.{truncated_decimal}"

folder_path = args.folder
results = []

# Iterate through each JSONL file
for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
    if not filename.endswith(".jsonl"):
        continue

    parts = filename.replace(".jsonl", "").split("_")
    dataset_name = parts[0]
    model_name = parts[2] if len(parts) > 2 else "unknown"

    rag_sign = imgw = textw = topk = None

    # Parse rag/imgw/topk from filename
    if "rag" in filename and "imgw" in filename and "topk" in filename:
        try:
            rag_index = parts.index("rag")
            imgw_part = parts[rag_index + 1]
            topk_part = parts[rag_index + 2]

            rag_sign = "rag"
            imgw = float(imgw_part.replace("imgw", ""))
            textw = 1 - imgw
            topk = int(topk_part.replace("topk", ""))
        except (ValueError, IndexError):
            pass

    # Classification evaluation
    filepath = os.path.join(folder_path, filename)
    total = 0
    correct = 0
    per_class_stats = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ref = next((msg["content"] for msg in data["messages"] if msg["role"] == "assistant"), "").strip().lower()
            pred = data.get("prediction", "").strip().lower()

            if not ref:
                continue

            total += 1
            if ref == pred:
                correct += 1

            if ref not in per_class_stats:
                per_class_stats[ref] = {"correct": 0, "total": 0}
            per_class_stats[ref]["total"] += 1
            if ref == pred:
                per_class_stats[ref]["correct"] += 1

    if total == 0:
        continue

    acc = correct / total
    per_class_str = " | ".join(
        f"{label}:{truncate_float(stats['correct'] / stats['total'], 4)}"
        for label, stats in per_class_stats.items()
    )

    results.append({
        "dataset": dataset_name,
        "model": model_name,
        "rag": rag_sign or "",
        "imgw": imgw if imgw is not None else "",
        "textw": textw if textw is not None else "",
        "topk": topk if topk is not None else "",
        "total_entries": total,
        "Accuracy": truncate_float(acc, 4),
        "Per-Class Accuracy": per_class_str
    })

# Save to CSV
csv_filename = f"classification_eval_{timestamp}.csv"
csv_path = os.path.join(folder_path, csv_filename)

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "dataset", "model", "rag", "imgw", "textw", "topk",
        "total_entries", "Accuracy", "Per-Class Accuracy"
    ])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"✅ Classification evaluation completed! Results saved to: {csv_path}")
