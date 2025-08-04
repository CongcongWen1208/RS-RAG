import os
import json
import csv
import argparse
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

"""
Example commands:

python evaluation_caption_vqa.py --folder /home/ylin/NEW_RS_RAG/---------GITHUB/TASKS/answer/model_answers_vqa/RSWK
python evaluation_caption_vqa.py --folder /home/ylin/NEW_RS_RAG/---------GITHUB/TASKS/answer/model_answers_caption/RSWK

"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate caption model outputs.")
parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing JSONL files.")
args = parser.parse_args()

# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Truncate float to fixed decimal places without rounding
def truncate_float(value, digits):
    str_val = f"{value:.10f}"
    integer_part, _, decimal_part = str_val.partition(".")
    truncated_decimal = decimal_part[:digits].ljust(digits, "0")
    return f"{integer_part}.{truncated_decimal}"

# Use specified folder path
folder_path = args.folder

# Initialize
results = []
smoothie = SmoothingFunction().method1
cider_scorer = Cider()

# Iterate through JSONL files
for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
    if not filename.endswith(".jsonl"):
        continue

    parts = filename.replace(".jsonl", "").split("_")
    dataset_name = parts[0]
    model_name = parts[2] if len(parts) > 2 else "unknown"

    rag_sign = imgw = textw = topk = None

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

    filepath = os.path.join(folder_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        # Pre-compute total lines for progress bar
        total_lines = sum(1 for _ in f)
        f.seek(0)

        references, predictions = [], []
        # Progress bar for each file
        for line in tqdm(f, total=total_lines, desc=f"Processing {filename}", leave=False):
            data = json.loads(line)
            ref = next((msg["content"] for msg in data["messages"] if msg["role"] == "assistant"), "")
            pred = data.get("prediction", "")
            if ref and pred:
                references.append(ref)
                predictions.append(pred)

    # Compute metrics
    bleu1 = bleu2 = bleu3 = bleu4 = meteor = rouge_l = 0.0
    count = len(references)
    if count == 0:
        continue

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    gts = {str(i): [ref] for i, ref in enumerate(references)}
    res = {str(i): [pred] for i, pred in enumerate(predictions)}
    cider_score, _ = cider_scorer.compute_score(gts, res)

    for ref, pred in tqdm(zip(references, predictions), total=count, desc="Calculating metrics", leave=False):
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        bleu1 += sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        meteor += meteor_score([ref_tokens], pred_tokens)
        rouge_l += scorer.score(ref, pred)['rougeL'].fmeasure

    results.append({
        "dataset": dataset_name,
        "model": model_name,
        "rag": rag_sign or "",
        "imgw": imgw if imgw is not None else "",
        "textw": textw if textw is not None else "",
        "topk": topk if topk is not None else "",
        "total_entries": count,
        "BLEU-1": bleu1 / count,
        "BLEU-2": bleu2 / count,
        "BLEU-3": bleu3 / count,
        "BLEU-4": bleu4 / count,
        "METEOR": meteor / count,
        "ROUGE-L": rouge_l / count,
        "CIDEr": cider_score
    })

# Save results to CSV with timestamp
csv_filename = f"evaluation_results_{timestamp}.csv"
csv_path = os.path.join(folder_path, csv_filename)
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "dataset", "model", "rag", "imgw", "textw", "topk", "total_entries",
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"
    ])
    writer.writeheader()
    for row in results:
        formatted_row = {
            k: truncate_float(v, 4) if isinstance(v, float) else v
            for k, v in row.items()
        }
        writer.writerow(formatted_row)

print(f"✅ Evaluation complete. Results saved to: {csv_path}")
