import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from swift.llm import PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template
from swift.tuners import Swift
from answer_utils import process_jsonl
from configs import (
    ChatEarthNet_sign, ChatEarthNet_file_path,
    RSWK_sign, RSWK_file_path_caption, RSWK_file_path_vqa, RSWK_file_path_classification,
    batch_size, clip_base_path,
    geochat_base_path, geochat_ft_path,
    qwen_base_path, qwen_ft_path,
    llama_base_path, llama_ft_path,
    intervl_base_path, intervl_ft_path,
    janus_base_path, janus_ft_path,
    max_tokens_caption, temperature_caption,
    max_tokens_vqa, temperature_vqa,
    max_tokens_classification, temperature_classification,
    caption_answer_path, vqa_answer_path, classification_answer_path
)

# eg: python answer.py --model llama --task classification --dataset RSWK_classification --weight 0.9 --gpu 3 --top_k 5 --rag_set

# ---------- Argument parser ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen",
                    choices=["qwen", "llama", "janus", "intervl", "geochat"],
                    help="LLM model to use")

parser.add_argument("--task", type=str, default="caption",
                    choices=["caption", "vqa", "classification"],
                    help="Task type")

parser.add_argument("--dataset", type=str, default="RSWK_caption",
                    choices=["ChatEarthNet", "RSWK_vqa", "RSWK_caption", "RSWK_classification"],
                    help="Dataset name")

parser.add_argument("--gpu", type=str, default="0", help="GPU device, e.g., '0' or '1,2'")
parser.add_argument("--top_k", type=int, nargs='+', default=[1],
                    choices=[1, 3, 5],
                    help="Top K retrieved chunks, e.g., --top_k 1 3 5")

parser.add_argument("--weight", type=float, nargs='+', default=[0.5],
                    choices=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="Weight(s) of image similarity score")

parser.add_argument("--rag_set", action="store_true", help="Enable RAG retrieval augmentation")

args = parser.parse_args()

# ---------- Environment setup ----------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model path selection ----------
model_paths = {
    "qwen": (qwen_base_path, qwen_ft_path),
    "llama": (llama_base_path, llama_ft_path),
    "intervl": (intervl_base_path, intervl_ft_path),
    "janus": (janus_base_path, janus_ft_path),
    "geochat": (geochat_base_path, geochat_ft_path),
}
base_path, ft_path = model_paths[args.model]

# ---------- Dataset path selection ----------
dataset_map = {
    "ChatEarthNet": (ChatEarthNet_sign, ChatEarthNet_file_path),
    "RSWK_caption": (RSWK_sign, RSWK_file_path_caption),
    "RSWK_vqa": (RSWK_sign, RSWK_file_path_vqa),
    "RSWK_classification": (RSWK_sign, RSWK_file_path_classification),
}
dataset_sign, input_file = dataset_map[args.dataset]

# ---------- Output path selection ----------
task_output_map = {
    "caption": caption_answer_path,
    "vqa": vqa_answer_path,
    "classification": classification_answer_path,
}
output_dir = Path(task_output_map[args.task]) / dataset_sign
output_dir.mkdir(parents=True, exist_ok=True)

# ---------- Task configuration ----------
task_config_map = {
    "caption": {
        "max_tokens": max_tokens_caption,
        "temperature": temperature_caption
    },
    "vqa": {
        "max_tokens": max_tokens_vqa,
        "temperature": temperature_vqa
    },
    "classification": {
        "max_tokens": max_tokens_classification,
        "temperature": temperature_classification
    }
}

# 
task_config = task_config_map[args.task]
max_tokens = task_config["max_tokens"]
temperature = task_config["temperature"]

output_dir = Path(task_output_map[args.task]) / dataset_sign
output_dir.mkdir(parents=True, exist_ok=True)

# ---------- load LLM ----------
model, tokenizer = get_model_tokenizer(base_path)
model = Swift.from_pretrained(model, safe_snapshot_download(ft_path))
template = get_template(model.model_meta.template, tokenizer)
engine = PtEngine.from_model_template(model, template)
request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)

# ---------- Load CLIP model ----------
clip_model = CLIPModel.from_pretrained(clip_base_path).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_base_path)
clip_model.eval()

# ---------- Qdrant vector database connection ----------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)

# ---------- Execute task ----------
use_rag = args.rag_set
for image_weight in args.weight:
    for top_k in args.top_k:
        rag_flag = "_rag" if use_rag else ""
        output_file = output_dir / f"{dataset_sign}_answer_{args.model}{rag_flag}_imgw{image_weight}_topk{top_k}.jsonl"
        process_jsonl(
            input_file=input_file,
            output_file=output_file,
            engine=engine,
            request_config=request_config,
            processor=clip_processor,
            model=clip_model,
            client=client,
            image_weight=image_weight,
            top_k=top_k,
            use_rag=use_rag,
            batch_size=batch_size,
            task_type=args.task,
            collection_name=" ", # qdrant dataset name
            device=device,
        )
