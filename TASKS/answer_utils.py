import os
import json
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
from swift.llm import InferRequest
from functools import lru_cache
from typing import List, Dict, Tuple
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Utility functions ----------
def get_image_embedding(image_path, processor, clip_model, device):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_feat = clip_model.get_image_features(**image_inputs)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    return image_feat.cpu().numpy()[0]

def get_text_embedding(text, processor, clip_model, device):
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy()[0]

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path, data):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

image_embedding_cache = {}

def hybrid_search(
    image_path: str,
    prompt: str,
    client: None,
    model: None,
    processor: None,
    device: None,
    task_type: str = None,   # Options: "caption", "vqa", "classification"
    alpha: float = 0.7,           # Image-text fusion weight
    top_k: int = 3,
    search_k: int = 100000,       # How many items to retrieve from Qdrant before re-ranking
    collection_name: str = None
) -> Tuple[List[Dict], str]:
    """
    Hybrid search using Qdrant vector retrieval + weighted image-text scoring.
    """

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    print(f"Using collection: {collection_name}, limit={search_k}")
    task_type = task_type.lower()

    if task_type == "vqa":
        allowed_chunk_types = [
            "basic_info_and_history",
            "architecture_and_function",
            "monthly_expert_knowledge",
            "culture_and_visitor",
            "expert_knowledge",
        ]
    elif task_type == "caption":
        allowed_chunk_types = ["caption"]
    elif task_type == "classification":
        allowed_chunk_types = ["basic_info_and_history", "image_description"]
    else:
        raise ValueError(f"❌ Invalid task_type: {task_type}")

    # Generate query vectors
    try:
        query_image_vec = get_image_embedding(image_path, processor, model, device).tolist()
        query_text_vec = get_text_embedding(prompt, processor, model, device).tolist()
    except Exception as e:
        print(f"❌ Failed to generate query vectors: {e}")
        return [], ""

    # Qdrant filter (only include allowed chunk types)
    filter_condition = {
        "must": [
            {"key": "chunk_type", "match": {"any": allowed_chunk_types}}
        ]
    }

    # Qdrant vector search
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_image_vec,
            limit=search_k,
            with_payload=True,
            with_vectors=True,
            query_filter=filter_condition
        )
        print("image vector search")
    except Exception as e:
        print(f"❌ Qdrant search failed: {e}")
        return [], ""

    # Compute image-text fusion scores
    scored = []
    for h in hits:
        stored_image_vec = h.payload.get("image_vector")
        stored_text_vec = h.vector
        if stored_image_vec is None or stored_text_vec is None:
            continue

        k1 = cosine_similarity(np.array(query_image_vec), np.array(stored_image_vec))
        k2 = cosine_similarity(np.array(query_text_vec), np.array(stored_text_vec))
        k3 = alpha * k1 + (1 - alpha) * k2

        scored.append({
            "image_path": h.payload.get("image_path", ""),
            "text_chunk": h.payload.get("text", ""),
            "chunk_type": h.payload.get("chunk_type", ""),
            "final_score": float(k3),
            "image_score": float(k1),
            "text_score": float(k2)
        })

    # Sort and select top_k results
    sorted_results = sorted(scored, key=lambda x: x["final_score"], reverse=True)[:top_k]

    # Concatenate context
    context_string = "\n".join([f"{i + 1}. {r['text_chunk']}" for i, r in enumerate(sorted_results)])
    print(f"search: {image_path}\nreturned:\n{context_string}")

    return sorted_results, context_string

# ---------- Inference ----------
def run_infer_batch(buffer, engine, request_config, output_file):
    infer_requests = [
        InferRequest(messages=[{"role": "user", "content": entry["prompt"]}], images=[entry["image"]])
        for entry in buffer
    ]
    responses = engine.infer(infer_requests, request_config)

    for entry, resp in zip(buffer, responses):
        entry["meta"]["prediction"] = resp.choices[0].message.content
        write_jsonl(output_file, entry["meta"])

# ---------- Main process ----------
def process_jsonl(
    input_file, 
    output_file,
    engine, 
    request_config,
    processor, 
    model, 
    client,
    image_weight=0.9, 
    top_k=3, 
    use_rag=True,
    batch_size=16, 
    task_type=None, 
    device=None,
    collection_name=None
):
    text_weight = round(1 - image_weight, 2)
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

    # Clear existing output
    with open(output_file, 'w', encoding='utf-8'):
        pass

    buffer = []
    for item in tqdm(read_jsonl(input_file), total=total_lines,
                     desc=f"[{'RAG' if use_rag else 'NoRAG'}] W={image_weight}, K={top_k}"):
        images = item.get("images", [])
        messages = item.get("messages", [])
        user_question = next((m["content"] for m in messages if m["role"] == "user"), None)

        if not images or not user_question:
            continue

        for img in images:
            try:
                if use_rag:
                    if img not in image_embedding_cache:
                        image_embedding_cache[img] = get_image_embedding(img, processor=processor, clip_model=model, device=device)
                    image_vec = image_embedding_cache[img]
                    text_vec = get_text_embedding(user_question, processor=processor, clip_model=model, device=device)

                    _, context = hybrid_search(
                        image_path=img, 
                        prompt=user_question, 
                        client=client,
                        model=model,
                        processor=processor,
                        device=device,
                        task_type=task_type,
                        alpha=image_weight,
                        top_k=top_k,
                        collection_name=collection_name
                    )

                    combined_prompt = f"with those context: {context}, {user_question}"
                    print("Context(combined retrived ):", combined_prompt)
                else:
                    combined_prompt = user_question
                    print("Prompt without rag:", combined_prompt)

                buffer.append({
                    "image": img,
                    "prompt": combined_prompt,
                    "meta": item
                })

                if len(buffer) >= batch_size:
                    run_infer_batch(buffer, engine, request_config, output_file)
                    buffer.clear()

            except Exception as e:
                print(f"❌ Context generation failed for {img}: {e}")
                continue

    if buffer:
        run_infer_batch(buffer, engine, request_config, output_file)

    print(f"✅ Finished: {output_file}")
