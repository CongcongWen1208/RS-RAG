import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from nltk.tokenize import word_tokenize, sent_tokenize
import json
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Batch
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple


# -----------------------------
# Configuration Parameters
# -----------------------------
path = "/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/clip-vit-base-patch32"
json_path_world_knowledge = "/home/ylin/LYT-RSRAG/RSGPT_dataset/dataset/chunk/world-expert_knowledge.json"
json_path_monthly_expert_knowledge = "/home/ylin/LYT-RSRAG/RSGPT_dataset/dataset/chunk/monthly_knowledge.json"
json_path_image_description = "/home/ylin/LYT-RSRAG/RSGPT_dataset/dataset/chunk/image_descriptions.json"
image_base_path = "/home/ylin/LYT-RSRAG/RSGPT_dataset/images"
device = "cuda:3"
qdrant_dataset_name  = "Test_dataset_RSWK"                                             # Qdrant collection name
batch_size           = 128                                                                      # Upload batch size
qdrant_name          = "localhost"                                                             # Qdrant host (default: localhost)
qdrant_port_number   = 6333                                                                    # Qdrant port (default: 6333)
vector_dimension     = 512                                                                     # Dimension of embedded vectors

# -----------------------------
# Load CLIP Model & Processor
# -----------------------------
try:
    clip_model = CLIPModel.from_pretrained(path)
    clip_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained(path)
    print(f"✅ Model loaded successfully, using device: {device}")
except Exception as e:
    print("❌ Failed to load model:", e)

# -----------------------------
# Multimodal Embedding Functions
# -----------------------------
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_feat = clip_model.get_image_features(**image_inputs)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    return image_feat.cpu().numpy()[0]

def get_text_embedding(text):
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy()[0]

def resolve_image_path(image_base_path, image_name):
    if not image_name.endswith(".png"):
        image_name += ".png"
    return os.path.join(image_base_path, image_name)

# -----------------------------
# Chunks creating Functions
# -----------------------------
def build_chunks_with_image_vec(data, image_base_path, image_vec_cache):
    chunks = []
    image_name = data.get("image_name")
    image_path = resolve_image_path(image_base_path, image_name)

    if image_path in image_vec_cache:
        image_vec = image_vec_cache[image_path]
    else:
        try:
            image_vec = get_image_embedding(image_path).tolist()
            image_vec_cache[image_path] = image_vec
        except Exception as e:
            print(f"❌  {image_name}: {e}")
            return chunks

    wk = data.get("world knowledge", {})
    rs = data.get("remote sensing expert knowledge", {})
    caption_parts = []

    # Chunk 1: basic_info_and_history
    basic_fields = ["Name", "Category", "Area", "Location", "Address", "Physical_Area", "Construction_Period"]
    basic_info = "; ".join(f"{field}: {wk[field]}" for field in basic_fields if field in wk)
    historical = f"Historical_Background: {wk.get('Historical_Background', '').strip()}"
    events = wk.get("Major_Events", [])
    event_texts = [f"- {e.get('Event', '')} ({e.get('Year', '')}): {e.get('Description', '')}" for e in events if "Description" in e]
    major_event_block = "Major Events:\n" + "\n".join(event_texts) if event_texts else ""
    chunk1_text = "\n".join([basic_info, historical, major_event_block]).strip()
    if chunk1_text:
        caption_parts.append(chunk1_text)
        vec = get_text_embedding(chunk1_text)
        chunks.append(PointStruct(id=uuid4().hex, vector=vec.tolist(), payload={
            "image_path": image_path,
            "image_vector": image_vec,
            "chunk_type": "basic_info_and_history",
            "text": chunk1_text
        }))

    # Chunk 2: architecture_and_function
    arch = wk.get("Architectural_Characteristics", "")
    func = wk.get("Primary_Function", "")
    details = wk.get("Details", "")
    chunk2_text = f"Architectural_Characteristics: {arch}\nPrimary_Function: {func}\nDetails: {details}".strip()
    if chunk2_text:
        caption_parts.append(chunk2_text)
        vec = get_text_embedding(chunk2_text)
        chunks.append(PointStruct(id=uuid4().hex, vector=vec.tolist(), payload={
            "image_path": image_path,
            "image_vector": image_vec,
            "chunk_type": "architecture_and_function",
            "text": chunk2_text
        }))

    # Chunk 3: culture_and_visitor
    culture = wk.get("Cultural_Significance", "")
    visitors = wk.get("Notable_Visitors", [])
    visitors_text = ", ".join(visitors) if isinstance(visitors, list) else visitors
    chunk3_text = f"Cultural_Significance: {culture}\nNotable_Visitors: {visitors_text}".strip()
    if chunk3_text:
        caption_parts.append(chunk3_text)
        vec = get_text_embedding(chunk3_text)
        chunks.append(PointStruct(id=uuid4().hex, vector=vec.tolist(), payload={
            "image_path": image_path,
            "image_vector": image_vec,
            "chunk_type": "culture_and_visitor",
            "text": chunk3_text
        }))

    # Chunk 4: expert_knowledge
    chunk4_text = ""
    if rs:
        rs_text = "; ".join(f"{k}: {v}" for k, v in rs.items())
        chunk4_text = f"expert knowledge: {rs_text}"
        caption_parts.append(chunk4_text)
        vec = get_text_embedding(chunk4_text)
        chunks.append(PointStruct(id=uuid4().hex, vector=vec.tolist(), payload={
            "image_path": image_path,
            "image_vector": image_vec,
            "chunk_type": "expert_knowledge",
            "text": chunk4_text
        }))
    # 
    return chunks, caption_parts


#--- monthly expert knowledge chunks
def build_monthly_chunks_with_image_vec(data, image_base_path, image_vec_cache):
    chunks = []
    image_name = data.get("image_name")
    image_path = resolve_image_path(image_base_path, image_name)

    if image_path in image_vec_cache:
        image_vec = image_vec_cache[image_path]
    else:
        try:
            image_vec = get_image_embedding(image_path).tolist()
            image_vec_cache[image_path] = image_vec
        except Exception as e:
            print(f"❌ Failed to get image embedding for {image_name}: {e}")
            return chunks

    monthly_data = data.get("monthly expert knowledge", {})
    monthly_chunks = defaultdict(dict)

    for full_key, value in monthly_data.items():
        parts = full_key.strip().split()
        if len(parts) < 3:
            continue
        field = " ".join(parts[:-2])
        year = parts[-2]
        month = parts[-1]
        full_month = f"{year} {month}"
        full_field_name = f"{field} {year} {month}"
        monthly_chunks[full_month][full_field_name] = value

    for month, field_values in monthly_chunks.items():
        field_text = "; ".join(f"{k}: {v}" for k, v in field_values.items())
        text = f"monthly expert knowledge - of {month}:\n{field_text}"
        vec = get_text_embedding(text)
        chunks.append(PointStruct(id=uuid4().hex, vector=vec.tolist(), payload={
            "image_path": image_path,
            "image_vector": image_vec,
            "chunk_type": "monthly_expert_knowledge",
            "month": month,
            "text": text
        }))

    return chunks

#--- image description chunk
def build_image_description_chunk(data, image_base_path, image_vec_cache):
    chunks = []
    image_name = data.get("image_name")
    description = data.get("description", "").strip()
    if not image_name or not description:
        return chunks

    image_path = resolve_image_path(image_base_path, image_name)

    if image_path in image_vec_cache:
        image_vec = image_vec_cache[image_path]
    else:
        try:
            image_vec = get_image_embedding(image_path).tolist()
            image_vec_cache[image_path] = image_vec
        except Exception as e:
            print(f"❌ Failed to get image embedding for {image_name}: {e}")

            return chunks

    text = f"Image Description: {description}"
    try:
        vec = get_text_embedding(text).tolist()
    except Exception as e:
        print(f"❌ Failed to generate text embedding for {image_name}: {e}")
        return chunks

    chunks.append(PointStruct(id=uuid4().hex, vector=vec, payload={
        "image_path": image_path,
        "image_vector": image_vec,
        "chunk_type": "image_description",
        "text": text
    }))

    return chunks

# -----------------------------
# Qdrant Initialization
# -----------------------------
client = QdrantClient(host=qdrant_name, port=qdrant_port_number, timeout=60.0)

if not client.collection_exists(qdrant_dataset_name):
    client.recreate_collection(
        collection_name=qdrant_dataset_name,
        vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{qdrant_dataset_name}' created.")
else:
    print(f"ℹ️ Collection '{qdrant_dataset_name}' already exists.")

# -----------------------------
# Create chunks
# -----------------------------
all_points = []
image_vec_cache = {}
caption_text_map = {} 

with open(json_path_world_knowledge, "r", encoding="utf-8") as f:
    records = json.load(f)

for data in tqdm(records, desc="📦 Processing world knowledge"):
    try:
        chunks, caption_parts = build_chunks_with_image_vec(data, image_base_path, image_vec_cache)
        all_points.extend(chunks)
        image_path = resolve_image_path(image_base_path, data.get("image_name"))
        caption_text_map[image_path] = caption_parts
    except Exception as e:
        print(f"❌ Error processing world record {data.get('image_name')}: {e}")

with open(json_path_monthly_expert_knowledge, "r", encoding="utf-8") as f:
    data_list = json.load(f)

for record in tqdm(data_list, desc="📅 Processing monthly knowledge"):
    try:
        chunks = build_monthly_chunks_with_image_vec(record, image_base_path, image_vec_cache)
        all_points.extend(chunks)
    except Exception as e:
        print(f"❌ Error processing monthly record {record.get('image_name')}: {e}")


with open(json_path_image_description, "r", encoding="utf-8") as f:
    image_desc_list = json.load(f)

for desc_data in tqdm(image_desc_list, desc="🖼 Processing image descriptions"):
    try:
        chunks = build_image_description_chunk(desc_data, image_base_path, image_vec_cache)
        all_points.extend(chunks)

        # 
        image_path = resolve_image_path(image_base_path, desc_data.get("image_name"))
        caption_parts = caption_text_map.get(image_path, [])
        desc = desc_data.get("description", "").strip()
        if desc:
            caption_parts.append(f"Image Description: {desc}")

        if caption_parts:
            full_caption = "\n\n".join(caption_parts)
            vec = get_text_embedding(full_caption).tolist()
            all_points.append(PointStruct(id=uuid4().hex, vector=vec, payload={
                "image_path": image_path,
                "image_vector": image_vec_cache[image_path],
                "chunk_type": "caption",
                "text": full_caption
            }))
    except Exception as e:
        print(f"❌ Error processing image description {desc_data.get('image_name')}: {e}")

print(f"✅ total point: {len(all_points)} ")

# -----------------------------
# Uploading to Qdrant database
# -----------------------------

client.recreate_collection(
    collection_name=qdrant_dataset_name,
    vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE)
)

for i in tqdm(range(0, len(all_points), batch_size), desc="🚀 Uploading to Qdrant"):
    try:
        client.upsert(
            collection_name=qdrant_dataset_name,
            points=all_points[i:i + batch_size]
        )
    except Exception as e:
        print(f"❌ Failed to upload batch {i // batch_size}: {e}")

print(f"✅ All chunks uploaded: {len(all_points)}")


######################################### Testing Code ##############################################

def hybrid_search(
    image_path: str,
    prompt: str,
    task_type: str = "caption",     # Options: "caption", "vqa", "classification"
    alpha: float = 0.7,             # Weight for image-text fusion
    top_k: int = 3,
    search_k: int = 1000,           # Initial number of results retrieved from Qdrant for reranking
    collection_name: str = qdrant_dataset_name
) -> Tuple[List[Dict], str]:
    """
    Hybrid search using Qdrant vector retrieval + multimodal (image-text) fusion scoring.
    """

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

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
        allowed_chunk_types = ["basic_info_and_history"]
    else:
        raise ValueError(f"❌ Invalid task_type: {task_type}")

    # Generate query vectors
    try:
        query_image_vec = get_image_embedding(image_path).tolist()
        query_text_vec = get_text_embedding(prompt).tolist()
    except Exception as e:
        print(f"❌ Failed to generate query embeddings: {e}")
        return [], ""

    # Construct Qdrant filter (only search allowed chunk types)
    filter_condition = {
        "must": [
            {"key": "chunk_type", "match": {"any": allowed_chunk_types}}
        ]
    }

    # Qdrant vector search (first use text vector to retrieve top search_k)
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_text_vec,
            limit=search_k,
            with_payload=True,
            with_vectors=True,
            query_filter=filter_condition
        )
    except Exception as e:
        print(f"❌ Qdrant search failed: {e}")
        return [], ""

    # Multimodal fusion scoring
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

    # Sort and truncate
    sorted_results = sorted(scored, key=lambda x: x["final_score"], reverse=True)[:top_k]

    # Compose context string
    context_string = "\n".join([f"{i + 1}. {r['text_chunk']}" for i, r in enumerate(sorted_results)])

    return sorted_results, context_string


# ========= Usage Example =========
results, context = hybrid_search(
    image_path="/home/ytlin/DATASET/images/Sunnyside North Beach.png",
    prompt="<image>Please provide detailed descriptions for this remote sensing image.",
    task_type="vqa",  # Options: "vqa", "caption", or "classification"
    alpha=0.9,
    top_k=1
)

# 1. Print search results
print("===== Retrieved Results (sorted by final score) =====")
for i, res in enumerate(results, 1):
    print(f"\n🔍 Result {i}:")
    print(f"📍 Image Path: {res['image_path']}")
    print(f"📊 Final Score: {res['final_score']:.4f}")
    print(f"🖼️ Image Similarity: {res['image_score']:.4f}")
    print(f"📝 Text Similarity: {res['text_score']:.4f}")
    print(f"📄 Text Content:\n{res['text_chunk']}")
    print("-" * 50)

# 2. Print final context string
print("\n===== Generated Context =====")
print(context)


