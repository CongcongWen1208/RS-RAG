import os
import json
from uuid import uuid4
from tqdm import tqdm
from PIL import Image
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Batch

# -----------------------------
# Configuration Parameters
# -----------------------------
path                 = "/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/clip-vit-base-patch32"  # Path to CLIP model weights
jsonl_path           = ""                                                                      # Path to ChatEarthNet dataset (.jsonl)
device               = "cuda:0"                                                                # Set device
qdrant_dataset_name  = "Test_dataset_ChatEarthNet"                                             # Qdrant collection name
batch_size           = 64                                                                      # Upload batch size
qdrant_name          = "localhost"                                                             # Qdrant host (default: localhost)
qdrant_port_number   = 6333                                                                    # Qdrant port (default: 6333)
vector_dimension     = 512                                                                     # Dimension of embedded vectors

# -----------------------------
# Load CLIP Model & Processor
# -----------------------------
try:
    clip_model = CLIPModel.from_pretrained(path).to(device)
    processor = CLIPProcessor.from_pretrained(path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)

# -----------------------------
# Multimodal Embedding Functions
# -----------------------------
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_feat = clip_model.get_image_features(**image_inputs)
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    return image_feat.detach().cpu().numpy()[0]

def get_text_embedding(text):
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.detach().cpu().numpy()[0]

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
# Load and Process Data
# -----------------------------
image_points = []
text_points = []
unified_points = []  # <-- Make sure this is initialized

with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(tqdm(lines)):
    try:
        data = json.loads(line)
        image_path = data["images"][0]
        full_text = next((msg["content"] for msg in data["messages"] if msg["role"] == "assistant"), "")

        image_vector = get_image_embedding(image_path)
        text_vector = get_text_embedding(full_text)

        point = PointStruct(
            id=str(uuid4()),
            vector=image_vector,
            payload={
                "image_path": image_path,
                "image_vector": image_vector.tolist(),
                "chunk_type": "caption"
                "text": full_text,
                "text_vector": text_vector.tolist(),
            }
        )
        unified_points.append(point)

    except Exception as e:
        print(f"❌ Failed to process idx={idx}: {e}")

# -----------------------------
# Batch Upload to Qdrant
# -----------------------------

print(f"🚀 Uploading {len(unified_points)} points to collection '{qdrant_dataset_name}'...")

for i in range(0, len(unified_points), batch_size):
    batch = unified_points[i:i + batch_size]
    client.upsert(collection_name=qdrant_dataset_name, points=batch)
    print(f"✅ Batch {i // batch_size + 1} uploaded successfully.")

print(f"🎉 Unified collection construction completed: {qdrant_dataset_name}")
