# Script to Convert Two Datasets (ChatEarthNet and RSWK) into Qdrant Vector Database Format

## How to Run:
###        python build_qdrant_dataset_ChatEarthNet.py
###        python build_qdrant_dataset_RSWK.py

## Description:
### - Specify the multimodal embedding model to use; by default, it uses `clip-vit-base-patch32`.
### - Set the path to the `.json` file (i.e., the dataset path). If you replace it with your own dataset,
      make sure the field names and structure match the format expected by this script.
### - Set the device to use (GPU index).
### - Set the name of the dataset (i.e., the Qdrant collection name). 
      When creating a new dataset, ensure the name is not the same as an existing one,
      otherwise it will overwrite the previous dataset.
### - The default embedding vector dimension is 512. 
      If your dataset contains a large number of text entries or very long texts, consider increasing this dimension.
