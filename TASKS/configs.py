############ Dataset Settings ############
ChatEarthNet_sign = 'ChatEarthNet'
RSWK_sign = 'RSWK'
ChatEarthNet_file_path = '/home/ylin/LYT-RSRAG/ChatEarthnet/json_files/converted/ChatEarthNet_caps_4v_test.jsonl'       # ChatEarthNet caption task (test)
RSWK_file_path_caption = '/home/ylin/LYT-RSRAG/RSGPT_dataset/json/caption-test-hou-748.jsonl'                           # RSWK caption task (test)
RSWK_file_path_vqa = '/home/ylin/LYT-RSRAG/RSGPT_dataset/task/vqa/merged_shuffled_test_half-vqa_no_task.jsonl'          # RSWK VQA task (test)
RSWK_file_path_classification = '/home/ylin/LYT-RSRAG/RSGPT_dataset/task/classification/test-classification.jsonl'      # RSWK classification task (test)

###
batch_size = 8

############ Model Checkpoint Paths ############
# CLIP
clip_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/clip-vit-base-patch32'

# GeoChat
geochat_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/geochat-7B'
# geochat_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/GeoChat-7B/v2-20250611-104704/checkpoint-3000' # for ChatEarthNet
geochat_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/GeoChat-7B/v5-20250709-214753/checkpoint-10250' # for RSWK

# InternVL
intervl_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/InternVL2_5-8B'
# intervl_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/InternVL2.5-8B/v3-20250611-103445/checkpoint-750' # for ChatEarthNet
intervl_ft_path = '/home/ylin/NEW RS RAG/vMs/fine tune weights/Internv2.5-8B/v62-20258788-235455/checkpoint-7050' # for RSWK

# LLaMA
llama_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/Llama-3.2-11B-Vision-Instruct'
# llama_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Llama-3.2-11B-Vision-Instruct/v1-20250610-222511/checkpoint-750' # for ChatEarthNet
llama_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Llama-3.2-11B-Vision-Instruct/v2-20250708-235435/checkpoint-7400' # for RSWK

# Qwen
qwen_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/Qwen2.5-VL-7B-Instruct'
qwen_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Qwen2.5-VL-7B-Instruct/v0-20250611-092916/checkpoint-3000' # for ChatEarthNet
# qwen_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Qwen2.5-VL-7B-Instruct/v4-20250708-235323/checkpoint-9600' # for RSWK

# Janus
janus_base_path = '/home/ylin/NEW_RS_RAG/VLMS/pretrained_weights/Janus-Pro-7B'
# janus_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Janus_Pro_7B/v0-20250610-222205/checkpoint-1125' # for ChatEarthNet
janus_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/Janus_Pro_7B/v21-20250708-235335/checkpoint-9550' # for RSWK

# RSGPT
rsgpt_cfg_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune/RSGPT-main/eval_configs/rsgpt_eval.yaml' # fixed path
# rsgpt_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/RSGPT/rsgpt_lora_epoch_3' # for ChatEarthNet
rsgpt_ft_path = '/home/ylin/NEW_RS_RAG/VLMS/fine_tune_weights/RSGPT/20250712_rsgpt_lora_epoch_3' # for RSWK

############ Generation Parameters ############
max_tokens_caption = 1536  # If using 'ChatEarthNet' dataset, it is recommended to set to 500 due to short ground truths; for 'RSWK', set to 2048
temperature_caption = 0.7

max_tokens_vqa = 500
temperature_vqa = 0.5

max_tokens_classification = 100
temperature_classification = 0.1

############ Output Path Settings ############
caption_answer_path = 'answer/model_answers_caption'
vqa_answer_path = 'answer/model_answers_vqa'
classification_answer_path = 'answer/model_answers_classification'
