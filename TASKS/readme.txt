swift_answer_backup.py & utils_backup.py 

快捷命令
conda activate swift-rag
cd /home/ylin/NEW_RS_RAG/TASKS

caption
hefei-2
python swift_answer_backup_copy_caption_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.9 --gpu 1 --top_k 1
python swift_answer_backup_copy_caption_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.9 --gpu 1 --top_k 3
python swift_answer_backup_copy_caption_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.9 --gpu 2 --top_k 5

hefei-3
python swift_answer_backup_copy_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.3 --gpu 2 --top_k 1
python swift_answer_backup_copy_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.5 --gpu 3 --top_k 1
python swift_answer_backup_copy_caption.py --model qwen --task caption --dataset RSWK_caption --weight 0.7 --gpu 3 --top_k 1


qdrant数据库名字说明 : 
hefei    ---> RSWK_collection_hefei_1
         ---> RSWK_collection_hefei_2_id_first == RSWK_collection_hefei_3_id_first（完全一致）

shenzhen ---> RSWK_collection_shenzhen_1 
              RSWK_collection_shenzhen_2 
              RSWK_collection_shenzhen_3_classification

note:
1. 确认 utils_backup.py 51行 hybrid_search 函数中的数据库名字
2. classification 任务的专用搜索函数和数据库均在 shenzhen
