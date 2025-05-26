python ..\..\..\utils\olive_eval.py --target qnn --config .\openai_clip_vision_b32_qdq.json --evaluator sanity_check
python ..\..\..\utils\olive_eval.py --target qnn --config .\openai_clip_text_b32_qdq.json --evaluator sanity_check

python ..\..\..\utils\olive_eval.py --target qnn --config .\openai_clip_vision_b16_qdq.json --evaluator sanity_check
python ..\..\..\utils\olive_eval.py --target qnn --config .\openai_clip_text_b16_qdq.json --evaluator sanity_check

python ..\..\..\utils\olive_eval.py --target qnn --config .\laion_clip_vision_b32_qdq.json --evaluator sanity_check
python ..\..\..\utils\olive_eval.py --target qnn --config .\laion_clip_text_b32_qdq.json --evaluator sanity_check

python ..\..\..\utils\olive_eval.py --target qnn --config .\sbert_clip_vision_b32_qdq.json --evaluator sanity_check
python ..\..\..\utils\olive_eval.py --target qnn --config .\sbert_clip_text_b32_qdq.json --evaluator sanity_check
python ..\..\..\utils\olive_eval.py --target qnn --config .\sbert_clip_text_b32_distilbert_qdq.json --evaluator sanity_check
