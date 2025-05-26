python .\eval_degrad.py --max-samples 1000 --encoder image --model-name "laion/clip-vit-b-32-laion2b-s34b-b79k" --model-path ".\models\laion\clip_b32\image\model.onnx"
python .\eval_degrad.py --max-samples 200 --encoder text --model-name "laion/clip-vit-b-32-laion2b-s34b-b79k" --model-path ".\models\laion\clip_b32\text\model.onnx"

python .\eval_degrad.py --max-samples 1000 --encoder image --model-name "openai/clip-vit-base-patch16" --model-path ".\models\openai\clip_b16\image\model.onnx"
python .\eval_degrad.py --max-samples 200 --encoder text --model-name "openai/clip-vit-base-patch16" --model-path ".\models\openai\clip_b16\text\model.onnx"

python .\eval_degrad.py --max-samples 1000 --encoder image --model-name "openai/clip-vit-base-patch32" --model-path ".\models\openai\clip_b32\image\model.onnx"
python .\eval_degrad.py --max-samples 200 --encoder text --model-name "openai/clip-vit-base-patch32" --model-path ".\models\openai\clip_b32\text\model.onnx"

python .\eval_degrad.py --max-samples 1000 --encoder image --model-name "openai/clip-vit-base-patch32" --model-path ".\models\sbert\clip_b32\image\model.onnx"
python .\eval_degrad.py --max-samples 200 --encoder text --model-name "sentence-transformers/clip-ViT-B-32-multilingual-v1" --model-path ".\models\sbert\clip_b32\distilbert\model\model.onnx"
