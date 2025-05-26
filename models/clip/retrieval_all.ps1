python .\eval_retrieval.py --model "openai/clip-vit-base-patch32" --text-encoder .\models\openai\clip_b32\text\model.onnx --image-encoder .\models\openai\clip_b32\image\model.onnx --dataset "nlphuji/flickr_1k_test_image_text_retrieval"

python .\eval_retrieval.py --model "openai/clip-vit-base-patch16" --text-encoder .\models\openai\clip_b16\text\model.onnx --image-encoder .\models\openai\clip_b16\image\model.onnx --dataset "nlphuji/flickr_1k_test_image_text_retrieval"

python .\eval_retrieval.py --model "openai/clip-vit-base-patch32" --tokenizer "sentence-transformers/clip-ViT-B-32-multilingual-v1" --text-encoder .\models\sbert\clip_b32\distilbert\model.onnx --image-encoder .\models\sbert\clip_b32\image\model.onnx --dataset "nlphuji/flickr_1k_test_image_text_retrieval"

python .\eval_retrieval.py --model "laion/clip-vit-b-32-laion2b-s34b-b79k" --text-encoder .\models\laion\clip_b32\text\model.onnx --image-encoder .\models\laion\clip_b32\image\model.onnx --dataset "nlphuji/flickr_1k_test_image_text_retrieval"
