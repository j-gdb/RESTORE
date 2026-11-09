set -e

python run.py --model huggyllama/llama-7b --data_path data/train_independent.json --test_data_path data/test_independent.json
python run.py --model mistralai/Mistral-7B-v0.1 --data_path data/train_independent.json --test_data_path data/test_independent.json
python run.py --model huggyllama/llama-7b --data_path data/train_dependent.json --test_data_path data/test_dependent.json
python run.py --model mistralai/Mistral-7B-v0.1 --data_path data/train_dependent.json --test_data_path data/test_dependent.json