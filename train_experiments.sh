#!/usr/bin/env bash
set -euo pipefail

STRUQ=huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37
STRUQ_CTRL=${STRUQ}_ctrl
ATTACKS="naive ignore completion_real completion_other"

UTILITY=${UTILITY-none}

# Download model and data
python setup.py --model
[ -f data/collision/test_original.json ] || python create_new_dataset.py

# copy the data so we can have one set with the new delimiters
[ -d "$STRUQ_CTRL" ] || python remap_delimiters.py --src "$STRUQ" --dst "$STRUQ_CTRL"

# run experiments using the existing model for baselines
python test_collision.py --skip_existing -m "$STRUQ" --token_set original --frontend none --standalone
python test_collision.py --skip_existing -m "$STRUQ" --token_set original --frontend filter --standalone
python test_collision.py --skip_existing -m "$STRUQ" --token_set original --frontend escape --standalone

# run one with the new delimieters, but keep struqs existing frontend
python test_collision.py --skip_existing -m "$STRUQ_CTRL" -token_set control  --frontend filter --standalone
STRUQ_DELM=original python test.py -m "$STRUQ" -a $UTILITY $ATTACKS

# train models
train() { [ -f "$1/config.json" ] || python train_lora.py --model "$2" --token_set "$3" --frontend "$4" --output_dir "$1"; }
A=models/llama-7b_SpclSpclSpcl_NaiveCompletion_lora-original-filter
B=models/llama-7b_SpclSpclSpcl_NaiveCompletion_lora-original-escape
C=models/llama-7b_SpclSpclSpcl_NaiveCompletion_lora-control-escape
train "$A" "$STRUQ" original filter
train "$B" "$STRUQ" original escape
train "$C" "$STRUQ_CTRL" control  escape

# evaluations
for spec in "$A original filter" "$B original escape" "$C control filter" "$C control escape"; do
  set -- $spec
  python test_collision.py --skip_existing -m "$1" --token_set "$2" --frontend "$3"
  python test_collision.py --skip_existing -m "$1" --token_set "$2" --frontend "$3" --standalone
  python test_collision.py --skip_existing -m "$1" --token_set "$2" --frontend "$3" --attack ignore
  STRUQ_DELM=$2 python test.py -m "$1" --frontend "$3" -a $UTILITY $ATTACKS
done
echo "done: see <model>/collision/summary.tsv and <model>/summary.tsv"
