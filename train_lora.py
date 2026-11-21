import argparse
import math
import os
import json
import random
import sys
import inspect
import collision

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="StruQ checkpoint folder (remapped copy for --token_set control)")
    p.add_argument("--token_set", required=True, choices=["original", "control"])
    p.add_argument("--frontend", required=True, choices=["filter", "escape"])
    p.add_argument("--alpaca", default="data/alpaca_data_cleaned.json")
    p.add_argument("--collision", default=None, help="default: data/collision/train_<token_set>.json")
    p.add_argument("--n_replay", type=int, default=6000, help="original Alpaca examples mixed in")
    p.add_argument("--n_collision", type=int, default=0, help="0 = all collision examples")
    p.add_argument("--attack", default="NaiveCompletion", help="StruQ training attacks, as in run.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    # we should keep the remaining the same
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--save_steps", type=int, default=100, help="checkpointing")
    p.add_argument("--no_merge", action="store_true", help="save only the adapter")
    p.add_argument("--build_only", action="store_true", help="write train_mixture.json and stop (no torch needed)")
    return p.parse_args(argv)

def default_output_dir(a):
    return f"models/llama-7b_SpclSpclSpcl_{a.attack}_lora-{a.token_set}-{a.frontend}"

def build_mixture(alpaca, collision_data, token_set, frontend_mode, n_replay, n_collision=0, seed=0):
    rng = random.Random(seed)
    used = {ex["collision"]["source_index"] for ex in collision_data}
    pool = [i for i in range(len(alpaca)) if i not in used]
    if n_replay > len(pool): raise ValueError(f"n_replay={n_replay} > {len(pool)} available")
    replay = []
    for i in rng.sample(pool, n_replay):
        ex = dict(alpaca[i])
        ex["input"] = collision._process(ex.get("input", ""), frontend_mode, token_set)
        replay.append(ex)
    coll = collision_data[:n_collision] if n_collision else list(collision_data)
    coll = [collision.render(ex, frontend_mode, token_set) for ex in coll]
    mix = replay + coll
    rng.shuffle(mix)
    return mix



def main(argv=None):
    a = parse_args(argv)
    os.environ["STRUQ_DELM"] = a.token_set
    out = a.output_dir or default_output_dir(a)
    os.makedirs(out, exist_ok=True)

    coll_path = a.collision or f"data/collision/train_{a.token_set}.json"
    collision_data = json.load(open(coll_path))
    if collision_data[0]["collision"]["token_set"] != a.token_set:
        sys.exit(f"{coll_path} was built for {collision_data[0]['collision']['token_set']!r} delimiters")
    mix = build_mixture(json.load(open(a.alpaca)), collision_data, a.token_set, a.frontend,
                        a.n_replay, a.n_collision, a.seed)
    mix_path = os.path.join(out, "train_mixture.json")
    json.dump(mix, open(mix_path, "w"), indent=1, ensure_ascii=True)
    json.dump(vars(a) | {"output_dir": out, "n_examples": len(mix)}, open(os.path.join(out, "run_config.json"), "w"), indent=2)
    print(f"{len(mix)} training examples -> {mix_path}")
    if a.build_only: return
    if "Completion" in a.attack and not os.path.exists("data/alpaca_data.json"):
        sys.exit("StruQ's Completion attack needs data/alpaca_data.json (run setup.py)")

    import numpy as np, torch, transformers
    from peft import LoraConfig, get_peft_model
    import collision
    from struq import SupervisedDataset
    from train import DataCollatorForSupervisedDataset

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(a.model, use_fast=False, model_max_length=a.max_length,
                                                           padding_side="right")
    missing = [t for t in collision.TOKEN_SETS[a.token_set] if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing:
        hint = " Run remap_delimiters.py first." if a.token_set == "control" else ""
        sys.exit(f"{a.model} has no special token for {missing!r}.{hint}")
    if tokenizer.pad_token is None: sys.exit("tokenizer has no pad token; expected StruQ's [PAD]")

    dtype_kw = "dtype" if "dtype" in inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters else "torch_dtype"
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, **{dtype_kw: torch.bfloat16}).to("cuda")
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        sys.exit("embedding table is smaller than the tokenizer; wrong checkpoint?")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # StruQ's own pipeline: adds Naive/Completion-attacked copies and downsamples to the original size
    dataset = SupervisedDataset(data_path=mix_path, tokenizer=tokenizer, attack=f"SpclSpclSpcl_{a.attack}", downsample=True)
    # Warmup as a step count (works on every transformers version; warmup_ratio was removed in v5)
    total_steps = math.ceil(math.ceil(len(dataset) / (a.batch_size * a.grad_accum)) * a.epochs)
    targs = dict(
        output_dir=os.path.join(out, "trainer"), 
        num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        warmup_steps=max(1, round(0.03 * total_steps)),
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=a.seed,
        remove_unused_columns=False
    )
    accepted = inspect.signature(transformers.TrainingArguments.__init__).parameters
    dropped = sorted(k for k in targs if k not in accepted)
    if dropped: # keep going on newer/older versions, but say what was not applied
        print(f"note: this transformers version does not accept {dropped}; continuing without them")
    for k in ("output_dir", "num_train_epochs", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "save_steps"):
        if k in dropped:
            sys.exit(f"TrainingArguments no longer accepts {k!r}; please report this")
    args = transformers.TrainingArguments(**{k: v for k, v in targs.items() if k in accepted})
    print(f"{len(dataset)} sequences, {total_steps} optimizer steps, warmup {targs['warmup_steps']}")
    trainer = transformers.Trainer(model=model, args=args, train_dataset=dataset, data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer))
    # Re-running the same command after an interruption continues from the newest checkpoint.
    # The data is rebuilt identically (fixed seeds), so the resumed run sees the same batches.
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last: print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    json.dump(trainer.state.log_history, open(os.path.join(out, "log_history.json"), "w"), indent=1)

    model.save_pretrained(os.path.join(out, "adapter"))
    tokenizer.save_pretrained(os.path.join(out, "adapter"))
    if not a.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
