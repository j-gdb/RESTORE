import argparse
import json
import os
import shutil
import sys
import transformers
from config import SPECIAL_DELM_SETS

TOKENIZER_FILES = ["added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]

# replace the mappings inside the json tree
def remap_json(obj, mapping):
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, list):
        return [remap_json(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {mapping.get(k, k): remap_json(v, mapping) for k, v in obj.items()}
    return obj

def remap_dir(src, dst, mapping):
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists; remove it first")
    os.makedirs(dst)
    found = set()
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in TOKENIZER_FILES:
            obj = json.load(open(s))
            text = json.dumps(obj)
            found |= {t for t in mapping if json.dumps(t)[1:-1] in text}
            json.dump(remap_json(obj, mapping), open(d, "w"), indent=2, ensure_ascii=True)
        elif os.path.isdir(s):
            continue
        else:
            os.symlink(os.path.abspath(s), d) # weights, tokenizer.model, configs are shared
    missing = set(mapping) - found # check if we did not map something that we should have
    if missing:
        shutil.rmtree(dst)
        sys.exit(f"{src} has no tokenizer entries for {sorted(missing)}; is it a StruQ checkpoint?")

def verify(src, dst, old, new):
    t_old = transformers.AutoTokenizer.from_pretrained(src, use_fast=False)
    t_new = transformers.AutoTokenizer.from_pretrained(dst, use_fast=False)
    for o, n in zip(old, new):
        io, in_ = t_old.convert_tokens_to_ids(o), t_new.convert_tokens_to_ids(n)
        assert io == in_ and io != t_old.unk_token_id, f"{o!r}->{io} but {n!r}->{in_}"
    prompt = "{m} {i}{c}\nSay hi.\n\n{m} {p}{c}\nSome data.\n\n{m} {r}{c}\n"
    p_old = prompt.format(i=old[0], p=old[1], r=old[2], m=old[3], c=old[4])
    p_new = prompt.format(i=new[0], p=new[1], r=new[2], m=new[3], c=new[4])
    ids_old, ids_new = t_old(p_old).input_ids, t_new(p_new).input_ids
    assert ids_old == ids_new, f"token IDs differ:\n{ids_old}\n{ids_new}"
    assert t_new.convert_tokens_to_ids(old[0]) == t_new.unk_token_id
    assert not set(t_new("[INST] text").input_ids) & {t_old.convert_tokens_to_ids(t) for t in old}
    # decoding gives the control characters back
    assert t_new.decode(ids_new, skip_special_tokens=False).count(new[0]) == 1
    print("verified: control-character prompt gives identical token IDs; [INST] is now plain text")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--from_set", default="original")
    p.add_argument("--to_set", default="control")
    a = p.parse_args(argv)
    sets = SPECIAL_DELM_SETS
    old, new = sets[a.from_set], sets[a.to_set]
    mapping = dict(zip(old, new))
    remap_dir(a.src, a.dst, mapping)
    print(f"wrote {a.dst}: " + ", ".join(f"{o} -> {n!r}" for o, n in mapping.items()))
    verify(a.src, a.dst, old, new)

if __name__ == "__main__":
    main()
