import argparse
import collections
import json
import os
import collision

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_src", default="data/alpaca_data_cleaned.json")
    p.add_argument("--test_src", default="data/davinci_003_outputs.json")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=0, help="0 = all test examples that have an input")
    p.add_argument("--token_sets", nargs="+", default=list(collision.TOKEN_SETS))
    p.add_argument("--out_dir", default="data/collision")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # separate the training and testing data into groups of data source and number of entries
    for split, src, n in [("train", args.train_src, args.n_train), ("test", args.test_src, args.n_test)]: 
        if not os.path.exists(src):
            print(f"skip {split}: {src} not found (run setup.py to download it)")
            continue
        data = json.load(open(src))
        if n == 0: 
            n = sum(1 for x in data if x.get("input", "").strip()) # if 0, we use a default of the number of inputs
        for ts in args.token_sets:
            out = collision.generate(data, split, ts, n, seed=args.seed) # generate the dataset
            path = os.path.join(args.out_dir, f"{split}_{ts}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=True) # dump the examples
            tasks = collections.Counter(x["collision"]["task"] for x in out)
            print(f"{path}: {len(out)} examples  {dict(tasks)}")
            if split == "test": # the delimiter question is the only task
                sa_path = os.path.join(args.out_dir, f"test_{ts}_standalone.json")
                with open(sa_path, "w") as f:
                    json.dump([collision.to_standalone(x) for x in out], f, indent=2, ensure_ascii=True)
                print(f"{sa_path}: {len(out)} examples (standalone)")

if __name__ == "__main__":
    main()