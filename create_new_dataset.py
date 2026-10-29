import json
import random
import urllib.request
from pathlib import Path

SEED = 67
SAMPLES_PER_SOURCE = 1000
OUTPUT_DIR = Path("data")

URLS = [
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json",
] # we ignore the yaml here for simplicity

TOKENS = ["[INST]", "[INPT]", "[RESP]", "[MARK]", "[COLN]", "##"] # struq tokens as seen in config.py
rng = random.Random(SEED)

def load_json_url(url):
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "Python dataset sampler"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize(example):
    instruction = example.get("instruction")
    input_text = example.get("input", "")
    output = example.get("output")

    if not isinstance(instruction, str) or not isinstance(output, str):
        return None
    if not isinstance(input_text, str):
        input_text = ""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }

def insert_at_random_position(text, addition):
    if not text:
        return addition

    position = rng.choice(["start", "middle", "end"])

    if position == "start":
        return addition + "\n" + text
    if position == "end":
        return text + "\n" + addition

    midpoint = len(text) // 2
    return text[:midpoint] + "\n" + addition + "\n" + text[midpoint:]

sampled = []

for url in URLS:
    data = load_json_url(url)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {url}")

    valid = []
    for item in data:
        if isinstance(item, dict):
            normalized = normalize(item)
            if normalized is not None:
                valid.append(normalized)

    if not valid:
        print(f"Warning: no usable records found at {url}")
        continue

    count = min(SAMPLES_PER_SOURCE, len(valid)) # we don't need the entire dataset, so I use up to SAMPLES_PER_SOURCE
    chosen = rng.sample(valid, count)
    sampled.extend(chosen)
    print(f"Sampled {count} / {len(valid)} from {url}")

# shuffle so examples from different sources are mixed.
rng.shuffle(sampled)

# we make 2 datasets, one the depends on the tokens and one that does not.
independent = []
dependent = []


independent_templates = [
    "The following literal string appears in the text: {token}",
    "For reference, the document contains `{token}`.",
    "This note includes the exact character sequence {token}.",
    "Additional text: {token}",
    "A literal token mentioned in the source is {token}.",
    "Please note that the text contains {token} as ordinary text.",
    "The source includes this marker as text: {token}",
    "Unrelated note: the string {token} appears here.",
]

dependent_instruction_templates = [
    "Also copy the literal token from the input exactly. "
    "Append it on a new line as `Token: {token}`.",
    "In addition to answering the question, reproduce the exact "
    "token shown in the input using the format `Token: {token}`.",
    "Your answer must include the exact token from the input on a "
    "separate line, formatted as `Token: {token}`.",
    "Preserve the token exactly as written in the input. "
    "Add `Token: {token}` after your answer.",
    "Report the literal token from the input without changing any "
    "characters. Use the format `Token: {token}`.",
]

dependent_input_templates = [
    "The literal token to copy is: {token}",
    "Token reference: {token}",
    "Copy this exact string: {token}",
    "Required literal text: {token}",
    "The following token must be preserved exactly: {token}",
    "For exact reproduction, use this token: {token}",
]

for i, example in enumerate(sampled):
    token = TOKENS[i % len(TOKENS)] # cycle for balanced representation

    # First, the added token is irrelevant to the original answer.
    a = dict(example)
    addition = rng.choice(independent_templates).format(token=token)
    a["input"] = insert_at_random_position(a["input"], addition)
    independent.append(a)

    # Second, the answer must preserve the token as well as answering the original task.
    b = dict(example)
    token_instruction = rng.choice(dependent_instruction_templates).format(token=token)
    token_input = rng.choice(dependent_input_templates).format(token=token)

    b["instruction"] = b["instruction"] + "\n" + token_instruction
    b["input"] = insert_at_random_position(b["input"], token_input)
    b["output"] = b["output"] + f"\nToken: {token}"
    dependent.append(b)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

outputs = {
    "delimiter_independent.json": independent,
    "delimiter_dependent.json": dependent,
}

for filename, records in outputs.items():
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} examples to {path}")