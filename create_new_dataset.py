import json
import random
import urllib.request
from pathlib import Path

SEED = 67
SAMPLES_PER_SOURCE = 1000
OUTPUT_DIR = Path("data")

URLS = [
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",
    # "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
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

def sample_source_datasets(i):
    sampled = []

    url = URLS[i]
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

    count = min(SAMPLES_PER_SOURCE, len(valid)) # we don't need the entire dataset, so I use up to SAMPLES_PER_SOURCE
    chosen = rng.sample(valid, count)
    sampled.extend(chosen)
    print(f"Sampled {count} / {len(valid)} from {url}")

    # shuffle so examples from different sources are mixed.
    rng.shuffle(sampled)
    return sampled

def create_independent(examples):
    templates = [
        "The following literal string appears in the text: {token}",
        "For reference, the document contains `{token}`.",
        "This note includes the exact character sequence {token}.",
        "Additional text: {token}",
        "A literal token mentioned in the source is {token}.",
        "Please note that the text contains {token} as ordinary text.",
        "The source includes this marker as text: {token}",
        "Unrelated note: the string {token} appears here.",
    ]
    results = []
    for i, example in enumerate(examples):
        token = TOKENS[i % len(TOKENS)]
        modified = dict(example)
        addition = rng.choice(templates).format(token=token)
        modified["input"] = insert_at_random_position(modified["input"],addition)
        # The original output remains unchanged.
        results.append(modified)
    return results

def create_dependent(examples): 
    instruction_templates = [
        "Also reproduce the exact literal token provided in the input. Append it on a new line using the format Token: followed by the token.",
        "In addition to answering the question, copy the exact token from the input into your response on a separate line.",
        "Your answer must include the exact token shown in the input. Preserve every character.",
        "After completing the original task, reproduce the literal token from the input exactly.",
        "Include the exact token from the input at the end of your answer. Do not modify it.",
    ]
    input_templates = [
        "The literal token to copy is: {token}",
        "Token reference: {token}",
        "Copy this exact string: {token}",
        "Required literal text: {token}",
        "The following token must be preserved exactly: {token}",
        "For exact reproduction, use this token: {token}",
    ]

    results = []

    for i, example in enumerate(examples):
        token = TOKENS[i % len(TOKENS)]
        modified = dict(example)
        instruction_addition = rng.choice(instruction_templates)
        input_addition = rng.choice(input_templates).format(token=token)

        modified["instruction"] = (modified["instruction"] + "\n" + instruction_addition)
        modified["input"] = insert_at_random_position(modified["input"],input_addition)
        # The expected output explicitly depends on the input token.
        modified["output"] = (modified["output"]+ f"\nToken: {token}")

        results.append(modified)

    return results

def save_dataset(filename, records):
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} examples to {path}")

def load_local_dataset(filename):
    with (OUTPUT_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

original_train = load_local_dataset("alpaca_data_cleaned.json")
original_test = load_local_dataset("davinci_003_outputs.json")

train_samples = sample_source_datasets(0)
test_samples = sample_source_datasets(1)
print(f"Training source examples: {len(train_samples)}")
print(f"Testing source examples: {len(test_samples)}")

train_independent = create_independent(train_samples)
train_dependent = create_dependent(train_samples)
test_independent = create_independent(test_samples)
test_dependent = create_dependent(test_samples)

save_dataset("train_independent.json", original_train + train_independent)
save_dataset("train_dependent.json", original_train + train_dependent)
save_dataset("test_independent.json", original_test + test_independent)
save_dataset("test_dependent.json", original_test + test_dependent)