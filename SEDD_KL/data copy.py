from datasets import load_dataset
from itertools import chain
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

def recipe_tokenizer(example):
    if "input" in example:
        return {"text": example["input"]}

    titles = example.get("title", [""] * len(example["ingredients"]))
    ingredients = example["ingredients"]
    directions = example["directions"]

    texts = []
    for title, ings, dirs in zip(titles, ingredients, directions):
        if isinstance(ings, list):
            ings = "\n- ".join(ings)
        if isinstance(dirs, list):
            dirs = "\n- ".join(dirs)
        texts.append(f"{title}\n\nIngredients:\n- {ings}\n\nDirections:\n- {dirs}")

    return {"text": texts}

def recipe_detokenizer(text):
    parts = text.split("\n\nIngredients:\n")
    if len(parts) < 2:
        return text
    title = parts[0].strip()
    rest = parts[1].split("\n\nDirections:\n")
    ingredients = rest[0].strip()
    directions = rest[1].strip() if len(rest) > 1 else ""
    return f"Title: {title}\n\nIngredients:\n{ingredients}\n\nDirections:\n{directions}"

def get_recipe_dataset(cache_dir=None):
    ds = load_dataset("corbt/all-recipes", cache_dir=cache_dir)
    ds = ds.map(recipe_tokenizer, batched=True, remove_columns=ds["train"].column_names)
    return ds

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name != "recipe":
        raise ValueError("This fork is configured for the recipe dataset only; use data.train=recipe and data.valid=recipe.")

    dataset = get_recipe_dataset(cache_dir=cache_dir)
    if mode not in dataset:
        if mode == "validation" and "train" in dataset:
            mode = "train"
        else:
            raise ValueError(f"Recipe dataset split {mode!r} is not available. Available splits: {list(dataset.keys())}")

    data = dataset[mode]
    detokenizer = None
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if "text" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader
