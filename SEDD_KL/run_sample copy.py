import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dataset", default="recipe", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        i = i.split(tokenizer.eos_token, 1)[0].strip()
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()
