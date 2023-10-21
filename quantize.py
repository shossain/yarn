import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset

os.environ['TRANSFORMERS_CACHE'] = '../../cache/'


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = load_dataset(args.dataset, split="train", cache_dir='../../cache/data/')
    dataset = [text for text in dataset["text"]]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # Set quantization configuration
    quantization_config = GPTQConfig(
        bits=4,
        group_size=32,
        dataset=dataset,
        desc_act=True,
        tokenizer=tokenizer,
        model_seqlen = 16 * 1024 # might not be needed
    )
    # Load the model from HF
    quant_model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config, 
        #device_map='auto'
    )#.to("cuda")

    quant_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--model", type=str)
    args.add_argument("--dataset", type=str)
    main(args.parse_args())
