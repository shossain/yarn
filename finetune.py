import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import argparse
import os
import gc
os.environ['TRANSFORMERS_CACHE'] = '../../cache/'
import wandb
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from tqdm import tqdm
from transformers import set_seed, default_data_collator
import shutil

# from deepspeed.accelerator import get_accelerator


from scaled_rope.modeling_llama_together_yarn import LlamaForCausalLM
from scaled_rope.configuration_llama import LlamaConfig


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout]
    )

    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "yarn",
    )
    accelerator.print(args)
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    config = LlamaConfig.from_pretrained(args.model)
    config.rope_scaling = {
        "type": args.scaling_type,
        "factor": args.scaling_factor,
        "original_max_position_embeddings": 4096
    }
    config.rope_theta = args.rope_theta
    config.max_position_embeddings = int(args.scaling_factor * 4096)

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        config=config
    )

    train_datasets = []
    for dataset in args.datasets.split(","):
        try:
            train_dataset = load_dataset(dataset, split="train", cache_dir='../../cache/data/')
        except:
            train_dataset = load_from_disk(dataset)
        train_datasets.append(train_dataset)
    train_dataset = concatenate_datasets(train_datasets)

    if args.truncate:
        def truncate(sample):
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            sample["labels"] = sample["labels"][0:args.truncate]
            sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        train_dataset = train_dataset.map(truncate, desc="Truncating", num_proc=32)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size
    )

    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        target_modules = find_all_linear_names(model)
        #target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        # target_modules = ['q_proj', 'v_proj']
        accelerator.print(f"LoRA target modules: {target_modules}")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules=target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
    model, optim, train_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, scheduler
    )


    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))


    if not args.lora:
        model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = (
            int(training_difference.replace("step_", ""))
        )

    full_train_loader = train_loader
    if args.resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, len(train_dataset) % resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    model.train()

    while completed_steps < args.max_train_steps:
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.log({"loss": loss.item()}, step=completed_steps)
                    if isinstance(args.grad_norm, float):
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.grad_norm)

                optim.step()
                scheduler.step()
                optim.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                # get_accelerator().empty_cache()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                    if completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                        previous_dir = f"step_{completed_steps-args.checkpointing_steps}"
                        previous_dir = os.path.join(args.output_dir, previous_dir)
                        try:
                            shutil.rmtree(previous_dir)
                        except OSError as e:
                            accelerator.print("Error: %s : %s" % (previous_dir, e.strerror))

            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            if completed_steps >= args.max_train_steps:
                break

        train_loader = full_train_loader

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    accelerator.print(f"Saving model to {args.output_dir}")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", action="store_true")
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", action="store_true")
    args.add_argument("--lora", action="store_true")
    args.add_argument("--lora_r", type=int, default=16)
    args.add_argument("--lora_alpha", type=int, default=64)
    args.add_argument("--model", type=str,
                      default="NousResearch/Llama-2-7b-hf")
    args.add_argument("--scaling-factor", type=float, default=16.0)
    args.add_argument("--scaling-type", type=str)
    args.add_argument("--rope-theta", type=float, default=10000.0)
    args.add_argument("--truncate", type=int)
    args.add_argument("--datasets", type=str,
                      default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    main(args.parse_args())
