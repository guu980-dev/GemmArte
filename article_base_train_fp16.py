import os, time, math
import pandas as pd
from datasets import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
import torch
from PIL import Image
from peft import get_peft_model, LoraConfig
import argparse


def load_custom_dataset_from_csv(csv_file, image_folder):
    data = pd.read_csv(csv_file)
    
    questions = data['question'].tolist()
    images = [os.path.join(image_folder, img) for img in data['image'].tolist()]
    answers = data['answer'].tolist()
    
    return Dataset.from_dict({
        'question': questions,
        'image': images,
        'answer': answers
    })


def load_custom_dataset_from_parquet(parquet_file, image_folder):
    data = pd.read_parquet(parquet_file)
    
    questions = data['question'].tolist()
    images = [os.path.join(image_folder, img) for img in data['image'].tolist()]
    answers = data['answer'].tolist()
    
    return Dataset.from_dict({
        'question': questions,
        'image': images,
        'answer': answers
    })


def load_dataset_by_type(metadata_type, dataset_dir, image_folder):
    if metadata_type == "csv":
        return load_custom_dataset_from_csv(
            os.path.join(dataset_dir, 'train_samples.csv'),
            image_folder
        )
    elif metadata_type == "parquet":
        return load_custom_dataset_from_parquet(
            os.path.join(dataset_dir, 'train.parquet'),
            image_folder
        )
    else:
        raise ValueError("Unsupported metadata type. Use 'csv' or 'parquet'.")


def load_model_and_args(use_qlora, model_id, device, output_dir):
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16  # Changed from bfloat16 to float16
        )
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"{math.floor(time.time())}"),
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            fp16=True,  # Changed from bf16 to fp16
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )
        
        return model, args
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)  # Changed from bfloat16 to float16
        for param in model.vision_tower.parameters():
            param.requires_grad = False

        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
        
        args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"{math.floor(time.time())}"),
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            logging_steps=100,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            fp16=True,  # Changed from bf16 to fp16
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

        return model, args


def main(args):
    dataset_dir = args.dataset_dir
    model_id = args.model_id
    output_dir = args.output_dir
    metadata_type = args.metadata_type
    
    dataset = load_dataset_by_type(metadata_type, dataset_dir, os.path.join(dataset_dir, 'images'))
    train_val_split = dataset.train_test_split(test_size=0.1)
    
    train_ds = train_val_split['train']
    val_ds = train_val_split['test']

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    device = "cuda"

    model, args = load_model_and_args(args.use_qlora, model_id, device, output_dir)

    def collate_fn(examples):
        texts = [example["question"] for example in examples]
        labels = [example['answer'] for example in examples]
        images = [Image.open(example['image']).convert("RGB") for example in examples]
        tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
        tokens = tokens.to(torch.float16).to(device)  # Changed from bfloat16 to float16
        return tokens
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
    )
    
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with custom dataset")
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Path to the folder containing the images')
    parser.add_argument('--model_id', type=str, default='google/paligemma-3b-pt-224', help='Model ID to use for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the output')
    parser.add_argument('--use_qlora', type=bool, default=False, help='Use QLoRA for training')
    parser.add_argument('--metadata_type', type=str, default='parquet', choices=['csv', 'parquet'], help='Metadata format (csv or parquet)')
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    main(args)