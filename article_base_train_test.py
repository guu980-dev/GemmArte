import os, time, math
import pandas as pd
from datasets import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
import torch
from PIL import Image
from peft import get_peft_model, LoraConfig

# Function to load custom dataset from CSV
def load_custom_dataset_from_csv(csv_file, image_folder):
    # Load CSV data using pandas
    data = pd.read_csv(csv_file)
    
    # Prepare dataset format for Hugging Face
    questions = data['question'].tolist()
    images = [os.path.join(image_folder, img) for img in data['image'].tolist()]
    answers = data['answer'].tolist()
    
    # Create a Hugging Face dataset from the loaded CSV
    return Dataset.from_dict({
        'question': questions,
        'image': images,
        'answer': answers
    })

# Main training function
def main():
    # Load custom datasets
    dataset = load_custom_dataset_from_csv('dataset/train_samples.csv', 'dataset/images/train')
    train_val_split = dataset.train_test_split(test_size=0.1)
    
    train_ds = train_val_split['train']
    val_ds = train_val_split['test']

    model_id = "google/paligemma-3b-pt-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    device = "cuda"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
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
        output_dir=f"./output/{math.floor(time.time())}",
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
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False
    )

    # Custom collate function
    def collate_fn(examples):
        texts = [example["question"] for example in examples]
        labels = [example['answer'] for example in examples]
        images = [Image.open(example['image']).convert("RGB") for example in examples]
        tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
        tokens = tokens.to(torch.bfloat16).to(device)
        return tokens
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
    )
    
    trainer.train()

if __name__ == "__main__":
    main()