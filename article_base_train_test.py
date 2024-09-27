import os
import json
from datasets import load_dataset, Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
import torch
from PIL import Image
from peft import get_peft_model, LoraConfig

# Function to load custom dataset
def load_custom_dataset(json_file, image_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Prepare dataset format for Hugging Face
    questions = []
    images = []
    answers = []
    multiple_choice_answers = []

    for item in data:
        questions.append(item['question'])
        images.append(os.path.join(image_folder, item['image_id']))
        answers.append(item['answer'])
        multiple_choice_answers.append(item['multiple_choice_answer'])
    
    return Dataset.from_dict({
        'question': questions,
        'image': images,
        'answer': answers,
        'multiple_choice_answer': multiple_choice_answers
    })

# Main training function
def main():
    # Load custom dataset
    train_ds = load_custom_dataset('dataset/train.json', 'dataset/images/train')
    val_ds = load_custom_dataset('dataset/val.json', 'dataset/images/val')
    
    model_id = "google/paligemma-3b-pt-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    device = "cuda"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
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
        num_train_epochs=2,
        remove_unused_columns=False,
        per_device_train_batch_size=16,
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
        texts = ["answer " + example["question"] for example in examples]
        labels = [example['multiple_choice_answer'] for example in examples]
        images = [Image.open(image_path).convert("RGB") for image_path in examples['image']]
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