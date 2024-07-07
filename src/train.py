import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from src.config import Config
from src.data_loader import load_math_dataset
from src.model import load_model_and_tokenizer, get_lora_config
from src.utils import setup_logging

def main():
    setup_logging()
    
    dataset = load_math_dataset(Config.DATASET_NAME)
    model, tokenizer = load_model_and_tokenizer(Config)
    peft_config = get_lora_config(Config)

    training_arguments = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        optim=Config.OPTIM,
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fp16=Config.get_compute_dtype() == torch.float16,
        bf16=Config.get_compute_dtype() == torch.bfloat16,
        max_grad_norm=Config.MAX_GRAD_NORM,
        max_steps=Config.MAX_STEPS,
        warmup_ratio=Config.WARMUP_RATIO,
        group_by_length=Config.GROUP_BY_LENGTH,
        lr_scheduler_type=Config.LR_SCHEDULER_TYPE,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=Config.PACKING,
    )

    trainer.train()
    trainer.model.save_pretrained(Config.NEW_MODEL_NAME)

if __name__ == "__main__":
    main()