import torch

class Config:
    MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
    DATASET_NAME = "utkarshpophli/mathematics_dataset"
    NEW_MODEL_NAME = "llama-2-7b-math-finetune"

    # QLoRA parameters
    LORA_R = 64
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # Training parameters
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = False
    OUTPUT_DIR = "./results"
    NUM_TRAIN_EPOCHS = 1
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM = 0.3
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.001
    OPTIM = "paged_adamw_32bit"
    LR_SCHEDULER_TYPE = "cosine"
    MAX_STEPS = -1
    WARMUP_RATIO = 0.03
    GROUP_BY_LENGTH = True
    SAVE_STEPS = 0
    LOGGING_STEPS = 25

    # SFT parameters
    MAX_SEQ_LENGTH = None
    PACKING = False
    DEVICE_MAP = {"": 0}

    @staticmethod
    def get_compute_dtype():
        return getattr(torch, Config.BNB_4BIT_COMPUTE_DTYPE)