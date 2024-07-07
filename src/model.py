import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel

def load_model_and_tokenizer(config):
    """
    Load the model and tokenizer with the specified configuration.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=config.get_compute_dtype(),
        bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map=config.DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def get_lora_config(config):
    """
    Get the LoRA configuration.
    """
    return LoraConfig(
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        r=config.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_finetuned_model(base_model_name, finetuned_model_name, device_map):
    """
    Load the fine-tuned model.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, finetuned_model_name)
    return model.merge_and_unload()