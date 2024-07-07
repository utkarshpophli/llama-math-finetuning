from datasets import load_dataset

def load_math_dataset(dataset_name, split="train[:20%]"):
    """
    Load the mathematics dataset.
    """
    return load_dataset(dataset_name, split=split)