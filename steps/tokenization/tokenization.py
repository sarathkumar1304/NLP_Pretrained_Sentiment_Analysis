from utils.logger import logger
from transformers import PreTrainedTokenizerBase,DistilBertTokenizer
from typing import Annotated
from datasets import DatasetDict
from zenml import step



@step(enable_cache=True)
def tokenization_step(
    dataset: DatasetDict,
    tokenizer:PreTrainedTokenizerBase,
    padding: bool = True,
    max_length: int = 512,
    text_column: str = "text",
    label_column: str = "label",
) -> Annotated[DatasetDict,"tokenized_data"]:
    """
    Tokenizes a dataset using a specified tokenizer.
    
    Args:
        dataset (DatasetDict): The dataset to tokenize.
        tokenizer_name (str): The name of the tokenizer to load.
        padding (bool): Whether to pad sequences.
        max_length (int): The maximum sequence length.
        text_column (str): The column containing text data.
        label_column (str): The column containing labels.
    
    Returns:
        DatasetDict: The tokenized dataset.
    """
    from transformers import AutoTokenizer

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    def preprocess_function(examples):
        result = tokenizer(
            examples[text_column],
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        result["label"] = examples[label_column]
        return result

    # Tokenize the dataset
    logger.info("Starting tokenization...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Log dataset size
    logger.info(f"Tokenized dataset size: {len(tokenized_datasets)}")

    # Remove unwanted columns and set format
    tokenized_datasets = tokenized_datasets.remove_columns([text_column])
    tokenized_datasets = tokenized_datasets.rename_column(label_column, "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets
