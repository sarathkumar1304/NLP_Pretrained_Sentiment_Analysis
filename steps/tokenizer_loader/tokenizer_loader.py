from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,PreTrainedTokenizer
from typing import Annotated
from zenml import step
from utils.logger import logger

@step(enable_cache=True)
def tokenizer_loader(lower_case:bool)->Annotated[PreTrainedTokenizer,"tokenizer"]:
    """
    Loads a pre-trained tokenizer.

    Args:
        lower_case (bool): Whether to convert the input text to lower case.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """
    logger.info("tokenizer loader initatied successfully")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased",do_lower_case=lower_case)
    logger.info("Tokenizer loader completed")
    return tokenizer
