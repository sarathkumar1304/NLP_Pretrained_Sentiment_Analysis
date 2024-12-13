import pandas as pd
from utils.logger import logger
from typing import Annotated
from datasets import load_dataset, DatasetDict, Dataset
from zenml import step


@step
def data_ingestion() -> Annotated[DatasetDict, "dataset"]:

    """
    Load the IMDB sentiment analysis dataset and create a balanced dataset with 10 samples each for positive and negative labels.

    Returns:
        Annotated[DatasetDict, "dataset"]: A balanced dataset with 20 samples, 10 each of positive and negative labels.
    """
    logger.info("Data ingestion started")
    dataset = load_dataset("imdb", trust_remote_code=True)
    
    # Separate the dataset into two subsets: label 0 and label 1
    label_0_data = [example for example in dataset['train'] if example['label'] == 0]
    label_1_data = [example for example in dataset['train'] if example['label'] == 1]
    
    # Ensure at least 10 samples for each label
    label_0_subset = label_0_data[:10]
    label_1_subset = label_1_data[:10]
    
    # Combine subsets into a balanced dataset
    balanced_data = label_0_subset + label_1_subset
    
    # Shuffle the data for randomness
    from random import shuffle
    shuffle(balanced_data)
    
    # Convert to a `Dataset` object
    balanced_dataset = Dataset.from_pandas(pd.DataFrame(balanced_data))
    
    # Wrap the dataset in a `DatasetDict`
    result_dataset = DatasetDict({"train": balanced_dataset})
    logger.info(result_dataset)

    
    logger.info(f"Balanced dataset created with {len(balanced_dataset)} samples.")
    # logger.info(balanced_dataset[0])  # Log the first sample
    return result_dataset


    

            
            

