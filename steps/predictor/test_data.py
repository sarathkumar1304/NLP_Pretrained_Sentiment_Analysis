
from zenml import step
@step(enable_cache=False)
def get_text_data():
    """
    Returns a sample text data for sentiment analysis.
    
    This is a hard-coded sample data and should be replaced with a more robust
    data source.
    
    Returns:
        str: Sample text data for sentiment analysis.
    """
    data = "This is the one of the best movie"
    return data 