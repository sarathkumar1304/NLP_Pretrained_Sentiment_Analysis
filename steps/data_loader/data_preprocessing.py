import re
import emoji
from utils.logger import logger

def clean_text(text):
    """
    This function cleans a given text by converting it to lowercase, removing HTML tags and special characters, 
    and replacing any URLs with an empty string. It returns the cleaned text. If the input text is a pandas Series, 
    it will be applied element-wise. If the input is a string, it will be cleaned and returned as a string.

    Parameters
    ----------
    text : str or pandas Series
        The text to be cleaned.

    Returns
    -------
    str or pandas Series
        The cleaned text. If the input is a pandas Series, it will be a pandas Series. If the input is a string, it will be a string.
    """
    try:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = emoji.replace_emoji(text, replace="") 
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^\w\s@#]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
 
    # 1. Convert text to lowercase
        text = text.str.lower()
        
        # 2. Remove HTML tags
        text = text.str.replace(r'<.*?>', '', regex=True)
        
        # 3. Remove special characters
        text = text.str.replace(r'[^\w\s]', '', regex=True)
        
        return text




