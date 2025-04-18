def clean_text(text):
    """Clean and preprocess text data"""
    import re
    import string
    from nltk.corpus import stopwords
    import nltk
    
    # Ensure nltk data is downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text