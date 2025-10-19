from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def scoring(text):
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    nlp = pipeline("sentiment-analysis", 
                   model=finbert, 
                   tokenizer=tokenizer
                   )
    result = nlp(text)

    return result