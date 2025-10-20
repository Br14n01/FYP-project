from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Sample text for entity prediction
text = """
Microsoft Corporation (NASDAQ:MSFT) is one of the AI Stocks Analysts Are Watching Closely. 
On October 16, Morgan Stanley reiterated the stock as “Overweight” and said that Microsoft is a “core holding.
"""
def entity_extraction(text):

    # Labels for entity prediction
    # Most GLiNER models should work best when entity types are in lower case or title case
    labels = ["Stock", "Corporation"]

    # Perform entity prediction
    entities = model.predict_entities(text, labels, threshold=0.5)
    result = {}
    # Display predicted entities and their labels
    for entity in entities:
        result[entity["label"]] = entity["text"]

    return result
