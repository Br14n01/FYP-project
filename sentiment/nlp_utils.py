from gliner import GLiNER
from sentence_transformers import SentenceTransformer

def entity_extraction(text):

    # Initialize GLiNER with the base model
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
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


def text_correlation(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)