def is_spacy_model(model_name: str) -> bool:
    """
    Return True if model_name is a spaCy model, False if HuggingFace.
    It naively supposes that if the model name contains a slash, it's a HuggingFace model, otherwise it's a spaCy model.
    """
    return "/" not in model_name
