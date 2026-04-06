from gliner import GLiNER
import re

model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")
# model = GLiNER.from_pretrained("Ihor/gliner-large-v2.5-biomed")
# model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-large-1stg-v1.0")


labels = [
    "Number", "Measurement", "Date", "Ratio", "Threshold", "Code", "Time", "Duration", "Age"
]
THRESHOLD = 0.5

WORD_NUMBERS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "half", "quarter", "double", "triple",
    "once", "twice",
}


def _has_numeric_content(entity_text):
    """Return True if the entity contains a digit or a number word."""
    if re.search(r'\d', entity_text):
        return True
    words = set(entity_text.lower().split())
    return bool(words & WORD_NUMBERS)


def merge_adjacent_entities(text, entities):
    """Merge consecutive entities with the same label that are separated
    only by non-alphanumeric characters (spaces, =, <, >, (, ), etc.)."""
    if not entities:
        return entities

    # Sort by position
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    merged = [dict(sorted_entities[0])]
    for current in sorted_entities[1:]:
        prev = merged[-1]
        gap = text[prev["end"]:current["start"]]

        if prev["label"] == current["label"] and not any(c.isalnum() for c in gap):
            prev["text"] = text[prev["start"]:current["end"]]
            prev["end"] = current["end"]
        else:
            merged.append(dict(current))

    return [
        {"text": e["text"], "label": e["label"], "start": e["start"], "end": e["end"]}
        for e in merged
        if _has_numeric_content(e["text"])
    ]


def split_sentences(text):
    """Split text on sentence-ending punctuation (.?!) while keeping the delimiter."""
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def process_text(text, labels=labels, threshold=THRESHOLD):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return model.predict_entities(text, labels, threshold=threshold)

    all_entities = []
    for sentence in sentences:
        offset = text.index(sentence)
        entities = model.predict_entities(sentence, labels, threshold=threshold)
        for e in entities:
            e["start"] += offset
            e["end"] += offset
        all_entities.extend(entities)
    return all_entities


def process(text: str, labels: list = labels, threshold: float = THRESHOLD):
    entities = process_text(text, labels, threshold)
    entities = merge_adjacent_entities(text, entities)
    return [{"text": entity["text"], "type": entity["label"], "start": entity["start"], "end": entity["end"]} for entity in entities]


def process_batch(texts: list, labels: list = labels, threshold: float = THRESHOLD):
    """Process multiple texts in batch using GLiNER's batch_predict_entities."""
    if not texts:
        return []

    # Collect all sentences with their source text index and offset
    all_sentences = []
    sentence_map = []  # (text_idx, offset)
    for text_idx, text in enumerate(texts):
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            all_sentences.append(text)
            sentence_map.append((text_idx, 0))
        else:
            for sentence in sentences:
                offset = text.index(sentence)
                all_sentences.append(sentence)
                sentence_map.append((text_idx, offset))

    # Batch predict all sentences at once
    all_predictions = model.batch_predict_entities(
        all_sentences, labels, threshold=threshold
    )

    # Group predictions back by source text
    text_entities = [[] for _ in texts]
    for (text_idx, offset), entities in zip(sentence_map, all_predictions):
        for e in entities:
            e["start"] += offset
            e["end"] += offset
        text_entities[text_idx].extend(entities)

    # Merge and filter per text
    results = []
    for text_idx, text in enumerate(texts):
        merged = merge_adjacent_entities(text, text_entities[text_idx])
        results.append([
            {"text": e["text"], "type": e["label"], "start": e["start"], "end": e["end"]}
            for e in merged
        ])
    return results
