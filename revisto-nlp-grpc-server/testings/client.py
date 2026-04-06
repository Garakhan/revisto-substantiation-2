import csv
import json
import re
import time

import httpx

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
        {"text": e["text"], "label": e["label"]}
        for e in merged
        if _has_numeric_content(e["text"])
    ]


def _call_api(text):
    url = "http://localhost:8000/process/"
    payload = {"text": text, "labels": labels, "threshold": THRESHOLD}

    response = httpx.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}")


def split_sentences(text):
    """Split text on sentence-ending punctuation (.?!) while keeping the delimiter."""
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def process_text(text):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return _call_api(text)

    all_entities = []
    for sentence in sentences:
        offset = text.index(sentence)
        entities = _call_api(sentence)
        for e in entities:
            e["start"] += offset
            e["end"] += offset
        all_entities.extend(entities)
    return all_entities

if __name__ == "__main__":

    results = []

    output_path = "data/nexobrid_extractions.json"
    csv_path = "data/valid_claims_Nexobrid.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    

    for i, row in enumerate(rows):
        text = row["Claim text"].strip()
        print(f"[{i + 1}/{len(rows)}] Processing: {text[:80]}...")

        t0 = time.perf_counter()
        entities = process_text(text)
        t1 = time.perf_counter()
        entities = merge_adjacent_entities(text, entities)
        t2 = time.perf_counter()

        api_ms = round((t1 - t0) * 1000, 2)
        postprocess_ms = round((t2 - t1) * 1000, 2)
        total_ms = round((t2 - t0) * 1000, 2)

        results.append({
            "text": text,
            "entities": entities,
            "timing": {
                "api_ms": api_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
            }
        })

    # test_data_path = "data/test_data.json"
    # output_path = "data/test_data_results_gliner.json"
    # with open(test_data_path, "r", encoding="utf-8") as f:
    #     test_data_path = json.load(f)

    # for i, item in enumerate(test_data_path):
    #     text = item["text"].strip()
    #     print(f"[{i + 1}/{len(test_data_path)}] Processing: {text[:80]}...")
    #     entities = process_text(text)
    #     entities = merge_adjacent_entities(text, entities)
    #     results.append({"text": text, "entities": entities})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(results)} results to {output_path}")

    # text = "To report negative side effects contact Vericel Corporation at 888-454-BURN (888-454-2876) or FDA at 1-800-FDA-1088 (1-800-332-1088) or www.fda.gov/medwatch."
    # text = "Clinical studies of NEXOBRID did not include sufficient numbers of subjects 65 years of age and older to determine whether they respond differently from younger adult subjects."
    # text = "Limitations: Not studied in burns >30% TBSA, patients with full or partial thickness facial burns, perineal and/or genital burns, pregnant women or nursing mothers, poorly controlled diabetes, and patients with cardiopulmonary disease"
    # text = "NexoBrid is available as 5 g lyophilized powder (containing 4.85 g of anacaulase-bcdb) mixed in 50 g gel vehicle for treatment of up to 450 cm2 of burn area after mixing1"
    # text = "Dosage in adults: \n. Up to 15% BSA in one application\n. A second application may be applied 24 hours later - total treatment area must not exceed 20% BSA"
    # text = "Dosage in Pediatrics 6-17 Years of Age:\n. Up to 15% BSA in one application\n." 
    # text = "A second application is not recommended"
    # text = "175 patients* with DPT and/or FT thermal burns with BSA â‰¤30% were randomized in a 3:3:1 ratioâ€  to evaluate the safety and efficacy of NexoBrid in comparison with standard of care (SOC)â€¡ treatment and gel vehicle (placebo)"
    # text = "A ph value of < 5.5 is considered acidic, while a ph value of > 7.5 is considered alkaline."
    # text = "5mg/kg of a bmi is considered a low dose, while 20mg/kg is considered a high dose."
    # text = "Perhaps losing 5 mg/day of minerals could not be better."
    # text = "An integrated analysis of safety data from the pediatric study compared NexoBrid (n=69) to standard of care (SOC) (n=70). The SOC treatment included both surgical and non-surgical eschar removal methods."
    # text = "Statistical analysis established the noninferiority of NexoBrid compared with SOC when incorporating a 7-day advantage for the SOC group (p<0.01)"
    # text = "The mean percent of DPT wound area autografted was significantly lower with NexoBrid vs SOC (8.4 ± 21.3 with NexoBrid vs 21.5 ± 34.8 with SOC)"
    # text = "16.2% of patients developed a new high-grade cartilage defect between biopsy & implantation 0.6 cm² mean change in defect size between biopsy & implantation"
    # entities = process_text(text)
    # print(json.dumps({"text": text, "entities": entities}, indent=2, ensure_ascii=False))
