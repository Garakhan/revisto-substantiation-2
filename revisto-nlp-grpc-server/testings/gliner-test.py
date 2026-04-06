from gliner import GLiNER
import fastapi
import time

app = fastapi.FastAPI()

model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")
# model = GLiNER.from_pretrained("Ihor/gliner-large-v2.5-biomed")
# model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-large-1stg-v1.0")

text = """
The patient, a 45-year-old male, was diagnosed with type 2 diabetes mellitus and hypertension.
He was prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily. 
A recent lab test showed elevated HbA1c levels at 8.2%.
"""
text = """
NexoBrid is available as 5 g lyophilized powder (containing 4.85 g of anacaulase-bcdb) mixed in 50 g gel vehicle for treatment of up to 450 cm2 of burn area after mixing1
"""

labels = ["Disease", "Drug", "Drug dosage", "Drug frequency", "Lab test",
          "Lab test value", "Demographic information", "Drug form", "Study number",
          "Study group", "Study outcome", "Other"]

@app.post("/process/")
def process(text: str = fastapi.Body(embed=True), labels: list = fastapi.Body(embed=True), threshold: float = fastapi.Body(embed=True, default=0.5)):
    s = time.time()
    entities = model.predict_entities(text, labels, threshold=threshold)
    print(f"Processing time: {time.time() - s} seconds")
    return [{"text": entity["text"], "label": entity["label"], "start": entity["start"], "end": entity["end"]} for entity in entities]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gliner-test:app", host="0.0.0.0", port=8000, reload=True)
