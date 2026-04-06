"""Named Entity Recognition module"""

from typing import List, Dict, Optional, Any

from ..config import NERConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class NERExtractor:
    """NER model wrapper for entity extraction using scispaCy"""

    def __init__(self, config: NERConfig):
        self.config = config
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load the spaCy NER pipeline"""
        if self._nlp is None:
            import spacy
            logger.info(f"Loading NER model: {self.config.model_name}")

            try:
                self._nlp = spacy.load(self.config.model_name)
            except OSError:
                # Model not installed, try to download it
                logger.info(f"Model not found, attempting to download: {self.config.model_name}")
                import subprocess
                subprocess.run([
                    "pip", "install",
                    f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{self.config.model_name}-0.5.4.tar.gz"
                ], check=True)
                self._nlp = spacy.load(self.config.model_name)

        return self._nlp

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        if not text or not text.strip():
            return []

        try:
            doc = self.nlp(text)

            # Clean and format results
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "score": 1.0,  # spaCy doesn't provide confidence scores
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            return entities

        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return []

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple texts"""
        results = []

        # Use spaCy's pipe for efficient batch processing
        for doc in self.nlp.pipe(texts, batch_size=self.config.batch_size):
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "score": 1.0,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            results.append(entities)

        return results

    def get_entity_types(self) -> List[str]:
        """Get available entity types from the model"""
        # BC5CDR entity types (diseases and chemicals)
        return ["DISEASE", "CHEMICAL"]


def extract_entities(
    text: str,
    ner_model: Optional[NERExtractor] = None,
    config: Optional[NERConfig] = None
) -> List[Dict[str, Any]]:
    """Convenience function to extract entities"""
    if ner_model is None:
        if config is None:
            config = NERConfig()
        ner_model = NERExtractor(config)
    
    return ner_model.extract(text)