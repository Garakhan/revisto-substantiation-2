"""NLP gRPC service implementation."""

from typing import List, Optional

import grpc
import numpy as np
import spacy
import stanza
import torch
from sentence_transformers import SentenceTransformer

from common.logger import logger
from generated.nlp import nlp_pb2, nlp_pb2_grpc
from settings import SETTINGS


class NLPService(nlp_pb2_grpc.NLPServiceServicer):
    """gRPC service for NLP operations (embedding, NER, and sentencisation)."""

    # Stanza entity types that represent numeric values
    NUMERIC_ENTITY_TYPES = {"CARDINAL", "PERCENT", "QUANTITY", "MONEY", "ORDINAL", "DATE", "TIME"}

    def __init__(self):
        self._embedding_model: Optional[SentenceTransformer] = None
        self._ner_model = None
        self._stanza_pipeline = None
        self._stanza_ner_pipeline = None
        self._models_loaded = False

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {SETTINGS.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(
                SETTINGS.EMBEDDING_MODEL,
                device=SETTINGS.EMBEDDING_DEVICE
            )
            logger.info(f"Embedding model loaded (dimension: {self._embedding_model.get_sentence_embedding_dimension()})")
        return self._embedding_model

    @property
    def ner_model(self):
        """Lazy load the scispaCy NER model."""
        if self._ner_model is None:
            logger.info(f"Loading NER model: {SETTINGS.NER_MODEL}")
            try:
                self._ner_model = spacy.load(SETTINGS.NER_MODEL)
            except OSError:
                logger.error(f"NER model {SETTINGS.NER_MODEL} not found. Please install it.")
                raise
            logger.info("NER model loaded")
        return self._ner_model

    @property
    def stanza_pipeline(self):
        """Lazy load the Stanza pipeline for sentence tokenization."""
        if self._stanza_pipeline is None:
            logger.info(f"Loading Stanza pipeline: lang={SETTINGS.STANZA_LANG}")
            self._stanza_pipeline = stanza.Pipeline(
                SETTINGS.STANZA_LANG,
                processors="tokenize"
            )
            logger.info("Stanza pipeline loaded")
        return self._stanza_pipeline

    @property
    def stanza_ner_pipeline(self):
        """Lazy load the Stanza pipeline for NER (numeric entity extraction)."""
        if self._stanza_ner_pipeline is None:
            logger.info(f"Loading Stanza NER pipeline: lang={SETTINGS.STANZA_LANG}")
            self._stanza_ner_pipeline = stanza.Pipeline(
                SETTINGS.STANZA_LANG,
                processors="tokenize,ner"
            )
            logger.info("Stanza NER pipeline loaded")
        return self._stanza_ner_pipeline

    def preload_models(self):
        """Preload all models (called at server startup if configured)."""
        _ = self.embedding_model
        if SETTINGS.NER_ENABLED:
            _ = self.ner_model
        if SETTINGS.SENTENCISER_ENABLED:
            _ = self.stanza_pipeline
        self._models_loaded = True

    async def Encode(
        self,
        request: nlp_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext
    ) -> nlp_pb2.EmbedResponse:
        """Encode texts to embeddings."""
        try:
            texts = list(request.texts)
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No texts provided")
                return nlp_pb2.EmbedResponse()

            logger.debug(f"Encoding {len(texts)} texts")

            # Determine normalize setting (use request value or default)
            normalize = request.normalize if request.HasField("normalize") else SETTINGS.EMBEDDING_NORMALIZE

            # Encode texts
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=SETTINGS.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            # Convert to response format
            embedding_arrays = []
            for emb in embeddings:
                float_array = nlp_pb2.FloatArray(values=emb.tolist())
                embedding_arrays.append(float_array)

            dimension = self.embedding_model.get_sentence_embedding_dimension()

            logger.debug(f"Encoded {len(texts)} texts -> {len(embedding_arrays)} embeddings (dim={dimension})")

            return nlp_pb2.EmbedResponse(
                embeddings=embedding_arrays,
                dimension=dimension
            )

        except Exception as e:
            logger.error(f"Error in Encode: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nlp_pb2.EmbedResponse()

    async def ExtractEntities(
        self,
        request: nlp_pb2.NERRequest,
        context: grpc.aio.ServicerContext
    ) -> nlp_pb2.NERResponse:
        """Extract named entities from texts using scispaCy."""
        try:
            texts = list(request.texts)
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No texts provided")
                return nlp_pb2.NERResponse()

            if not SETTINGS.NER_ENABLED:
                # Return empty results if NER is disabled
                return nlp_pb2.NERResponse(
                    results=[nlp_pb2.TextEntities(entities=[]) for _ in texts]
                )

            logger.debug(f"Extracting entities from {len(texts)} texts")

            all_results = []

            # Use spaCy's pipe for efficient batch processing
            for doc in self.ner_model.pipe(texts, batch_size=SETTINGS.NER_BATCH_SIZE):
                entities = []
                for ent in doc.ents:
                    entity = nlp_pb2.Entity(
                        text=ent.text,
                        type=ent.label_,
                        score=1.0,  # spaCy doesn't provide confidence scores
                        start=ent.start_char,
                        end=ent.end_char
                    )
                    entities.append(entity)
                all_results.append(nlp_pb2.TextEntities(entities=entities))

            logger.debug(f"Extracted entities from {len(texts)} texts")

            return nlp_pb2.NERResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in ExtractEntities: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nlp_pb2.NERResponse()

    async def ExtractNumericEntities(
        self,
        request: nlp_pb2.NumericRequest,
        context: grpc.aio.ServicerContext
    ) -> nlp_pb2.NumericResponse:
        """Extract numeric entities from texts using Stanza NER.

        Extracts entities of types: CARDINAL, PERCENT, QUANTITY, MONEY, ORDINAL, DATE, TIME
        """
        try:
            texts = list(request.texts)
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No texts provided")
                return nlp_pb2.NumericResponse()

            logger.debug(f"Extracting numeric entities from {len(texts)} texts")

            all_results = []

            for text in texts:
                if not text or not text.strip():
                    all_results.append(nlp_pb2.TextNumericEntities(entities=[], tokens=[]))
                    continue

                # Process with Stanza NER
                doc = self.stanza_ner_pipeline(text)

                entities = []
                tokens = []
                seen_tokens = set()

                for ent in doc.ents:
                    # Filter for numeric entity types only
                    if ent.type in self.NUMERIC_ENTITY_TYPES:
                        entity = nlp_pb2.NumericEntity(
                            text=ent.text,
                            type=ent.type,
                            start=ent.start_char,
                            end=ent.end_char
                        )
                        entities.append(entity)

                        # Add to tokens list (deduplicated, normalized)
                        normalized = ent.text.lower().replace(' ', '')
                        if normalized not in seen_tokens:
                            seen_tokens.add(normalized)
                            tokens.append(ent.text)

                all_results.append(nlp_pb2.TextNumericEntities(
                    entities=entities,
                    tokens=tokens
                ))

            logger.debug(f"Extracted numeric entities from {len(texts)} texts")

            return nlp_pb2.NumericResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in ExtractNumericEntities: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nlp_pb2.NumericResponse()

    async def Sentencise(
        self,
        request: nlp_pb2.SentenciseRequest,
        context: grpc.aio.ServicerContext
    ) -> nlp_pb2.SentenciseResponse:
        """Split texts into sentences using Stanza."""
        try:
            texts = list(request.texts)
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No texts provided")
                return nlp_pb2.SentenciseResponse()

            if not SETTINGS.SENTENCISER_ENABLED:
                # Return original texts as single sentences if disabled
                return nlp_pb2.SentenciseResponse(
                    results=[nlp_pb2.TextSentences(sentences=[t]) for t in texts]
                )

            logger.debug(f"Sentencising {len(texts)} texts")

            all_results = []

            for text in texts:
                if not text or not text.strip():
                    all_results.append(nlp_pb2.TextSentences(sentences=[]))
                    continue

                # Process with Stanza
                doc = self.stanza_pipeline(text)
                sentences = [s.text for s in doc.sentences]
                all_results.append(nlp_pb2.TextSentences(sentences=sentences))

            logger.debug(f"Sentencised {len(texts)} texts")

            return nlp_pb2.SentenciseResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in Sentencise: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nlp_pb2.SentenciseResponse()

    async def Health(
        self,
        request: nlp_pb2.HealthRequest,
        context: grpc.aio.ServicerContext
    ) -> nlp_pb2.HealthResponse:
        """Health check endpoint."""
        return nlp_pb2.HealthResponse(
            healthy=True,
            embedding_model=SETTINGS.EMBEDDING_MODEL,
            ner_model=SETTINGS.NER_MODEL,
            stanza_lang=SETTINGS.STANZA_LANG,
            models_loaded=self._models_loaded
        )
