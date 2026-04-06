"""NLP gRPC service implementation."""

import asyncio
from typing import Optional

import grpc
import spacy
import stanza
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from common.logger import logger
from generated.nlp import nlp_pb2, nlp_pb2_grpc
from settings import SETTINGS

from .src.constants import NUMERIC_ENTITY_TYPES
from .src.helpers import is_spacy_model
from .src import ExtractNumericEntitiesAlternative, ExtractNumericEntitiesAlternativeBatch


class NLPService(nlp_pb2_grpc.NLPServiceServicer):
    """gRPC service for NLP operations.

    Provides five RPCs: Encode, ExtractEntities, ExtractNumericEntities,
    Sentencise, and Health. Each can be independently enabled via SETTINGS.

    All ML inference is offloaded to a thread executor to avoid blocking
    the async event loop. Models are loaded lazily and can be preloaded
    concurrently at startup via preload_models().
    """

    def __init__(self):
        self._embedding_model: Optional[SentenceTransformer] = None
        self._ner_model = None
        self._ner_is_spacy: bool = True
        self._stanza_pipeline = None
        self._stanza_ner_pipeline = None
        self._dimension: Optional[int] = None

    def load_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {SETTINGS.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(SETTINGS.EMBEDDING_MODEL, device=SETTINGS.EMBEDDING_DEVICE)
            self._dimension = self._embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded (dimension: {self._dimension})")

    def load_ner_model(self):
        """Lazy load the NER model (spaCy or HuggingFace)."""
        if self._ner_model is None:
            model_name = SETTINGS.NER_MODEL
            logger.info(f"Loading NER model: {model_name}")
            if is_spacy_model(model_name):
                self._ner_is_spacy = True
                try:
                    self._ner_model = spacy.load(model_name)
                except OSError:
                    logger.info(f"Model not found, attempting to download: {model_name}")
                    import subprocess

                    subprocess.run(
                        [
                            "pip",
                            "install",
                            f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz",  # noqa: E501
                        ],
                        check=True,
                    )
                    self._ner_model = spacy.load(model_name)
            else:
                self._ner_is_spacy = False
                self._ner_model = hf_pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple",
                    device=torch.device(SETTINGS.EMBEDDING_DEVICE),
                )
            logger.info("NER model loaded")

    def load_stanza_pipeline(self):
        """Lazy load the Stanza pipeline for sentence tokenization."""
        if self._stanza_pipeline is None:
            logger.info(f"Loading Stanza pipeline: lang={SETTINGS.STANZA_LANG}")
            self._stanza_pipeline = stanza.Pipeline(SETTINGS.STANZA_LANG, processors="tokenize")
            logger.info("Stanza pipeline loaded")

    def load_stanza_ner_pipeline(self):
        """Lazy load the Stanza pipeline for NER."""
        if self._stanza_ner_pipeline is None:
            logger.info(f"Loading Stanza NER pipeline: lang={SETTINGS.STANZA_LANG}")
            self._stanza_ner_pipeline = stanza.Pipeline(SETTINGS.STANZA_LANG, processors="tokenize,ner")
            logger.info("Stanza NER pipeline loaded")

    def is_ready(self) -> bool:
        """Check if all enabled models are loaded (no inference, fast)."""
        return (
            (not SETTINGS.EMBEDDING_ENABLED or self._embedding_model is not None)
            and (not SETTINGS.NER_ENABLED or self._ner_model is not None)
            and (not SETTINGS.SENTENCISER_ENABLED or self._stanza_pipeline is not None)
            and (not SETTINGS.NUMERIC_NER_ENABLED or self._stanza_ner_pipeline is not None)
        )

    async def preload_models(self):
        """Preload all models concurrently (called at server startup)."""
        loop = asyncio.get_running_loop()
        tasks = []
        if SETTINGS.EMBEDDING_ENABLED:
            tasks.append(loop.run_in_executor(None, self.load_embedding_model))
        if SETTINGS.NER_ENABLED:
            tasks.append(loop.run_in_executor(None, self.load_ner_model))
        if SETTINGS.SENTENCISER_ENABLED:
            tasks.append(loop.run_in_executor(None, self.load_stanza_pipeline))
        if SETTINGS.NUMERIC_NER_ENABLED:
            tasks.append(loop.run_in_executor(None, self.load_stanza_ner_pipeline))
        if tasks:
            await asyncio.gather(*tasks)

    async def Encode(self, request: nlp_pb2.EmbedRequest, context: grpc.aio.ServicerContext) -> nlp_pb2.EmbedResponse:
        """Encode texts to embeddings."""
        if not SETTINGS.EMBEDDING_ENABLED:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Embedding service is disabled")

        if self._embedding_model is None:
            await context.abort(grpc.StatusCode.INTERNAL, "Embedding model not loaded")

        try:
            texts = list(request.texts)
            if not texts:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

            logger.debug(f"Encoding {len(texts)} texts")

            # Determine normalize setting (use request value or default)
            normalize = request.normalize if request.HasField("normalize") else SETTINGS.EMBEDDING_NORMALIZE

            # Encode texts

            def encode():
                return self._embedding_model.encode(
                    texts,
                    batch_size=SETTINGS.EMBEDDING_BATCH_SIZE,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )

            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, encode)

            # Convert to response format
            embedding_arrays = []
            for emb in embeddings:
                float_array = nlp_pb2.FloatArray(values=emb.tolist())
                embedding_arrays.append(float_array)

            logger.debug(f"Encoded {len(texts)} texts -> {len(embedding_arrays)} embeddings (dim={self._dimension})")

            return nlp_pb2.EmbedResponse(embeddings=embedding_arrays, dimension=self._dimension)

        except Exception as e:
            logger.error(f"Error in Encode: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def _extract_entities_spacy(self, texts: list[str]) -> list[nlp_pb2.TextEntities]:
        """Extract entities using a spaCy model."""
        all_results = []
        loop = asyncio.get_running_loop()

        def piper():
            return list(self._ner_model.pipe(texts, batch_size=SETTINGS.NER_BATCH_SIZE))

        docs = await loop.run_in_executor(None, piper)
        for doc in docs:
            entities = []
            for ent in doc.ents:
                entity = nlp_pb2.Entity(
                    text=ent.text, type=ent.label_, score=1.0, start=ent.start_char, end=ent.end_char
                )
                entities.append(entity)
            all_results.append(nlp_pb2.TextEntities(entities=entities))
        return all_results

    async def _extract_entities_hf(self, texts: list[str]) -> list[nlp_pb2.TextEntities]:
        """Extract entities using a HuggingFace model."""
        loop = asyncio.get_running_loop()
        batch_results = await loop.run_in_executor(
            None, lambda: self._ner_model(texts, batch_size=SETTINGS.NER_BATCH_SIZE)
        )

        all_results = []
        for ner_results in batch_results:
            entities = []
            for ent in ner_results:
                entity = nlp_pb2.Entity(
                    text=ent["word"],
                    type=ent["entity_group"],
                    score=float(ent["score"]),
                    start=ent["start"],
                    end=ent["end"],
                )
                entities.append(entity)
            all_results.append(nlp_pb2.TextEntities(entities=entities))
        return all_results

    async def ExtractEntities(
        self, request: nlp_pb2.NERRequest, context: grpc.aio.ServicerContext
    ) -> nlp_pb2.NERResponse:
        """Extract named entities from texts using spaCy or HuggingFace."""
        if not SETTINGS.NER_ENABLED:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "NER service is disabled")

        if self._ner_model is None:
            await context.abort(grpc.StatusCode.INTERNAL, "NER model not loaded")

        try:
            texts = list(request.texts)
            if not texts:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

            logger.debug(f"Extracting entities from {len(texts)} texts")

            all_results = []

            if self._ner_is_spacy:
                all_results = await self._extract_entities_spacy(texts)
            else:
                all_results = await self._extract_entities_hf(texts)

            logger.debug(f"Extracted entities from {len(texts)} texts")

            return nlp_pb2.NERResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in ExtractEntities: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def ExtractNumericEntities(
        self, request: nlp_pb2.NumericRequest, context: grpc.aio.ServicerContext
    ) -> nlp_pb2.NumericResponse:
        """Extract numeric entities from texts using Stanza NER.

        Extracts entities of types: CARDINAL, PERCENT, QUANTITY, MONEY, ORDINAL, DATE, TIME
        """
        if not SETTINGS.NUMERIC_NER_ENABLED:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Numeric NER service is disabled")

        if self._stanza_ner_pipeline is None:
            await context.abort(grpc.StatusCode.INTERNAL, "Stanza NER pipeline not loaded")

        try:
            texts = list(request.texts)
            if not texts:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

            # Separate non-empty texts for processing
            empty_result = nlp_pb2.TextNumericEntities(entities=[], tokens=[])
            non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]

            if not non_empty_indices:
                return nlp_pb2.NumericResponse(results=[empty_result] * len(texts))

            logger.debug(f"Extracting numeric entities from {len(texts)} texts")

            non_empty_texts = [texts[i] for i in non_empty_indices]

            loop = asyncio.get_running_loop()

            def piper():
                return ExtractNumericEntitiesAlternativeBatch(non_empty_texts)

            all_doc_entities = await loop.run_in_executor(None, piper)

            # Build results for non-empty docs
            processed = {}
            for idx, doc_entities in zip(non_empty_indices, all_doc_entities):
                entities = []
                tokens = []
                seen_tokens = set()

                for ent in doc_entities:
                    entity = nlp_pb2.NumericEntity(
                        text=ent["text"], type=ent["type"], start=ent["start"], end=ent["end"]
                    )
                    entities.append(entity)

                    normalized = ent["text"].lower().replace(' ', '')
                    if normalized not in seen_tokens:
                        seen_tokens.add(normalized)
                        tokens.append(ent["text"])

                processed[idx] = nlp_pb2.TextNumericEntities(entities=entities, tokens=tokens)

            # Merge: empty result for skipped indices, processed for the rest
            all_results = [processed.get(i, empty_result) for i in range(len(texts))]

            logger.debug(f"Extracted numeric entities from {len(texts)} texts")

            return nlp_pb2.NumericResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in ExtractNumericEntities: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Sentencise(
        self, request: nlp_pb2.SentenciseRequest, context: grpc.aio.ServicerContext
    ) -> nlp_pb2.SentenciseResponse:
        """Split texts into sentences using Stanza."""
        if not SETTINGS.SENTENCISER_ENABLED:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Sentenciser service is disabled")

        if self._stanza_pipeline is None:
            await context.abort(grpc.StatusCode.INTERNAL, "Stanza pipeline not loaded")

        try:
            texts = list(request.texts)
            if not texts:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

            logger.debug(f"Sentencising {len(texts)} texts")

            # Separate non-empty texts for processing
            empty_result = nlp_pb2.TextSentences(sentences=[])
            non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]

            if not non_empty_indices:
                return nlp_pb2.SentenciseResponse(results=[empty_result] * len(texts))

            non_empty_texts = [texts[i] for i in non_empty_indices]
            stanza_docs = [stanza.Document([], text=text) for text in non_empty_texts]

            loop = asyncio.get_running_loop()

            def piper():
                return self._stanza_pipeline(stanza_docs)

            stanza_docs = await loop.run_in_executor(None, piper)

            # Build results for non-empty docs
            processed = {}
            for idx, doc in zip(non_empty_indices, stanza_docs):
                sentences = [s.text for s in doc.sentences]
                processed[idx] = nlp_pb2.TextSentences(sentences=sentences)

            # Merge: empty result for skipped indices, processed for the rest
            all_results = [processed.get(i, empty_result) for i in range(len(texts))]

            logger.debug(f"Sentencised {len(texts)} texts")

            return nlp_pb2.SentenciseResponse(results=all_results)

        except Exception as e:
            logger.error(f"Error in Sentencise: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    def _check_embedding(self) -> bool:
        """Test embedding model with a dummy input."""
        try:
            self._embedding_model.encode(["health check"], show_progress_bar=False)
            return True
        except Exception:
            return False

    def _check_ner(self) -> bool:
        """Test NER model with a dummy input."""
        try:
            self._ner_model("health check")
            return True
        except Exception:
            return False

    def _check_stanza_pipeline(self) -> bool:
        """Test Stanza sentenciser with a dummy input."""
        try:
            self._stanza_pipeline("health check")
            return True
        except Exception:
            return False

    def _check_stanza_ner_pipeline(self) -> bool:
        """Test Stanza NER pipeline with a dummy input."""
        try:
            self._stanza_ner_pipeline("health check")
            return True
        except Exception:
            return False

    async def Health(self, request: nlp_pb2.HealthRequest, context: grpc.aio.ServicerContext) -> nlp_pb2.HealthResponse:
        """Deep health check — runs dummy inference on each enabled model concurrently.

        For lightweight load-balancer probes, use the standard grpc.health.v1
        service configured in main.py (checks model load status only).
        """
        loop = asyncio.get_running_loop()

        checks = {
            "embedding": (SETTINGS.EMBEDDING_ENABLED, self._embedding_model, self._check_embedding),
            "ner": (SETTINGS.NER_ENABLED, self._ner_model, self._check_ner),
            "sentenciser": (SETTINGS.SENTENCISER_ENABLED, self._stanza_pipeline, self._check_stanza_pipeline),
            "numeric_ner": (SETTINGS.NUMERIC_NER_ENABLED, self._stanza_ner_pipeline, self._check_stanza_ner_pipeline),
        }

        async def run_check(enabled, model, check_fn):
            if enabled and model is not None:
                return await loop.run_in_executor(None, check_fn)
            return False

        results = await asyncio.gather(
            *(run_check(enabled, model, check_fn) for enabled, model, check_fn in checks.values())
        )

        health_map = {}
        for key, ready in zip(checks, results):
            enabled = checks[key][0]
            health_map[key] = nlp_pb2.ModelHealth(enabled=enabled, ready=ready)

        healthy = all(h.ready for h in health_map.values() if h.enabled)

        return nlp_pb2.HealthResponse(
            healthy=healthy,
            embedding=health_map["embedding"],
            ner=health_map["ner"],
            sentenciser=health_map["sentenciser"],
            numeric_ner=health_map["numeric_ner"],
        )
