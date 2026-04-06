"""Integration tests for NLP gRPC service.

Strategy:
  - If a live gRPC server is running (localhost:50051), tests hit it directly.
  - Otherwise, an in-process server with mocked models is spun up as fallback.
"""

from collections import namedtuple
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from generated.nlp import nlp_pb2, nlp_pb2_grpc
from services.nlp_services import NLPService
from tests.conftest import _is_server_ready

# ---------------------------------------------------------------------------
# Mock model factories
# ---------------------------------------------------------------------------


def _make_mock_embedding_model(dimension=128):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimension

    def encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        return np.random.rand(len(texts), dimension).astype(np.float32)

    model.encode = MagicMock(side_effect=encode)
    return model


SpacyEnt = namedtuple("SpacyEnt", ["text", "label_", "start_char", "end_char"])


def _make_mock_spacy_ner():
    model = MagicMock()

    def pipe(texts, batch_size=16):
        docs = []
        for _text in texts:
            doc = MagicMock()
            doc.ents = [SpacyEnt(text="aspirin", label_="CHEMICAL", start_char=0, end_char=7)]
            docs.append(doc)
        return docs

    model.pipe = MagicMock(side_effect=pipe)
    model.return_value = MagicMock(ents=[SpacyEnt("aspirin", "CHEMICAL", 0, 7)])
    return model


StanzaEnt = namedtuple("StanzaEnt", ["text", "type", "start_char", "end_char"])
StanzaSentence = namedtuple("StanzaSentence", ["text"])


def _make_mock_stanza_pipeline():
    pipeline = MagicMock()

    def process(docs):
        if isinstance(docs, list):
            results = []
            for doc in docs:
                mock_doc = MagicMock()
                text = doc.text if hasattr(doc, "text") else str(doc)
                parts = text.split(". ")
                sents = []
                for i, part in enumerate(parts):
                    s = part if i == len(parts) - 1 else part + "."
                    sents.append(StanzaSentence(text=s))
                mock_doc.sentences = sents
                results.append(mock_doc)
            return results
        mock_doc = MagicMock()
        mock_doc.sentences = [StanzaSentence(text=str(docs))]
        return mock_doc

    pipeline.side_effect = process
    return pipeline


def _make_mock_stanza_ner_pipeline():
    pipeline = MagicMock()

    def process(docs):
        if isinstance(docs, list):
            results = []
            for _doc in docs:
                mock_doc = MagicMock()
                mock_doc.ents = [
                    StanzaEnt(text="45", type="CARDINAL", start_char=0, end_char=2),
                    StanzaEnt(text="10%", type="PERCENT", start_char=5, end_char=8),
                ]
                mock_doc.sentences = [StanzaSentence(text="dummy")]
                results.append(mock_doc)
            return results
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.sentences = [StanzaSentence(text=str(docs))]
        return mock_doc

    pipeline.side_effect = process
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures — live server or mock fallback
# ---------------------------------------------------------------------------


@pytest.fixture
async def nlp_service_and_stub(server_address):
    """
    Try to connect to a live server first.
    If unavailable, spin up an in-process server with mocked models.
    Yields (service_or_none, stub, is_live).
    """
    if await _is_server_ready("localhost", 50051):
        async with grpc.aio.insecure_channel(server_address) as channel:
            stub = nlp_pb2_grpc.NLPServiceStub(channel)
            yield None, stub, True
        return

    # Fallback: in-process server with mocks
    with (
        patch("services.nlp_services.SETTINGS.EMBEDDING_ENABLED", True),
        patch("services.nlp_services.SETTINGS.NER_ENABLED", True),
        patch("services.nlp_services.SETTINGS.SENTENCISER_ENABLED", True),
        patch("services.nlp_services.SETTINGS.NUMERIC_NER_ENABLED", True),
    ):
        service = NLPService()
        service._embedding_model = _make_mock_embedding_model(dimension=128)
        service._dimension = 128
        service._ner_model = _make_mock_spacy_ner()
        service._ner_is_spacy = True
        service._stanza_pipeline = _make_mock_stanza_pipeline()
        service._stanza_ner_pipeline = _make_mock_stanza_ner_pipeline()

        server = grpc.aio.server()
        nlp_pb2_grpc.add_NLPServiceServicer_to_server(service, server)
        port = server.add_insecure_port("[::]:0")
        await server.start()

        async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
            stub = nlp_pb2_grpc.NLPServiceStub(channel)
            yield service, stub, False

        await server.stop(0)


@pytest.fixture
def nlp_stub(nlp_service_and_stub):
    return nlp_service_and_stub[1]


@pytest.fixture
def nlp_service(nlp_service_and_stub):
    return nlp_service_and_stub[0]


@pytest.fixture
def is_live(nlp_service_and_stub):
    return nlp_service_and_stub[2]


# ===========================================================================
# Encode (Embedding) tests
# ===========================================================================


class TestEncode:
    async def test_encode_single_text(self, nlp_stub):
        resp = await nlp_stub.Encode(nlp_pb2.EmbedRequest(texts=["hello world"]))
        assert len(resp.embeddings) == 1
        assert resp.dimension > 0
        assert len(resp.embeddings[0].values) == resp.dimension

    async def test_encode_multiple_texts(self, nlp_stub):
        texts = ["first", "second", "third"]
        resp = await nlp_stub.Encode(nlp_pb2.EmbedRequest(texts=texts))
        assert len(resp.embeddings) == 3
        dim = resp.dimension
        for emb in resp.embeddings:
            assert len(emb.values) == dim

    async def test_encode_empty_texts_returns_error(self, nlp_stub):
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await nlp_stub.Encode(nlp_pb2.EmbedRequest(texts=[]))
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_encode_normalize_flag(self, nlp_stub, nlp_service, is_live):
        if is_live:
            pytest.skip("Cannot inspect model call args on live server")
        resp = await nlp_stub.Encode(nlp_pb2.EmbedRequest(texts=["test"], normalize=False))
        assert len(resp.embeddings) == 1
        call_kwargs = nlp_service._embedding_model.encode.call_args
        assert call_kwargs[1]["normalize_embeddings"] is False

    async def test_encode_disabled_returns_unimplemented(self):
        """When EMBEDDING_ENABLED=False, Encode should abort with UNIMPLEMENTED."""
        with patch("services.nlp_services.SETTINGS") as mock_settings:
            mock_settings.EMBEDDING_ENABLED = False

            service = NLPService()
            server = grpc.aio.server()
            nlp_pb2_grpc.add_NLPServiceServicer_to_server(service, server)
            port = server.add_insecure_port("[::]:0")
            await server.start()

            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = nlp_pb2_grpc.NLPServiceStub(channel)
                with pytest.raises(grpc.aio.AioRpcError) as exc:
                    await stub.Encode(nlp_pb2.EmbedRequest(texts=["test"]))
                assert exc.value.code() == grpc.StatusCode.UNIMPLEMENTED

            await server.stop(0)


# ===========================================================================
# ExtractEntities (NER) tests
# ===========================================================================


class TestExtractEntities:
    async def test_extract_entities_single_text(self, nlp_stub):
        resp = await nlp_stub.ExtractEntities(nlp_pb2.NERRequest(texts=["aspirin is a drug"]))
        assert len(resp.results) == 1
        assert len(resp.results[0].entities) > 0
        ent = resp.results[0].entities[0]
        assert ent.text
        assert ent.type

    async def test_extract_entities_multiple_texts(self, nlp_stub):
        texts = ["aspirin", "ibuprofen"]
        resp = await nlp_stub.ExtractEntities(nlp_pb2.NERRequest(texts=texts))
        assert len(resp.results) == 2

    async def test_extract_entities_empty_texts_returns_error(self, nlp_stub):
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await nlp_stub.ExtractEntities(nlp_pb2.NERRequest(texts=[]))
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_extract_entities_has_start_end(self, nlp_stub):
        resp = await nlp_stub.ExtractEntities(nlp_pb2.NERRequest(texts=["aspirin is a drug"]))
        ent = resp.results[0].entities[0]
        assert ent.start >= 0
        assert ent.end > ent.start


# ===========================================================================
# ExtractNumericEntities tests
# ===========================================================================


class TestExtractNumericEntities:
    async def test_numeric_entities_single_text(self, nlp_stub):
        resp = await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=["45 units at 10% discount"]))
        assert len(resp.results) == 1
        entities = resp.results[0].entities
        assert len(entities) >= 1
        types = {e.type for e in entities}
        # Should find at least one numeric type
        assert types & {"CARDINAL", "PERCENT", "QUANTITY", "MONEY", "ORDINAL", "DATE", "TIME"}

    async def test_numeric_entities_tokens_present(self, nlp_stub):
        resp = await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=["45 units at 10% discount"]))
        tokens = list(resp.results[0].tokens)
        assert len(tokens) >= 1

    async def test_numeric_entities_empty_text_skipped(self, nlp_stub):
        resp = await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=["", "  ", "45 items"]))
        assert len(resp.results) == 3
        assert len(resp.results[0].entities) == 0
        assert len(resp.results[1].entities) == 0
        assert len(resp.results[2].entities) >= 1

    async def test_numeric_entities_all_empty_texts(self, nlp_stub):
        resp = await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=["", "  "]))
        assert len(resp.results) == 2
        for r in resp.results:
            assert len(r.entities) == 0

    async def test_numeric_entities_empty_request_returns_error(self, nlp_stub):
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=[]))
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_numeric_entities_preserves_order(self, nlp_stub):
        resp = await nlp_stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=["", "45 items", "", "10 things"]))
        assert len(resp.results) == 4
        assert len(resp.results[0].entities) == 0
        assert len(resp.results[1].entities) >= 1
        assert len(resp.results[2].entities) == 0
        assert len(resp.results[3].entities) >= 1


# ===========================================================================
# Sentencise tests
# ===========================================================================


class TestSentencise:
    async def test_sentencise_single_text(self, nlp_stub):
        resp = await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=["Hello world. How are you?"]))
        assert len(resp.results) == 1
        sentences = list(resp.results[0].sentences)
        assert len(sentences) >= 2

    async def test_sentencise_multiple_texts(self, nlp_stub):
        resp = await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=["First. Second.", "Third."]))
        assert len(resp.results) == 2

    async def test_sentencise_empty_text_skipped(self, nlp_stub):
        resp = await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=["", "  ", "Hello. World."]))
        assert len(resp.results) == 3
        assert len(resp.results[0].sentences) == 0
        assert len(resp.results[1].sentences) == 0
        assert len(resp.results[2].sentences) >= 1

    async def test_sentencise_all_empty_texts(self, nlp_stub):
        resp = await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=["", "   "]))
        assert len(resp.results) == 2
        for r in resp.results:
            assert len(r.sentences) == 0

    async def test_sentencise_empty_request_returns_error(self, nlp_stub):
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=[]))
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_sentencise_preserves_order(self, nlp_stub):
        resp = await nlp_stub.Sentencise(nlp_pb2.SentenciseRequest(texts=["", "A sentence. Another one.", ""]))
        assert len(resp.results) == 3
        assert len(resp.results[0].sentences) == 0
        assert len(resp.results[1].sentences) >= 2
        assert len(resp.results[2].sentences) == 0


# ===========================================================================
# Health tests
# ===========================================================================


class TestHealth:
    async def test_health_response_structure(self, nlp_stub):
        resp = await nlp_stub.Health(nlp_pb2.HealthRequest())
        # Response should have all fields
        assert resp.HasField("embedding")
        assert resp.HasField("ner")
        assert resp.HasField("sentenciser")
        assert resp.HasField("numeric_ner")

    async def test_health_all_models_ready(self, nlp_stub):
        resp = await nlp_stub.Health(nlp_pb2.HealthRequest())
        assert resp.healthy is True
        assert resp.embedding.ready is True
        assert resp.ner.ready is True
        assert resp.sentenciser.ready is True
        assert resp.numeric_ner.ready is True

    async def test_health_no_models_loaded(self):
        """Service with no models loaded — enabled but not ready = unhealthy."""
        with (
            patch("services.nlp_services.SETTINGS.EMBEDDING_ENABLED", True),
            patch("services.nlp_services.SETTINGS.NER_ENABLED", True),
            patch("services.nlp_services.SETTINGS.SENTENCISER_ENABLED", True),
            patch("services.nlp_services.SETTINGS.NUMERIC_NER_ENABLED", True),
        ):
            service = NLPService()

            server = grpc.aio.server()
            nlp_pb2_grpc.add_NLPServiceServicer_to_server(service, server)
            port = server.add_insecure_port("[::]:0")
            await server.start()

            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = nlp_pb2_grpc.NLPServiceStub(channel)
                resp = await stub.Health(nlp_pb2.HealthRequest())
                assert resp.healthy is False
                assert resp.embedding.enabled is True
                assert resp.embedding.ready is False

            await server.stop(0)

    async def test_health_disabled_models_still_healthy(self):
        """If all models are disabled, healthy should be True (vacuous truth)."""
        with patch("services.nlp_services.SETTINGS") as mock_settings:
            mock_settings.EMBEDDING_ENABLED = False
            mock_settings.NER_ENABLED = False
            mock_settings.SENTENCISER_ENABLED = False
            mock_settings.NUMERIC_NER_ENABLED = False

            service = NLPService()

            server = grpc.aio.server()
            nlp_pb2_grpc.add_NLPServiceServicer_to_server(service, server)
            port = server.add_insecure_port("[::]:0")
            await server.start()

            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = nlp_pb2_grpc.NLPServiceStub(channel)
                resp = await stub.Health(nlp_pb2.HealthRequest())
                assert resp.healthy is True
                assert resp.embedding.enabled is False

            await server.stop(0)

    async def test_health_partial_failure(self, nlp_service, nlp_stub, is_live):
        """If one model's health check raises, that model is not ready."""
        if is_live:
            pytest.skip("Cannot inject model failures on live server")
        nlp_service._embedding_model.encode.side_effect = RuntimeError("broken")
        resp = await nlp_stub.Health(nlp_pb2.HealthRequest())
        assert resp.healthy is False
        assert resp.embedding.ready is False
        assert resp.ner.ready is True
        assert resp.sentenciser.ready is True
