import asyncio
import grpc
from generated.nlp import nlp_pb2, nlp_pb2_grpc

SERVER = "nlp-grpc.dev.internal.cloud-revisto.com:50051"
CA_CERT = "/home/garakhan/root-ca.crt"

TEXTS = [
    "Barack Obama was the 45th president of the United States. He was a guy born in Hawaii.",
    "Aspirin 500mg tablets cost $12.99 for a pack of 30. Buy it or not to buy it, that is the question.",
]


async def main():
    with open(CA_CERT, "rb") as f:
        ca_cert = f.read()

    credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)

    async with grpc.aio.secure_channel(SERVER, credentials) as channel:
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        # 1. Health check
        print("=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)
        health = await stub.Health(nlp_pb2.HealthRequest())
        print(f"Healthy: {health.healthy}")
        print(f"  Embedding:   enabled={health.embedding.enabled}, ready={health.embedding.ready}")
        print(f"  NER:         enabled={health.ner.enabled}, ready={health.ner.ready}")
        print(f"  Sentenciser: enabled={health.sentenciser.enabled}, ready={health.sentenciser.ready}")
        print(f"  Numeric NER: enabled={health.numeric_ner.enabled}, ready={health.numeric_ner.ready}")

        # 2. Encode
        print(f"\n{'=' * 60}")
        print("ENCODE")
        print("=" * 60)
        try:
            resp = await stub.Encode(nlp_pb2.EmbedRequest(texts=TEXTS))
            print(f"Dimension: {resp.dimension}")
            for i, emb in enumerate(resp.embeddings):
                print(f"  Text {i}: [{emb.values[0]:.4f}, {emb.values[1]:.4f}, ...] (len={len(emb.values)})")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 3. Extract Entities (NER)
        print(f"\n{'=' * 60}")
        print("EXTRACT ENTITIES")
        print("=" * 60)
        try:
            resp = await stub.ExtractEntities(nlp_pb2.NERRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}: {TEXTS[i][:50]}...")
                for ent in result.entities:
                    print(f"    - {ent.text!r} [{ent.type}] score={ent.score:.2f} ({ent.start}:{ent.end})")
                if not result.entities:
                    print("    (no entities)")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 4. Extract Numeric Entities
        print(f"\n{'=' * 60}")
        print("EXTRACT NUMERIC ENTITIES")
        print("=" * 60)
        try:
            resp = await stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}: {TEXTS[i][:50]}...")
                for ent in result.entities:
                    print(f"    - {ent.text!r} [{ent.type}] ({ent.start}:{ent.end})")
                print(f"    Tokens: {list(result.tokens)}")
                if not result.entities:
                    print("    (no numeric entities)")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 5. Sentencise
        print(f"\n{'=' * 60}")
        print("SENTENCISE")
        print("=" * 60)
        try:
            resp = await stub.Sentencise(nlp_pb2.SentenciseRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}:")
                for j, sent in enumerate(result.sentences):
                    print(f"    Sentence {j}: {sent!r}")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")


if __name__ == "__main__":
    asyncio.run(main())
