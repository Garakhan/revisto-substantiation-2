# gRPC client for NLP service
from .client import NLPGrpcClient, GrpcEmbeddingModel, GrpcNERExtractor, GrpcSentenciser, GrpcNumericExtractor

__all__ = ["NLPGrpcClient", "GrpcEmbeddingModel", "GrpcNERExtractor", "GrpcSentenciser", "GrpcNumericExtractor"]
