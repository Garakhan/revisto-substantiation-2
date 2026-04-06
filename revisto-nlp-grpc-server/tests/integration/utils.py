class BaseGRPCTestSuite:
    """Base test suite for NLP gRPC service tests."""

    stub_cls = None
    request_cls = None
    method_name = None
    expected_response_cls = None
