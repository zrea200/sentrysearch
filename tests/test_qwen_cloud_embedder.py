"""Tests for DashScope qwen-cloud embedder helpers."""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.qwen_cloud_embedder import (
    DashScopeAPIError,
    QwenCloudEmbedder,
    _RateLimiter,
    _is_transient_transport_error,
    _retry,
    default_dashscope_embedding_model,
)


class TestDefaultDashscopeEmbeddingModel:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "my-custom-model")
        assert default_dashscope_embedding_model() == "my-custom-model"

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_EMBEDDING_MODEL", raising=False)
        assert default_dashscope_embedding_model() == "qwen3-vl-embedding"


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = _RateLimiter(max_per_minute=5)
        for _ in range(5):
            limiter.wait()

    @patch("sentrysearch.qwen_cloud_embedder.time.sleep")
    @patch("sentrysearch.qwen_cloud_embedder.time.monotonic")
    def test_blocks_at_limit(self, mock_mono, mock_sleep):
        limiter = _RateLimiter(max_per_minute=2)
        mock_mono.side_effect = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 62.0, 62.0, 63.0, 63.0]
        limiter.wait()
        limiter.wait()
        limiter.wait()
        mock_sleep.assert_called_once()


class TestIsTransientTransportError:
    def test_timeout_error(self):
        assert _is_transient_transport_error(TimeoutError())

    def test_connection_reset(self):
        assert _is_transient_transport_error(ConnectionResetError())

    def test_value_error_not_transient(self):
        assert not _is_transient_transport_error(ValueError("bad input"))


class TestRetry:
    def test_retries_dashscope_rate_limit(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise DashScopeAPIError("HTTP 429 rate limit")
            return "ok"

        with patch("sentrysearch.qwen_cloud_embedder.time.sleep"):
            assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        assert calls["n"] == 2

    def test_retries_connection_error(self):
        try:
            import requests
        except ImportError:
            pytest.skip("requests not installed")
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise requests.exceptions.ConnectionError("reset by peer")
            return "ok"

        with patch("sentrysearch.qwen_cloud_embedder.time.sleep"):
            assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"

    def test_non_retryable_api_error_raises(self):
        def fn():
            raise DashScopeAPIError("invalid model xyz")

        with pytest.raises(DashScopeAPIError, match="invalid model"):
            _retry(fn, max_retries=2, initial_delay=0.01)


def _fake_resp(status_code, output=None, code="", message=""):
    r = MagicMock()
    r.status_code = status_code

    def getter(key, default=None):
        if key == "output":
            return output
        if key == "code":
            return code
        if key == "message":
            return message
        return default

    r.get.side_effect = getter
    return r


class TestParseEmbeddingResponse:
    def test_ok(self):
        out = {"embeddings": [{"embedding": [0.5, -0.5]}]}
        vec = QwenCloudEmbedder._parse_embedding_response(
            _fake_resp(HTTPStatus.OK, output=out),
        )
        assert vec == [0.5, -0.5]

    def test_http_error(self):
        with pytest.raises(DashScopeAPIError, match="DashScope API error"):
            QwenCloudEmbedder._parse_embedding_response(
                _fake_resp(HTTPStatus.BAD_REQUEST, code="Invalid", message="nope"),
            )

    def test_empty_embeddings(self):
        with pytest.raises(DashScopeAPIError, match="no embeddings"):
            QwenCloudEmbedder._parse_embedding_response(
                _fake_resp(HTTPStatus.OK, output={"embeddings": []}),
            )

    def test_missing_vector(self):
        with pytest.raises(DashScopeAPIError, match="missing embedding"):
            QwenCloudEmbedder._parse_embedding_response(
                _fake_resp(HTTPStatus.OK, output={"embeddings": [{}]}),
            )
