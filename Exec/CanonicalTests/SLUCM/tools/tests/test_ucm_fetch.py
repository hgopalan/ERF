"""test_ucm_fetch.py — pytest tests for ucm_fetch module.

Tests failover and retry logic with mocked requests.
Runs in CI without any network access (all responses mocked).
"""

import pytest
from unittest.mock import patch, MagicMock
from ucm_fetch import fetch_with_failover, FetchAllFailed


def test_fetch_first_endpoint_succeeds():
    """First endpoint returns 200 → immediate success."""
    endpoints = ["https://api1.example.com", "https://api2.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        result = fetch_with_failover(endpoints, "TEST")
        assert result == b"test data"
        assert mock_get.call_count == 1


def test_fetch_first_fails_second_succeeds():
    """First endpoint returns 500, second returns 200 → success on second."""
    endpoints = ["https://api1.example.com", "https://api2.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        response_500 = MagicMock()
        response_500.status_code = 500
        response_500.reason = "Server Error"
        response_500.headers = {}
        
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"success"
        response_200.headers = {}
        
        mock_get.side_effect = [response_500, response_200]
        
        # Suppress time.sleep to speed up test
        with patch("ucm_fetch.time.sleep"):
            result = fetch_with_failover(endpoints, "TEST", max_retries=0)
        
        assert result == b"success"
        # First endpoint: 1 attempt → fail
        # Second endpoint: 1 attempt → succeed
        # Total: 2 calls
        assert mock_get.call_count == 2


def test_fetch_all_fail_raises_exception():
    """All endpoints return 500 → raises FetchAllFailed."""
    endpoints = ["https://api1.example.com", "https://api2.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason = "Server Error"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with patch("ucm_fetch.time.sleep"):
            with pytest.raises(FetchAllFailed):
                fetch_with_failover(endpoints, "TEST", max_retries=0)
        
        # Each endpoint tried once
        assert mock_get.call_count == 2


def test_http_404_terminal_no_failover():
    """HTTP 404 is terminal (client error) → fail immediately, don't try next."""
    endpoints = ["https://api1.example.com", "https://api2.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        response_404 = MagicMock()
        response_404.status_code = 404
        response_404.reason = "Not Found"
        response_404.headers = {}
        
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"success"
        response_200.headers = {}
        
        # First endpoint 404, should not try second
        mock_get.side_effect = [response_404, response_200]
        
        with pytest.raises(FetchAllFailed):
            fetch_with_failover(endpoints, "TEST", max_retries=0)
        
        # Only first endpoint called (404 is terminal)
        assert mock_get.call_count == 1


def test_http_429_retries():
    """HTTP 429 with Retry-After → wait and retry."""
    endpoints = ["https://api1.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.reason = "Too Many Requests"
        response_429.headers = {"Retry-After": "2"}
        
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"success"
        response_200.headers = {}
        
        # First call: 429, second call (after retry): 200
        mock_get.side_effect = [response_429, response_200]
        
        with patch("ucm_fetch.time.sleep") as mock_sleep:
            result = fetch_with_failover(endpoints, "TEST", max_retries=2)
        
        assert result == b"success"
        mock_sleep.assert_called_with(2)  # Called with Retry-After value


def test_retry_exhaustion():
    """Retries exhausted → move to next endpoint."""
    endpoints = ["https://api1.example.com", "https://api2.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        response_500 = MagicMock()
        response_500.status_code = 500
        response_500.reason = "Server Error"
        response_500.headers = {}
        
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"success"
        response_200.headers = {}
        
        # First endpoint: 500, 500, 500 (attempt 0, 1, 2 with max_retries=2)
        # Second endpoint: 200 (attempt 0)
        mock_get.side_effect = [response_500, response_500, response_500, response_200]
        
        with patch("ucm_fetch.time.sleep"):
            result = fetch_with_failover(endpoints, "TEST", max_retries=2)
        
        assert result == b"success"
        # First endpoint: 3 retries exhausted
        # Second endpoint: 1 success
        assert mock_get.call_count == 4


def test_format_kwargs():
    """URL .format() with format_kwargs."""
    endpoints = ["https://example.com/files/{country}.geojsonl.gz"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"data"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        result = fetch_with_failover(endpoints, "TEST", format_kwargs={"country": "US"})
        
        assert result == b"data"
        # Check that the formatted URL was used
        called_url = mock_get.call_args[0][0]
        assert "US" in called_url


def test_connection_error_retry():
    """ConnectionError → retry with backoff."""
    endpoints = ["https://api1.example.com"]
    
    with patch("ucm_fetch.requests.get") as mock_get:
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"success"
        response_200.headers = {}
        
        # First call: connection error, second call: success
        import requests
        mock_get.side_effect = [requests.ConnectionError("Connection refused"), response_200]
        
        with patch("ucm_fetch.time.sleep") as mock_sleep:
            result = fetch_with_failover(endpoints, "TEST", max_retries=1)
        
        assert result == b"success"
        mock_sleep.assert_called()  # Backoff was applied
