"""ucm_fetch.py — Hybrid resilience fetcher with endpoint failover and retry logic.

Phase 2.9: Implements robust fetching across all data sources with intelligent
failover handling per HTTP status code.

Retry policy:
  - HTTP 5xx / timeout / connection error: retry with exponential backoff
  - HTTP 429 (Too Many Requests): honor Retry-After header
  - HTTP 4xx (except 429): fail immediately (client error, no retry/failover)
  - HTTP 200: return content

Structured logging:
  [ucm_fetch] <source> Trying <endpoint> … [status] [duration]
"""

import requests
import time
from typing import List, Optional, Dict


class FetchAllFailed(RuntimeError):
    """Raised when all endpoints fail to fetch data."""
    pass


def fetch_with_failover(
    endpoints: List[str],
    source: str,
    timeout: int = 60,
    max_retries: int = 2,
    format_kwargs: Optional[Dict[str, str]] = None,
) -> bytes:
    """Try each endpoint in order with retry logic per status code.
    
    Args:
        endpoints: List of URLs to try (in order of preference)
        source: Short label for logs (e.g., "OSM", "MICROSOFT_ML")
        timeout: Timeout per request in seconds (default 60)
        max_retries: Max retries per endpoint (default 2)
        format_kwargs: Dict of kwargs for URL .format() (e.g., {"country": "US"})
    
    Returns:
        bytes: Content on success
        
    Raises:
        FetchAllFailed: If all endpoints exhausted
    """
    if format_kwargs is None:
        format_kwargs = {}
    
    formatted_endpoints = []
    for ep in endpoints:
        try:
            formatted_endpoints.append(ep.format(**format_kwargs))
        except KeyError as e:
            # Missing format arg; skip this endpoint
            print(f"[ucm_fetch] {source} Skipping {ep}: missing format arg {e}")
            continue
    
    if not formatted_endpoints:
        raise FetchAllFailed(f"{source}: No valid endpoints after formatting")
    
    last_error = None
    terminal_error = False  # Track if we hit a terminal error (4xx non-429)
    
    for ep_idx, url in enumerate(formatted_endpoints):
        if terminal_error:
            break  # Don't try remaining endpoints if we hit 4xx
        
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                print(f"[ucm_fetch] {source} Trying {url.split('/')[-1] if '/' in url else url} … ", end="", flush=True)
                
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                elapsed = time.time() - t0
                
                # HTTP 4xx (except 429): terminal error, don't retry or try next
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    print(f"HTTP {response.status_code} (terminal client error)")
                    last_error = f"HTTP {response.status_code}: {response.reason}"
                    terminal_error = True
                    break  # Break out of retry loop to skip remaining endpoints
                
                # HTTP 429: honor Retry-After if present
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after else 60
                    print(f"HTTP 429 (rate limited, wait {wait_time}s)")
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        continue  # Retry this endpoint
                    else:
                        last_error = f"HTTP 429: rate limited (exhausted retries)"
                        break  # Move to next endpoint
                
                # HTTP 200+: success
                if response.status_code == 200:
                    size_mb = len(response.content) / (1024 * 1024)
                    print(f"OK ({size_mb:.1f} MB, {elapsed:.1f}s)")
                    return response.content
                
                # HTTP 5xx / 3xx (not terminal): retry
                if response.status_code >= 500 or response.status_code >= 300:
                    print(f"HTTP {response.status_code} ({elapsed:.1f}s)")
                    if attempt < max_retries:
                        backoff = 30 * (attempt + 1)  # 30s, 60s, ...
                        print(f"[ucm_fetch] {source} Retrying in {backoff}s …")
                        time.sleep(backoff)
                        continue
                    else:
                        last_error = f"HTTP {response.status_code}: {response.reason} (exhausted retries)"
                        break  # Move to next endpoint
                
                # Other status: treat as error
                print(f"HTTP {response.status_code}")
                last_error = f"HTTP {response.status_code}: {response.reason}"
                break
                
            except (requests.Timeout, requests.ConnectionError) as e:
                elapsed = time.time() - t0
                err_msg = "timeout" if isinstance(e, requests.Timeout) else "connection error"
                print(f"{err_msg} ({elapsed:.1f}s)")
                if attempt < max_retries:
                    backoff = 30 * (attempt + 1)
                    print(f"[ucm_fetch] {source} Retrying in {backoff}s …")
                    time.sleep(backoff)
                    continue
                else:
                    last_error = f"{err_msg} (exhausted retries)"
                    break  # Move to next endpoint
            
            except Exception as e:
                elapsed = time.time() - t0
                print(f"error: {e} ({elapsed:.1f}s)")
                last_error = f"Exception: {str(e)}"
                break  # Move to next endpoint
    
    # All endpoints failed
    raise FetchAllFailed(f"{source}: All {len(formatted_endpoints)} endpoint(s) failed. Last error: {last_error}")
