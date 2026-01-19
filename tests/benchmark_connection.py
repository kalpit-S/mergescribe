"""
Benchmark: Fresh connections vs persistent session vs HTTP/2
Also compares OpenRouter vs direct Gemini API.

Tests the actual latency difference for API calls.
"""

import time
import statistics
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# Try to import httpx for HTTP/2 testing
HAS_HTTPX = False
try:
    import httpx
    # Test if HTTP/2 is actually available
    try:
        _test = httpx.Client(http2=True)
        _test.close()
        HAS_HTTPX = True
    except ImportError:
        print("httpx installed but HTTP/2 not available")
        print("Install with: pip install httpx[http2]")
except ImportError:
    print("httpx not installed - skipping HTTP/2 tests")
    print("Install with: pip install httpx[http2]")


def load_api_keys():
    """Load API keys from environment and .env files."""
    keys = {"OPENROUTER_API_KEY": None, "GEMINI_API_KEY": None}

    for env_path in [Path(".env"), Path.home() / ".mergescribe" / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key in keys:
                            keys[key] = value

    # Environment overrides
    for key in keys:
        env_val = os.getenv(key)
        if env_val:
            keys[key] = env_val

    return keys


API_KEYS = load_api_keys()

# OpenRouter config
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {API_KEYS['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json",
}
OPENROUTER_PAYLOAD = {
    "model": "google/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Say 'hi'"}],
    "max_tokens": 5,
    "temperature": 0,
    "reasoning": {"max_tokens": 0},  # Disable reasoning/thinking
}

# Direct Gemini config
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_PAYLOAD = {
    "contents": [{"parts": [{"text": "Say 'hi'"}]}],
    "generationConfig": {
        "maxOutputTokens": 5,
        "temperature": 0,
        "thinkingConfig": {"thinkingBudget": 0},  # Disable thinking
    },
}

NUM_REQUESTS = 5
WARMUP_REQUESTS = 1


def benchmark_openrouter_session():
    """OpenRouter with persistent session."""
    if not API_KEYS["OPENROUTER_API_KEY"]:
        print("  Skipped - no OPENROUTER_API_KEY")
        return []

    times = []
    session = requests.Session()

    # Warmup
    for _ in range(WARMUP_REQUESTS):
        session.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        response = session.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            times.append(elapsed)
            print(f"  OpenRouter #{i+1}: {elapsed:.1f}ms")
        else:
            print(f"  OpenRouter #{i+1}: ERROR {response.status_code} - {response.text[:100]}")

    session.close()
    return times


def benchmark_gemini_direct():
    """Direct Gemini API with persistent session."""
    if not API_KEYS["GEMINI_API_KEY"]:
        print("  Skipped - no GEMINI_API_KEY")
        return []

    times = []
    session = requests.Session()
    url = f"{GEMINI_URL}?key={API_KEYS['GEMINI_API_KEY']}"

    # Warmup
    for _ in range(WARMUP_REQUESTS):
        session.post(url, json=GEMINI_PAYLOAD, timeout=30)

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        response = session.post(url, json=GEMINI_PAYLOAD, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            times.append(elapsed)
            print(f"  Gemini Direct #{i+1}: {elapsed:.1f}ms")
        else:
            print(f"  Gemini Direct #{i+1}: ERROR {response.status_code} - {response.text[:100]}")

    session.close()
    return times


def benchmark_fresh_connections():
    """Each request creates a new TCP/TLS connection (OpenRouter)."""
    if not API_KEYS["OPENROUTER_API_KEY"]:
        print("  Skipped - no OPENROUTER_API_KEY")
        return []

    times = []

    # Warmup
    for _ in range(WARMUP_REQUESTS):
        requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            times.append(elapsed)
            print(f"  Fresh #{i+1}: {elapsed:.1f}ms")
        else:
            print(f"  Fresh #{i+1}: ERROR {response.status_code}")

    return times


def benchmark_session():
    """Reuse connection via requests.Session (OpenRouter)."""
    return benchmark_openrouter_session()


def benchmark_httpx_http2():
    """HTTP/2 with httpx (OpenRouter)."""
    if not HAS_HTTPX:
        return []
    if not API_KEYS["OPENROUTER_API_KEY"]:
        print("  Skipped - no OPENROUTER_API_KEY")
        return []

    times = []
    client = httpx.Client(http2=True, timeout=30)

    # Warmup
    for _ in range(WARMUP_REQUESTS):
        client.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD)

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        response = client.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            times.append(elapsed)
            print(f"  HTTP/2 #{i+1}: {elapsed:.1f}ms")
        else:
            print(f"  HTTP/2 #{i+1}: ERROR {response.status_code}")

    client.close()
    return times


def benchmark_concurrent_hedged():
    """Simulate hedged requests - 2 concurrent calls, take first response."""
    if not API_KEYS["OPENROUTER_API_KEY"]:
        print("  Skipped - no OPENROUTER_API_KEY")
        return [], []

    import concurrent.futures

    times_fresh = []
    times_session = []

    session = requests.Session()

    def make_request_fresh():
        requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)

    def make_request_session():
        session.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)

    # Warmup
    session.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=OPENROUTER_PAYLOAD, timeout=30)

    print("\n  Hedged (2 concurrent) - Fresh connections:")
    for i in range(NUM_REQUESTS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            start = time.perf_counter()
            futures = [executor.submit(make_request_fresh) for _ in range(2)]
            # Wait for FIRST to complete (not both!)
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            elapsed = (time.perf_counter() - start) * 1000
            times_fresh.append(elapsed)
            print(f"    Pair #{i+1}: {elapsed:.1f}ms (first of 2)")
            # Let the other one finish in background
            for f in pending:
                f.cancel()

    print("\n  Hedged (2 concurrent) - Session:")
    for i in range(NUM_REQUESTS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            start = time.perf_counter()
            futures = [executor.submit(make_request_session) for _ in range(2)]
            # Wait for FIRST to complete (not both!)
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            elapsed = (time.perf_counter() - start) * 1000
            times_session.append(elapsed)
            print(f"    Pair #{i+1}: {elapsed:.1f}ms (first of 2)")
            for f in pending:
                f.cancel()

    session.close()
    return times_fresh, times_session


def benchmark_gemini_hedged():
    """Hedged requests directly to Gemini API."""
    if not API_KEYS["GEMINI_API_KEY"]:
        print("  Skipped - no GEMINI_API_KEY")
        return []

    import concurrent.futures

    times = []
    session = requests.Session()
    url = f"{GEMINI_URL}?key={API_KEYS['GEMINI_API_KEY']}"

    def make_request():
        session.post(url, json=GEMINI_PAYLOAD, timeout=30)

    # Warmup
    session.post(url, json=GEMINI_PAYLOAD, timeout=30)

    for i in range(NUM_REQUESTS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            start = time.perf_counter()
            futures = [executor.submit(make_request) for _ in range(2)]
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  Gemini Hedged #{i+1}: {elapsed:.1f}ms (first of 2)")
            for f in pending:
                f.cancel()

    session.close()
    return times


def print_stats(name, times):
    if not times:
        return
    print(f"\n{name}:")
    print(f"  Mean:   {statistics.mean(times):.1f}ms")
    print(f"  Median: {statistics.median(times):.1f}ms")
    print(f"  Stdev:  {statistics.stdev(times):.1f}ms" if len(times) > 1 else "")
    print(f"  Min:    {min(times):.1f}ms")
    print(f"  Max:    {max(times):.1f}ms")


def main():
    print("=" * 60)
    print("OpenRouter vs Direct Gemini API Benchmark")
    print("=" * 60)
    print(f"\nMaking {NUM_REQUESTS} requests per test (after {WARMUP_REQUESTS} warmup)")
    print(f"OpenRouter model: {OPENROUTER_PAYLOAD['model']}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print()

    print("1. OpenRouter (session, keep-alive):")
    openrouter_times = benchmark_openrouter_session()

    print("\n2. Gemini Direct (session, keep-alive):")
    gemini_times = benchmark_gemini_direct()

    print("\n3. Gemini Direct - Hedged (2 concurrent):")
    gemini_hedged_times = benchmark_gemini_hedged()

    print("\n4. OpenRouter - Fresh connections (for comparison):")
    fresh_times = benchmark_fresh_connections()

    if HAS_HTTPX:
        print("\n4. HTTP/2 (httpx):")
        http2_times = benchmark_httpx_http2()
    else:
        http2_times = []

    print("\n5. Hedged requests (2 concurrent):")
    hedged_fresh, hedged_session = benchmark_concurrent_hedged()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print_stats("OpenRouter (session)", openrouter_times)
    print_stats("Gemini Direct (session)", gemini_times)
    print_stats("Gemini Direct (hedged)", gemini_hedged_times)
    print_stats("OpenRouter (fresh)", fresh_times)
    if http2_times:
        print_stats("HTTP/2 (httpx)", http2_times)
    print_stats("OpenRouter Hedged - Fresh", hedged_fresh)
    print_stats("OpenRouter Hedged - Session", hedged_session)

    # Calculate overhead
    if openrouter_times and gemini_times:
        overhead = statistics.mean(openrouter_times) - statistics.mean(gemini_times)
        print(f"\n→ OpenRouter overhead vs direct Gemini: ~{overhead:.0f}ms")

    if gemini_times and gemini_hedged_times:
        hedging_benefit = statistics.mean(gemini_times) - statistics.mean(gemini_hedged_times)
        print(f"→ Gemini hedging benefit: ~{hedging_benefit:.0f}ms (costs 2x tokens)")

    if fresh_times and openrouter_times:
        savings = statistics.mean(fresh_times) - statistics.mean(openrouter_times)
        print(f"→ Session saves ~{savings:.0f}ms per request vs fresh connections")

    if http2_times and openrouter_times:
        diff = statistics.mean(openrouter_times) - statistics.mean(http2_times)
        print(f"→ HTTP/2 saves ~{diff:.0f}ms per request vs HTTP/1.1 session")


if __name__ == "__main__":
    main()
