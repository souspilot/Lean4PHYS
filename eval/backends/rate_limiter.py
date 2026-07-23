"""
Cross-process rate limiter (token bucket) for API backends.

Needed because the pipeline uses ProcessPoolExecutor -- separate OS processes
that can't see each other's local state. A plain time.sleep() inside one
worker does nothing to slow down the other workers. This uses
multiprocessing.Manager to hold a shared "ticket count" that every process
checks against before making a request.

IMPORTANT: only the RateLimiter's proxy objects (lock, dict) get passed to
worker processes -- never the Manager itself. The Manager object holds
internal auth credentials that Python refuses to pickle across processes.
Keep the Manager alive as a local variable in the process that creates it.

Usage (in the main process only):
    rate_limiter, manager = create_rate_limiter(requests_per_minute=20)
    ...
    # pass rate_limiter (not manager) into ProcessPoolExecutor.submit(...)
    # keep `manager` alive (in scope) for as long as workers are running
"""

import time
from multiprocessing.managers import SyncManager


class RateLimiter:
    """
    Token-bucket limiter shared across processes.

    Only holds Manager *proxy* objects (a Lock proxy, a dict proxy) -- these
    are safe to pickle and pass into ProcessPoolExecutor workers. Never store
    a reference to the SyncManager itself on this object.
    """

    def __init__(self, lock, state, requests_per_minute: int):
        self._lock = lock
        self._state = state  # Manager dict: {"tokens": float, "last_refill": float}
        self._rpm = requests_per_minute

    def acquire(self):
        """Block until a request ticket is available, then take one."""
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._state["last_refill"]

                refill = elapsed * (self._rpm / 60.0)
                if refill > 0:
                    self._state["tokens"] = min(
                        self._rpm, self._state["tokens"] + refill
                    )
                    self._state["last_refill"] = now

                if self._state["tokens"] >= 1.0:
                    self._state["tokens"] -= 1.0
                    return

                deficit = 1.0 - self._state["tokens"]
                wait_time = deficit / (self._rpm / 60.0)

            time.sleep(min(wait_time, 1.0))


def create_rate_limiter(requests_per_minute: int = 20):
    """
    Start a Manager and build a RateLimiter handle.

    Returns (rate_limiter, manager). Pass `rate_limiter` into worker
    processes. Keep `manager` alive (just leave it in scope) for as long as
    any worker might still call rate_limiter.acquire() -- once `manager`
    itself gets garbage collected, the shared state disappears.
    """
    manager = SyncManager()
    manager.start()
    lock = manager.Lock()
    state = manager.dict()
    state["tokens"] = float(requests_per_minute)
    state["last_refill"] = time.time()
    rate_limiter = RateLimiter(lock, state, requests_per_minute)
    return rate_limiter, manager