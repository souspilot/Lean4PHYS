"""
Cross-process rate limiter (token bucket) for API backends.

Needed because the pipeline uses ProcessPoolExecutor -- separate OS processes
that can't see each other's local state. A plain time.sleep() inside one
worker does nothing to slow down the other workers. This uses
multiprocessing.Manager to hold a shared "ticket count" that every process
checks against before making a request.

Usage:
    limiter = RateLimiter.create(requests_per_minute=20)
    ...
    limiter.acquire()   # blocks until a ticket is available
    response = client.chat.completions.create(...)
"""

import time
from multiprocessing.managers import SyncManager


class RateLimiter:
    """
    Token-bucket limiter shared across processes via a Manager.

    Not instantiated directly -- use RateLimiter.create(...), which spins up
    the Manager and returns a lightweight handle. The handle itself IS
    picklable (it only holds Manager proxy objects), so it can be passed
    straight into ProcessPoolExecutor.submit(...) like any other argument.
    """

    def __init__(self, lock, state, requests_per_minute: int):
        self._lock = lock
        self._state = state  # Manager dict: {"tokens": float, "last_refill": float}
        self._rpm = requests_per_minute

    @classmethod
    def create(cls, requests_per_minute: int = 20) -> "RateLimiter":
        manager = SyncManager()
        manager.start()
        lock = manager.Lock()
        state = manager.dict()
        state["tokens"] = float(requests_per_minute)
        state["last_refill"] = time.time()
        # Keep a reference so the manager process isn't garbage collected.
        limiter = cls(lock, state, requests_per_minute)
        limiter._manager = manager
        return limiter

    def acquire(self):
        """Block until a request ticket is available, then take one."""
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._state["last_refill"]

                # Refill tokens continuously based on elapsed time
                # (e.g. after 3 seconds at 20/min, ~1 new token has accrued).
                refill = elapsed * (self._rpm / 60.0)
                if refill > 0:
                    self._state["tokens"] = min(
                        self._rpm, self._state["tokens"] + refill
                    )
                    self._state["last_refill"] = now

                if self._state["tokens"] >= 1.0:
                    self._state["tokens"] -= 1.0
                    return

                # Not enough tokens yet -- figure out how long until one frees up.
                deficit = 1.0 - self._state["tokens"]
                wait_time = deficit / (self._rpm / 60.0)

            time.sleep(min(wait_time, 1.0))  # recheck at least once a second