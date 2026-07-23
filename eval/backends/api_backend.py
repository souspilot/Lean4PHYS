"""
OpenAI-compatible API backend for proof generation.
...
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import extract_code_blocks_as_list, write_to_json
from backends.rate_limiter import RateLimiter


def _build_extra_headers(base_url: str) -> Optional[Dict[str, str]]:
    if base_url and "openrouter.ai" in base_url:
        return {
            "HTTP-Referer": "https://localhost",
            "X-Title": "LeanPhysBench Proof Generation",
        }
    return None


def _single_proof_attempt(
    prompt_messages: List[Dict],
    model_name: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retry_limit: int = 3,
    rate_limiter: Optional[RateLimiter] = None,   # <-- new
) -> Dict:
    """Generate a single proof attempt via the API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_headers = _build_extra_headers(base_url)

    retries = 0
    while retries < retry_limit:
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()   # <-- blocks here until a ticket is free

            kwargs = {
                "model": model_name,
                "messages": prompt_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if not model_name.startswith("o1"):
                kwargs["top_p"] = top_p
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            response = client.chat.completions.create(**kwargs)

            if not response.choices:
                retries += 1
                print(f"[API] Empty choices list, retrying ({retries}/{retry_limit})")
                time.sleep(2)
                continue

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            resp_text = choice.message.content
            reasoning_text = getattr(choice.message, "reasoning", None)

            if finish_reason == "length":
                retries += 1
                print(f"[API] Truncated (finish_reason=length), retrying ({retries}/{retry_limit})")
                time.sleep(1)
                continue

            tokens = response.usage.completion_tokens if response.usage else 0

            blocks = extract_code_blocks_as_list(resp_text, code_type="lean4")
            if blocks != -1 and blocks:
                return {
                    "success": True,
                    "proof_text": blocks[-1],
                    "resp": resp_text,
                    "reasoning": reasoning_text,
                    "tokens": tokens,
                }
            else:
                retries += 1
                time.sleep(1)
        except Exception as e:
            retries += 1
            print(f"[API] Error: {e}")
            # A 429 means we guessed wrong about capacity -- back off harder
            # than a normal retry, since the bucket may be shared with other
            # traffic on this key (e.g. runs kicked off from another terminal).
            if "429" in str(e):
                time.sleep(15)
            else:
                time.sleep(2)

    return {"success": False}


def _process_single_record(
    rec: Dict,
    prompt_messages: List[Dict],
    model_name: str,
    base_url: str,
    api_key: str,
    ckpt_path: str,
    proof_num: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retry_limit: int = 3,
    attempt_workers: int = 4,
    rate_limiter: Optional[RateLimiter] = None,   # <-- new
) -> Dict:
    """Process a single dataset record: generate proof_num proofs with parallel attempts."""
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    rec.setdefault("Generated_proof", [])
    rec.setdefault("Proof_generation_log", [])
    rec.setdefault("Proof_attempts", rec.get("Proof_attempts", 0))

    record_file = Path(ckpt_path) / f"{rec['Name']}.json"

    if record_file.exists():
        with open(record_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("Generated_proof", "Proof_generation_log", "Proof_attempts"):
            rec[key] = existing.get(key, rec[key])
        remaining = proof_num - rec.get("Proof_attempts", 0)
        if remaining <= 0:
            return rec
    else:
        remaining = proof_num

    with ProcessPoolExecutor(max_workers=attempt_workers) as executor:
        futures = [
            executor.submit(
                _single_proof_attempt,
                prompt_messages, model_name, base_url, api_key,
                max_tokens, temperature, top_p, retry_limit,
                rate_limiter,   # <-- new
            )
            for _ in range(remaining)
        ]

        for f in as_completed(futures):
            result = f.result()
            rec["Proof_attempts"] += 1
            if result.get("success"):
                proof_text = result["proof_text"]
                rec["Generated_proof"].append(proof_text)
                rec["Proof_generation_log"].append({
                    "generation_idx": hashlib.sha256(proof_text.encode("utf-8")).hexdigest(),
                    "generated_content": result["resp"],
                    "reasoning_content": result.get("reasoning"),
                    "generated_token_num": result["tokens"],
                    "generated_proof": proof_text,
                })
                write_to_json(str(record_file), rec)

    return rec


class APIBackend:
    """OpenAI-compatible API backend for batch proof generation."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

    def generate_proofs(
        self,
        dataset: List[Dict],
        prompt_messages_fn,
        *,
        proof_num: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 14000,
        ckpt_path: str = "./checkpoints",
        dataset_workers: int = 4,
        attempt_workers: int = 4,
        retry_limit: int = 3,
        requests_per_minute: Optional[int] = None,   # <-- new
    ) -> List[Dict]:
        """
        ...
        requests_per_minute: If set, all workers share a single token-bucket
            limiter capping total requests/minute across every process
            (needed for OpenRouter free-tier's 20 RPM cap). None = unlimited.
        """
        rate_limiter = (
            RateLimiter.create(requests_per_minute)
            if requests_per_minute else None
        )

        results = []
        with ProcessPoolExecutor(max_workers=dataset_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_record,
                    rec, prompt_messages_fn(rec),
                    self.model_name, self.base_url, self.api_key,
                    ckpt_path, proof_num, max_tokens, temperature, top_p,
                    retry_limit, attempt_workers,
                    rate_limiter,   # <-- new
                ): rec
                for rec in dataset
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc="Generating proofs"):
                results.append(f.result())

        return results
