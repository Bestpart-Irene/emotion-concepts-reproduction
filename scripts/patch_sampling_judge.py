#!/usr/bin/env python3
"""Add a SAMPLING scoring path to OpenAIBackend so a local OpenAI-compatible judge
(SGLang/Qwen) scores correctly. The default logprob-on-first-token method reads only
the first token of the answer — fine when a 2-digit score like "85" is ONE token
(gpt-4.1-mini), but Qwen's tokenizer splits "85" into "8"+"5", so logprob scoring
reads 8 instead of 85 and everything collapses to 0-9. When require_logprobs=False,
generate the full number as text and parse it. Also make supports_logprobs() reflect
the flag, and let _make_backend read TRAIT_JUDGE_REQUIRE_LOGPROBS.

Run ON THE CLUSTER against the traitinterp checkout. Idempotent; keeps .bak files.
Repo path defaults to $TI_REMOTE_REPO or /scratch/$USER/traitinterp."""
import os, pathlib, sys, shutil

REPO = pathlib.Path(os.environ.get("TI_REMOTE_REPO") or f"/scratch/{os.environ.get('USER','')}/traitinterp")

# ---- 1. judge_backends.py: sampling path in OpenAIBackend ----
bp = REPO / "utils" / "judge_backends.py"
src = bp.read_text()

old_supports = "    def supports_logprobs(self) -> bool:\n        return True\n"
new_supports = "    def supports_logprobs(self) -> bool:\n        return self.require_logprobs\n"

old_score = '''    async def score_prompt(
        self,
        messages: List[Message],
        *,
        min_val: int = 0,
        max_val: int = 100,
        min_weight: float = 0.25,
        n_samples: int = 1,  # ignored
    ) -> Optional[float]:
        logprobs = await self._get_logprobs(messages)
        return aggregate_logprob_score(
            logprobs, min_weight=min_weight, min_val=min_val, max_val=max_val,
        )'''
new_score = '''    async def _sample_text(self, messages, *, max_tokens=8, temperature=0.0) -> str:
        """Generate a short text completion (no logprobs) — used for sampling-mode scoring."""
        try:
            response = await _retry_async(
                self._call, messages,
                max_tokens=max_tokens, temperature=temperature, logprobs=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI sampling call failed: {e}")
            return ""

    async def score_prompt(
        self,
        messages: List[Message],
        *,
        min_val: int = 0,
        max_val: int = 100,
        min_weight: float = 0.25,
        n_samples: int = 1,
    ) -> Optional[float]:
        # Sampling path (require_logprobs=False): read the FULL integer from generated
        # text. Needed for endpoints/tokenizers (SGLang/Qwen) where a 2-digit answer
        # tokenizes as separate digits and first-token logprob scoring truncates it.
        if not self.require_logprobs:
            n = max(1, n_samples)
            temp = 0.0 if n == 1 else 0.7
            texts = await asyncio.gather(*[
                self._sample_text(messages, max_tokens=8, temperature=temp) for _ in range(n)
            ])
            samples = [parse_integer_response(t, min_val=min_val, max_val=max_val) for t in texts]
            return aggregate_sampled_integers(samples)
        logprobs = await self._get_logprobs(messages)
        return aggregate_logprob_score(
            logprobs, min_weight=min_weight, min_val=min_val, max_val=max_val,
        )'''

changed = False
if new_supports in src:
    print("judge_backends: supports_logprobs already patched")
elif old_supports in src:
    src = src.replace(old_supports, new_supports, 1); changed = True
else:
    print("WARN: supports_logprobs pattern not found (skipping)")

if "_sample_text" in src:
    print("judge_backends: score_prompt sampling path already present")
elif old_score in src:
    src = src.replace(old_score, new_score, 1); changed = True
else:
    print("ERROR: score_prompt pattern not found — aborting"); sys.exit(1)

if changed:
    shutil.copy(str(bp), str(bp) + ".bak")
    bp.write_text(src)
    print(f"patched {bp} (backup .bak)")

# ---- 2. judge.py: _make_backend reads TRAIT_JUDGE_REQUIRE_LOGPROBS ----
jp = REPO / "utils" / "judge.py"
jsrc = jp.read_text()
old_openai = '''            model=model or os.environ.get("TRAIT_JUDGE_MODEL") or DEFAULT_OPENAI_MODEL,
            base_url=base_url,'''
new_openai = '''            model=model or os.environ.get("TRAIT_JUDGE_MODEL") or DEFAULT_OPENAI_MODEL,
            base_url=base_url,
            require_logprobs=os.environ.get("TRAIT_JUDGE_REQUIRE_LOGPROBS", "1").strip().lower() not in ("0", "false", "no"),'''
if "TRAIT_JUDGE_REQUIRE_LOGPROBS" in jsrc:
    print("judge.py: require_logprobs env already wired")
elif old_openai in jsrc:
    shutil.copy(str(jp), str(jp) + ".bak2")
    jp.write_text(jsrc.replace(old_openai, new_openai, 1))
    print(f"patched {jp} (backup .bak2); TRAIT_JUDGE_REQUIRE_LOGPROBS honored")
else:
    print("ERROR: judge.py openai-branch pattern not found (run patch_judge_shim.py first) — aborting"); sys.exit(1)

print("DONE")
