"""
Anthropic Batch API runner for Opus — 50% cost savings.

Uses step-at-a-time batching:
  1. Decompose each scaffold into discrete steps
  2. Collect all cases at the same step
  3. Submit to Messages Batches API (max 10K per batch)
  4. Poll for completion
  5. Process results, advance each case
  6. Repeat until all cases are done

Produces results compatible with the real-time experiment runner's
checkpoint and results format.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from pipeline.config import (
    EXPERIMENT_PARAMS,
    MODELS,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    LOGS_DIR,
)
from pipeline.scaffolds.base import ScaffoldResult
from pipeline.scaffolds.react import TOOL_DESCRIPTIONS, _ACTION_RE, _TOOL_DISPATCH
from pipeline.scaffolds.multi_agent import (
    CRITIC_SYSTEM_PROMPT,
    REVISION_INSTRUCTION,
)
from pipeline.scaffolds.map_reduce import (
    DECOMPOSE_SYSTEM,
    CHUNK_SYSTEM_TEMPLATE,
    REDUCE_SYSTEM_TEMPLATE,
)

logger = logging.getLogger("experiment")

# Anthropic batch limits
BATCH_MAX_REQUESTS = 10_000

# Opus batch pricing (50% of standard)
OPUS_BATCH_COST_PER_1K_INPUT = 0.0075    # $7.50/M
OPUS_BATCH_COST_PER_1K_OUTPUT = 0.0375   # $37.50/M


# ---------------------------------------------------------------------------
# Case state tracking
# ---------------------------------------------------------------------------

@dataclass
class BatchCaseState:
    """Tracks a single case through multi-step batch processing."""
    # Identity
    config_id: str
    benchmark_id: str
    case_id: str
    context_condition: str
    # Prompts
    system_prompt: str
    user_prompt: str
    original_prompt: str          # pre-wrapping prompt (for blinding)
    case_data: dict[str, Any]     # full case dict
    case_metadata: dict[str, Any] # scoring-relevant fields
    # Step tracking
    step: int = 0
    messages_history: list[dict[str, str]] = field(default_factory=list)
    intermediate_data: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    # Aggregates
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    n_api_calls: int = 0
    call_ids: list[str] = field(default_factory=list)
    scaffold_metadata: dict[str, Any] = field(default_factory=dict)
    # Completion
    final_response: str = ""
    is_complete: bool = False
    error: str | None = None

    # Abbreviation maps for compact custom_id (must be <=64 chars, [a-zA-Z0-9_-])
    _CFG_ABBR = {"direct": "dir", "react": "rea", "multi_agent": "ma", "map_reduce": "mr"}
    _BM_ABBR = {"truthfulqa": "tqa", "bbq": "bbq", "sycophancy": "syc", "xstest_orbench": "xst"}
    _CTX_ABBR = {"short": "s", "long": "l"}

    @property
    def custom_id(self) -> str:
        """Unique key for batch API.  Must match ^[a-zA-Z0-9_-]{1,64}$."""
        cfg = self._CFG_ABBR.get(self.config_id, self.config_id[:3])
        bm = self._BM_ABBR.get(self.benchmark_id, self.benchmark_id[:3])
        ctx = self._CTX_ABBR.get(self.context_condition, self.context_condition[0])
        return f"{cfg}-{bm}-{self.case_id}-{ctx}"


# ---------------------------------------------------------------------------
# Scaffold steppers — decompose each scaffold into discrete request steps
# ---------------------------------------------------------------------------

class ScaffoldStepper:
    """Base class for scaffold step logic."""

    config_id: str = ""

    def prepare_requests(self, state: BatchCaseState) -> list[dict[str, Any]]:
        """Return list of batch request params dicts.

        Most steps produce 1 request.  Map-reduce chunk step may produce
        multiple requests per case.  Returns [] if case is complete.

        Each dict must have keys: custom_id, params.
        """
        raise NotImplementedError

    def process_response(
        self,
        state: BatchCaseState,
        custom_id: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Update state with a response.  Set is_complete=True when done."""
        raise NotImplementedError


class DirectStepper(ScaffoldStepper):
    config_id = "direct"

    def prepare_requests(self, state: BatchCaseState) -> list[dict[str, Any]]:
        if state.step > 0:
            return []
        return [{
            "custom_id": f"{state.custom_id}-s0",
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                "system": state.system_prompt,
                "messages": [{"role": "user", "content": state.user_prompt}],
                "temperature": EXPERIMENT_PARAMS.temperature,
            },
        }]

    def process_response(self, state, custom_id, response_text, inp, out):
        state.trace = [
            {"role": "system", "content": state.system_prompt},
            {"role": "user", "content": state.user_prompt},
            {"role": "assistant", "content": response_text},
        ]
        state.final_response = response_text
        state.scaffold_metadata = {"scaffold": "direct"}
        state.is_complete = True
        state.step += 1


class ReActStepper(ScaffoldStepper):
    config_id = "react"

    def prepare_requests(self, state: BatchCaseState) -> list[dict[str, Any]]:
        max_iters = EXPERIMENT_PARAMS.react_max_iterations
        if state.step >= max_iters:
            # Hit iteration cap — finalize with last assistant message
            state.is_complete = True
            if not state.final_response:
                for entry in reversed(state.trace):
                    if entry.get("role") == "assistant":
                        state.final_response = entry["content"]
                        break
            return []

        augmented_system = f"{state.system_prompt}\n\n{TOOL_DESCRIPTIONS}"

        if state.step == 0:
            # Initial request
            messages = [{"role": "user", "content": state.user_prompt}]
            state.messages_history = list(messages)
            state.trace = [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": state.user_prompt},
            ]
        else:
            # Continuation — messages_history already has observation appended
            messages = list(state.messages_history)

        return [{
            "custom_id": f"{state.custom_id}-s{state.step}",
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                "system": augmented_system,
                "messages": messages,
                "temperature": EXPERIMENT_PARAMS.temperature,
            },
        }]

    def process_response(self, state, custom_id, response_text, inp, out):
        state.trace.append({"role": "assistant", "content": response_text})
        state.messages_history.append({"role": "assistant", "content": response_text})

        # Check for tool call
        match = _ACTION_RE.search(response_text)
        if match is None:
            # No tool call — final answer
            state.final_response = response_text
            state.scaffold_metadata = {
                "scaffold": "react",
                "iterations_used": state.step + 1,
                "max_iterations": EXPERIMENT_PARAMS.react_max_iterations,
            }
            state.is_complete = True
        else:
            # Execute tool locally, prepare next step
            tool_name = match.group(1)
            tool_arg = match.group(2).strip()
            tool_fn = _TOOL_DISPATCH.get(tool_name)
            if tool_fn is None:
                observation = f"Error: unknown tool '{tool_name}'."
            else:
                observation = tool_fn(tool_arg, state.user_prompt)

            obs_msg = f"Observation: {observation}"
            state.trace.append({"role": "tool", "content": obs_msg, "tool": tool_name})
            state.messages_history.append({"role": "user", "content": obs_msg})

        state.step += 1


class MultiAgentStepper(ScaffoldStepper):
    config_id = "multi_agent"

    def prepare_requests(self, state: BatchCaseState) -> list[dict[str, Any]]:
        if state.step == 0:
            # Step 0: Primary initial generation
            state.trace = [
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": state.user_prompt},
            ]
            return [{
                "custom_id": f"{state.custom_id}-s0p",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                    "system": state.system_prompt,
                    "messages": [{"role": "user", "content": state.user_prompt}],
                    "temperature": EXPERIMENT_PARAMS.temperature,
                },
            }]
        elif state.step == 1:
            # Step 1: Critic evaluation
            primary_response = state.intermediate_data.get("primary_response", "")
            return [{
                "custom_id": f"{state.custom_id}-s1c",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                    "system": CRITIC_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": (
                        f"Original question:\n{state.user_prompt}\n\n"
                        f"AI response to evaluate:\n{primary_response}"
                    )}],
                    "temperature": EXPERIMENT_PARAMS.temperature,
                },
            }]
        elif state.step == 2:
            # Step 2: Revision (only if critic said NEEDS_REVISION or UNSAFE)
            verdict = state.intermediate_data.get("critic_verdict", "")
            if verdict.upper() == "SAFE":
                state.final_response = state.intermediate_data["primary_response"]
                state.is_complete = True
                return []

            critique_text = state.intermediate_data.get("critique_text", "")
            primary_response = state.intermediate_data.get("primary_response", "")
            revision_prompt = REVISION_INSTRUCTION.format(
                critique=critique_text,
                original_prompt=state.user_prompt,
                original_response=primary_response,
            )
            return [{
                "custom_id": f"{state.custom_id}-s2r",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                    "system": state.system_prompt,
                    "messages": [{"role": "user", "content": revision_prompt}],
                    "temperature": EXPERIMENT_PARAMS.temperature,
                },
            }]
        else:
            return []

    def process_response(self, state, custom_id, response_text, inp, out):
        if state.step == 0:
            # Primary response received
            state.intermediate_data["primary_response"] = response_text
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": "primary", "step": "initial",
            })
            state.step = 1

        elif state.step == 1:
            # Critic response received
            state.intermediate_data["critique_text"] = response_text
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": "critic", "step": "critique_round_0",
            })
            # Parse verdict
            verdict = "NEEDS_REVISION"
            if "VERDICT:" in response_text:
                for line in response_text.split("\n"):
                    if "VERDICT:" in line:
                        verdict = line.split("VERDICT:")[-1].strip()
                        break
            state.intermediate_data["critic_verdict"] = verdict
            state.step = 2

            # If SAFE, finalize now (prepare_requests will handle)
            if verdict.upper() == "SAFE":
                state.final_response = state.intermediate_data["primary_response"]
                state.scaffold_metadata = {
                    "scaffold": "multi_agent",
                    "critic_verdict": verdict,
                    "revision_rounds": 0,
                }
                state.is_complete = True
                state.trace.append({
                    "role": "info",
                    "content": "Critic verdict SAFE. Skipping revision.",
                })

        elif state.step == 2:
            # Revision response received
            state.final_response = response_text
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": "primary", "step": "revision_round_0",
            })
            state.scaffold_metadata = {
                "scaffold": "multi_agent",
                "critic_verdict": state.intermediate_data.get("critic_verdict", ""),
                "revision_rounds": 1,
            }
            state.is_complete = True
            state.step = 3


class MapReduceStepper(ScaffoldStepper):
    config_id = "map_reduce"

    def _parse_decomposition(self, text: str) -> list[str]:
        """Extract sub-questions from decomposition response."""
        max_chunks = EXPERIMENT_PARAMS.map_reduce_max_chunks
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group())
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return parsed[:max_chunks]
            except json.JSONDecodeError:
                pass
        lines = [
            line.strip().lstrip("0123456789.-) ").strip('"').strip()
            for line in text.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        if lines:
            return lines[:max_chunks]
        return [text.strip()]

    def prepare_requests(self, state: BatchCaseState) -> list[dict[str, Any]]:
        if state.step == 0:
            # Step 0: Decompose
            state.trace = [
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": state.user_prompt},
            ]
            return [{
                "custom_id": f"{state.custom_id}-s0d",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                    "system": DECOMPOSE_SYSTEM,
                    "messages": [{"role": "user", "content": state.user_prompt}],
                    "temperature": EXPERIMENT_PARAMS.temperature,
                },
            }]
        elif state.step == 1:
            # Step 1: Map chunks (multiple requests per case!)
            sub_questions = state.intermediate_data.get("sub_questions", [])
            chunk_system = CHUNK_SYSTEM_TEMPLATE.format(system_prompt=state.system_prompt)
            requests = []
            for i, sub_q in enumerate(sub_questions):
                requests.append({
                    "custom_id": f"{state.custom_id}-s1k{i}",
                    "params": {
                        "model": "claude-opus-4-6",
                        "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                        "system": chunk_system,
                        "messages": [{"role": "user", "content": sub_q}],
                        "temperature": EXPERIMENT_PARAMS.temperature,
                    },
                })
            return requests
        elif state.step == 2:
            # Step 2: Reduce
            sub_questions = state.intermediate_data.get("sub_questions", [])
            chunk_responses = state.intermediate_data.get("chunk_responses", [])
            chunk_summaries = "\n\n".join(
                f"Sub-question {i+1}: {sq}\nAnalysis: {resp}"
                for i, (sq, resp) in enumerate(zip(sub_questions, chunk_responses))
            )
            reduce_prompt = REDUCE_SYSTEM_TEMPLATE.format(
                system_prompt=state.system_prompt,
                chunk_summaries=chunk_summaries,
                original_prompt=state.user_prompt,
            )
            return [{
                "custom_id": f"{state.custom_id}-s2x",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": EXPERIMENT_PARAMS.max_tokens,
                    "system": reduce_prompt,
                    "messages": [{"role": "user", "content": "Please provide your final synthesized answer."}],
                    "temperature": EXPERIMENT_PARAMS.temperature,
                },
            }]
        else:
            return []

    def process_response(self, state, custom_id, response_text, inp, out):
        if state.step == 0:
            # Decomposition received
            sub_questions = self._parse_decomposition(response_text)
            state.intermediate_data["sub_questions"] = sub_questions
            state.intermediate_data["chunk_responses"] = [None] * len(sub_questions)
            state.intermediate_data["chunks_received"] = 0
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": "decomposer", "step": "decompose",
            })
            state.step = 1

        elif state.step == 1:
            # A chunk response received — identify which chunk
            # custom_id ends with |step1_chunkN
            chunk_match = re.search(r"s1k(\d+)$", custom_id)
            if chunk_match:
                chunk_idx = int(chunk_match.group(1))
            else:
                chunk_idx = state.intermediate_data["chunks_received"]

            chunk_responses = state.intermediate_data["chunk_responses"]
            if chunk_idx < len(chunk_responses):
                chunk_responses[chunk_idx] = response_text
            state.intermediate_data["chunks_received"] += 1

            sub_questions = state.intermediate_data["sub_questions"]
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": f"mapper_{chunk_idx}",
                "step": f"map_chunk_{chunk_idx}",
                "sub_question": sub_questions[chunk_idx] if chunk_idx < len(sub_questions) else "",
            })

            # Check if all chunks received
            if state.intermediate_data["chunks_received"] >= len(sub_questions):
                state.step = 2

        elif state.step == 2:
            # Reduce response received
            state.final_response = response_text
            state.trace.append({
                "role": "assistant", "content": response_text,
                "agent": "reducer", "step": "reduce",
            })
            state.scaffold_metadata = {
                "scaffold": "map_reduce",
                "n_chunks": len(state.intermediate_data.get("sub_questions", [])),
                "sub_questions": state.intermediate_data.get("sub_questions", []),
                "max_chunks": EXPERIMENT_PARAMS.map_reduce_max_chunks,
            }
            state.is_complete = True
            state.step = 3


# Stepper registry
STEPPER_REGISTRY: dict[str, ScaffoldStepper] = {
    "direct": DirectStepper(),
    "react": ReActStepper(),
    "multi_agent": MultiAgentStepper(),
    "map_reduce": MapReduceStepper(),
}


# ---------------------------------------------------------------------------
# Batch runner orchestrator
# ---------------------------------------------------------------------------

class BatchRunner:
    """Orchestrates Opus experiment via Anthropic Messages Batches API."""

    def __init__(self, poll_interval: int = 30, dry_run: bool = False):
        self.client = anthropic.Anthropic()
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self._total_batches = 0
        self._total_requests = 0

    def run(
        self,
        cases: list[BatchCaseState],
        on_complete: Any | None = None,
    ) -> list[BatchCaseState]:
        """Process all cases through step-at-a-time batching.

        Parameters
        ----------
        cases : list[BatchCaseState]
            Cases to process (may have been partially advanced by recovery).
        on_complete : callable | None
            Called with (case: BatchCaseState) for each newly completed/errored
            case.  Useful for incremental result saving.

        Returns the same list with is_complete=True and results populated.
        """
        previously_done = {id(c) for c in cases if c.is_complete or c.error}

        wave = 0
        while True:
            active = [c for c in cases if not c.is_complete and c.error is None]
            if not active:
                break

            wave += 1
            # Collect all requests for the current step of each active case
            all_requests: list[dict[str, Any]] = []
            request_case_map: dict[str, BatchCaseState] = {}

            for case in active:
                stepper = STEPPER_REGISTRY[case.config_id]
                reqs = stepper.prepare_requests(case)
                if not reqs:
                    # Stepper said no more requests needed (may have set is_complete)
                    continue
                for req in reqs:
                    all_requests.append(req)
                    request_case_map[req["custom_id"]] = case

            if not all_requests:
                break

            _log(f"Wave {wave}: {len(all_requests)} requests across "
                 f"{len(set(id(c) for c in request_case_map.values()))} cases")

            if self.dry_run:
                self._process_dry_run(all_requests, request_case_map)
            else:
                self._process_batch(all_requests, request_case_map)

            # Notify about newly completed/errored cases
            if on_complete:
                for case in cases:
                    if (case.is_complete or case.error is not None) and id(case) not in previously_done:
                        on_complete(case)
                        previously_done.add(id(case))

        completed = sum(1 for c in cases if c.is_complete)
        errored = sum(1 for c in cases if c.error is not None)
        _log(f"Batch runner done: {completed} completed, {errored} errors, "
             f"{self._total_batches} batches, {self._total_requests} API requests")

        return cases

    def _process_batch(
        self,
        requests: list[dict[str, Any]],
        case_map: dict[str, BatchCaseState],
    ) -> None:
        """Submit requests in batches of 10K, poll, and process results."""
        for batch_start in range(0, len(requests), BATCH_MAX_REQUESTS):
            chunk = requests[batch_start:batch_start + BATCH_MAX_REQUESTS]
            self._total_batches += 1
            self._total_requests += len(chunk)

            _log(f"Submitting batch {self._total_batches} with {len(chunk)} requests")

            try:
                batch = self.client.messages.batches.create(requests=chunk)
            except Exception as e:
                _log(f"Batch submission failed: {e}", level="error")
                # Mark cases in this chunk as errored (only if not already handled)
                for req in chunk:
                    case = case_map.get(req["custom_id"])
                    if case and not case.error and not case.is_complete:
                        case.error = f"Batch submission failed: {e}"
                continue  # try remaining chunks

            _log(f"Batch {batch.id} submitted, polling...")

            # Poll for completion
            while True:
                status = self.client.messages.batches.retrieve(batch.id)
                counts = status.request_counts
                _log(f"Batch {batch.id}: processing={counts.processing}, "
                     f"succeeded={counts.succeeded}, errored={counts.errored}, "
                     f"expired={counts.expired}")
                if status.processing_status == "ended":
                    break
                time.sleep(self.poll_interval)

            # Process results
            for result in self.client.messages.batches.results(batch.id):
                cid = result.custom_id
                case = case_map.get(cid)
                if case is None:
                    _log(f"WARNING: Unknown custom_id in results: {cid}", level="warning")
                    continue

                if result.result.type == "succeeded":
                    msg = result.result.message
                    text = ""
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text += block.text
                    inp = msg.usage.input_tokens
                    out = msg.usage.output_tokens
                    cost = (
                        (inp / 1000) * OPUS_BATCH_COST_PER_1K_INPUT
                        + (out / 1000) * OPUS_BATCH_COST_PER_1K_OUTPUT
                    )

                    case.total_input_tokens += inp
                    case.total_output_tokens += out
                    case.total_cost_usd += cost
                    case.n_api_calls += 1
                    case.call_ids.append(str(uuid.uuid4()))

                    stepper = STEPPER_REGISTRY[case.config_id]
                    stepper.process_response(case, cid, text, inp, out)
                else:
                    error_type = result.result.type
                    error_msg = ""
                    if hasattr(result.result, "error"):
                        error_msg = str(result.result.error)
                    case.error = f"Batch {error_type}: {error_msg}"
                    _log(f"Case {cid} failed: {case.error}", level="error")

    def _process_dry_run(
        self,
        requests: list[dict[str, Any]],
        case_map: dict[str, BatchCaseState],
    ) -> None:
        """Simulate batch processing without API calls."""
        self._total_batches += 1
        self._total_requests += len(requests)

        for req in requests:
            cid = req["custom_id"]
            case = case_map.get(cid)
            if case is None:
                continue

            case.n_api_calls += 1
            case.call_ids.append(str(uuid.uuid4()))

            stepper = STEPPER_REGISTRY[case.config_id]
            # Simulate a plausible response per scaffold step
            if case.config_id == "direct":
                stepper.process_response(case, cid, "[DRY RUN] Direct response.", 0, 0)
            elif case.config_id == "react":
                stepper.process_response(case, cid, "[DRY RUN] Final answer without tools.", 0, 0)
            elif case.config_id == "multi_agent":
                if case.step == 0:
                    stepper.process_response(case, cid, "[DRY RUN] Primary response.", 0, 0)
                elif case.step == 1:
                    stepper.process_response(case, cid, "CRITIQUE:\n- All good.\nVERDICT: SAFE\nREVISION_GUIDANCE: No changes needed.", 0, 0)
                else:
                    stepper.process_response(case, cid, "[DRY RUN] Revised response.", 0, 0)
            elif case.config_id == "map_reduce":
                if case.step == 0:
                    stepper.process_response(case, cid, '["Sub-question 1", "Sub-question 2"]', 0, 0)
                elif "chunk" in cid:
                    stepper.process_response(case, cid, "[DRY RUN] Chunk response.", 0, 0)
                else:
                    stepper.process_response(case, cid, "[DRY RUN] Reduced response.", 0, 0)


def replay_recovered_results(
    cases: list[BatchCaseState],
    recovery_path: Path | str,
) -> int:
    """Replay previously saved batch results into case states.

    Reads a JSONL file (from batch API result recovery) and advances each
    case's stepper state accordingly, avoiding re-spending on already-
    completed API calls.

    Returns the number of results successfully replayed.
    """
    recovery_path = Path(recovery_path)
    if not recovery_path.exists():
        return 0

    # Build lookup: case.custom_id -> case
    case_lookup: dict[str, BatchCaseState] = {}
    for case in cases:
        case_lookup[case.custom_id] = case

    # Step suffix pattern matches all step formats:
    #   -s0, -s0p, -s1c, -s2r, -s0d, -s1k0, -s1k10, -s2x, etc.
    step_suffix_re = re.compile(r"-s\d+[a-z]*\d*$")

    replayed = 0
    with open(recovery_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            cid = result["custom_id"]

            # Strip step suffix to get case custom_id
            case_cid = step_suffix_re.sub("", cid)
            case = case_lookup.get(case_cid)
            if case is None:
                continue
            if case.is_complete or case.error:
                continue

            text = result.get("content", "")
            inp = result.get("input_tokens", 0)
            out = result.get("output_tokens", 0)
            cost = (
                (inp / 1000) * OPUS_BATCH_COST_PER_1K_INPUT
                + (out / 1000) * OPUS_BATCH_COST_PER_1K_OUTPUT
            )

            case.total_input_tokens += inp
            case.total_output_tokens += out
            case.total_cost_usd += cost
            case.n_api_calls += 1
            case.call_ids.append(str(uuid.uuid4()))

            stepper = STEPPER_REGISTRY[case.config_id]
            stepper.process_response(case, cid, text, inp, out)
            replayed += 1

    completed = sum(1 for c in cases if c.is_complete)
    _log(f"Replayed {replayed} recovered results: "
         f"{completed} cases now complete, "
         f"{sum(1 for c in cases if not c.is_complete and not c.error)} still need more steps")
    return replayed


def _log(message: str, level: str = "info") -> None:
    """Log via experiment logger or print."""
    if logger.handlers:
        getattr(logger, level)(f"[BatchRunner] {message}")
    else:
        print(f"[BatchRunner] {message}")
