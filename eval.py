"""
eval.py — Evaluation harness for the Confluence RAG Agent.

Pipeline
--------
1. Loads the local eval_dataset.json
2. Uploads each item to a named Langfuse dataset (creates it if it doesn't exist)
3. Runs a Langfuse experiment that:
   a. Calls ask_agent_with_context() for every dataset item, keeping the exact
      chunks the LLM saw so the judges score against the same context
   b. Scores each response with four LLM-as-a-judge evaluators:
      - faithfulness   : is every claim in the answer supported by the retrieved context?
      - relevance      : does the answer address every part of the question?
      - correctness    : does the answer contain the key facts of the ground truth?
      - context_recall : does the retrieved context contain the key facts of the reference?
                         (diagnostic — separates retrieval failures from generation failures)
4. Computes run-level averages and posts them back to Langfuse
5. Prints a human-readable summary to stdout

Each judge breaks its task into items (claims / question parts / facts) and
scores each one; the final score is the mean. This is far more stable than
asking a small model for a single holistic number.

Usage
-----
    python eval.py                          # run with defaults
    python eval.py --experiment "run-v2"   # custom experiment name

Environment variables required (same as the rest of the project):
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse, get_client
from langchain_ollama import OllamaLLM

from RAG_Agent import ask_agent_with_context

# ── Environment & logging ──────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "confluence-rag-eval"
DATASET_FILE = Path(__file__).parent / "eval_dataset.json"

# Scoring thresholds used in the summary report
PASS_THRESHOLD = 0.7   # score >= this is considered a pass

# Metrics an item must pass; context_recall is reported but is a retrieval
# diagnostic, not a judgement of the answer itself
PASS_METRICS = ["faithfulness", "relevance", "correctness"]
ALL_METRICS = PASS_METRICS + ["context_recall"]

# How many times to re-ask the judge if its output can't be parsed
JUDGE_MAX_ATTEMPTS = 3

# ── Judge LLM (reuses the same local Ollama model) ────────────────────────────
# temperature=0 makes verdicts reproducible; format="json" forces valid JSON output.
# A stronger judge model than the one being evaluated will give more reliable scores.
judge_llm = OllamaLLM(model="llama3.2", temperature=0, format="json")

# ── Judge prompt templates ─────────────────────────────────────────────────────
# Every judge returns the same shape so one parser handles all of them:
#   {"items": [{"text": "...", "score": 0 | 0.5 | 1}], "reason": "..."}
JUDGE_OUTPUT_FORMAT = """\
Return ONLY a JSON object in exactly this shape, nothing else:
{{"items": [{{"text": "<the item>", "score": <0, 0.5 or 1>}}], "reason": "<one sentence summary>"}}"""

FAITHFULNESS_PROMPT = """\
You are an objective evaluator. Your job is to check whether an answer is
faithful to the provided context — i.e. it makes no claims that are not
supported by the context.

Context (retrieved from the knowledge base):
{context}

Question:
{question}

Answer:
{answer}

Step 1: List every distinct factual claim made in the answer.
Step 2: Score each claim:
  1   = the claim is directly supported by the context
  0.5 = the claim is loosely implied by the context but not stated
  0   = the claim is not in the context or contradicts it
A statement that the documentation does not cover something counts as
supported (1) if the context indeed does not cover it.

""" + JUDGE_OUTPUT_FORMAT

RELEVANCE_PROMPT = """\
You are an objective evaluator. Your job is to check whether an answer
addresses every part of the question that was asked.

Question:
{question}

Answer:
{answer}

Step 1: Split the question into its distinct parts (sub-questions). A question
like "What is X, and when should Y happen?" has two parts.
Step 2: Score each part:
  1   = the answer addresses this part directly and specifically
  0.5 = the answer touches on this part but is vague or incomplete
  0   = the answer does not address this part

""" + JUDGE_OUTPUT_FORMAT

CORRECTNESS_PROMPT = """\
You are an objective evaluator. Your job is to check whether an answer contains
the key facts of a reference (ground truth) answer.

Question:
{question}

Reference answer:
{ground_truth}

Answer to evaluate:
{answer}

Step 1: List each key fact in the reference answer.
Step 2: Score each fact:
  1   = the answer states this fact (paraphrasing is fine)
  0.5 = the answer mentions it only vaguely or partially
  0   = the answer omits or contradicts this fact

""" + JUDGE_OUTPUT_FORMAT

CONTEXT_RECALL_PROMPT = """\
You are an objective evaluator. Your job is to check whether retrieved context
contains the information needed to answer a question.

Question:
{question}

Reference information:
{reference}

Retrieved context:
{context}

Step 1: List each key fact in the reference information.
Step 2: Score each fact:
  1   = the retrieved context contains this fact
  0.5 = the retrieved context contains it only partially
  0   = the retrieved context does not contain this fact

""" + JUDGE_OUTPUT_FORMAT

RETRY_SUFFIX = """

Your previous response could not be parsed. Respond with ONLY the JSON object
in the exact shape described above, with a non-empty "items" list."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_judge_output(raw: str) -> tuple[float, str] | None:
    """Extract (score, reason) from the judge LLM's JSON output.

    The score is the mean of the per-item scores. Returns None if the output
    can't be parsed or contains no scored items, so the caller can retry.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} block in case of surrounding prose
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning("Judge output did not contain JSON: %r", raw)
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse judge JSON (%s): %r", exc, raw)
            return None

    try:
        items = data.get("items") or []
        scores = [max(0.0, min(1.0, float(it["score"]))) for it in items]  # clamp to [0, 1]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Judge JSON has unexpected shape (%s): %r", exc, raw)
        return None
    if not scores:
        logger.warning("Judge returned no scored items: %r", raw)
        return None

    breakdown = "; ".join(
        f"{it.get('text', '?')} = {float(it['score']):g}" for it in items
    )
    reason = f"{data.get('reason', '')} [{breakdown}]".strip()
    return sum(scores) / len(scores), reason


def _run_judge(prompt_template: str, **kwargs) -> tuple[float | None, str]:
    """Fill a judge prompt template and call the local LLM.

    Retries with a format reminder if the output can't be parsed. Returns a
    score of None (rather than 0.0) if the judge never produces usable output,
    so judge failures don't drag down the averages.
    """
    prompt = prompt_template.format(**kwargs)
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            raw = judge_llm.invoke(prompt if attempt == 1 else prompt + RETRY_SUFFIX)
        except Exception as exc:
            logger.error("Judge LLM call failed: %s", exc)
            return None, f"Judge error: {exc}"
        parsed = _parse_judge_output(raw)
        if parsed is not None:
            return parsed
        logger.warning("Judge attempt %d/%d unparseable.", attempt, JUDGE_MAX_ATTEMPTS)
    return None, f"Judge output unparseable after {JUDGE_MAX_ATTEMPTS} attempts"


def _format_context(chunks: list[str]) -> str:
    """Join chunks with labels so the judge can tell them apart."""
    if not chunks:
        return "(no context retrieved)"
    return "\n---\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks, start=1))


def _to_evaluation(name: str, score: float | None, reason: str) -> Evaluation | list:
    """Wrap a judge result; unscored items return [] so no fake 0 is recorded."""
    logger.info("%s score=%s | reason=%s", name, score, reason)
    if score is None:
        return []
    return Evaluation(name=name, value=round(score, 4), comment=reason)


# ── Dataset upload ─────────────────────────────────────────────────────────────

def upload_dataset(langfuse: Langfuse) -> None:
    """Create the Langfuse dataset and upload items from eval_dataset.json.

    Idempotent — if the dataset already exists the call succeeds silently.
    Each item is upserted so running this script multiple times is safe.
    """
    logger.info("Loading dataset from %s", DATASET_FILE)
    items = json.loads(DATASET_FILE.read_text())

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Golden Q&A pairs for the Confluence RAG Agent (eligibility / EDI domain).",
    )
    logger.info("Dataset '%s' created/confirmed in Langfuse.", DATASET_NAME)

    for idx, item in enumerate(items, start=1):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
        )
    logger.info("Uploaded %d item(s) to dataset '%s'.", len(items), DATASET_NAME)


# ── Task function ──────────────────────────────────────────────────────────────

def rag_task(*, item, **kwargs) -> dict:
    """Run the RAG pipeline for one dataset item.

    Returns a dict so the evaluators can access both the generated answer and
    the exact chunks the LLM was given (needed for faithfulness and recall).
    """
    question = item.input["question"]
    logger.info("Running RAG for: %r", question[:80])

    answer, chunks = ask_agent_with_context(question)

    return {
        "answer": answer,
        "chunks": chunks,   # passed through to evaluators via `output`
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def faithfulness_evaluator(*, input, output, expected_output, **kwargs):
    """Score: is every claim in the answer grounded in the retrieved context?"""
    score, reason = _run_judge(
        FAITHFULNESS_PROMPT,
        context=_format_context(output.get("chunks", [])),
        question=input["question"],
        answer=output.get("answer", ""),
    )
    return _to_evaluation("faithfulness", score, reason)


def relevance_evaluator(*, input, output, expected_output, **kwargs):
    """Score: does the answer address every part of the question?"""
    score, reason = _run_judge(
        RELEVANCE_PROMPT,
        question=input["question"],
        answer=output.get("answer", ""),
    )
    return _to_evaluation("relevance", score, reason)


def correctness_evaluator(*, input, output, expected_output, **kwargs):
    """Score: does the answer contain the key facts of the ground truth?"""
    ground_truth = (expected_output or {}).get("ground_truth")
    if not ground_truth:
        return []
    score, reason = _run_judge(
        CORRECTNESS_PROMPT,
        question=input["question"],
        ground_truth=ground_truth,
        answer=output.get("answer", ""),
    )
    return _to_evaluation("correctness", score, reason)


def context_recall_evaluator(*, input, output, expected_output, **kwargs):
    """Score: did retrieval return the facts needed to answer the question?"""
    reference = (expected_output or {}).get("reference")
    if not reference:
        return []
    score, reason = _run_judge(
        CONTEXT_RECALL_PROMPT,
        question=input["question"],
        reference=reference,
        context=_format_context(output.get("chunks", [])),
    )
    return _to_evaluation("context_recall", score, reason)


# ── Run-level evaluators ───────────────────────────────────────────────────────

def _make_avg_evaluator(metric: str):
    """Build a run-level evaluator that averages one item-level metric."""

    def avg_evaluator(*, item_results, **kwargs) -> Evaluation:
        scores = [
            ev.value
            for r in item_results
            for ev in r.evaluations
            if ev.name == metric and ev.value is not None
        ]
        if not scores:
            return Evaluation(name=f"avg_{metric}", value=None)
        avg = sum(scores) / len(scores)
        return Evaluation(
            name=f"avg_{metric}",
            value=round(avg, 4),
            comment=f"Average {metric} across {len(scores)} item(s): {avg:.2%}",
        )

    avg_evaluator.__name__ = f"avg_{metric}"
    return avg_evaluator


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(result, experiment_name: str) -> None:
    """Print a human-readable eval summary to stdout and write it to a .txt file.

    The output file is always saved as:
        eval_results_<experiment_name>.txt
    in the same directory as this script.
    """
    output_path = Path(__file__).parent / f"eval_results_{experiment_name}.txt"
    lines: list[str] = []

    def tee(text: str = "") -> None:
        """Write a line to both stdout and the in-memory buffer."""
        print(text)
        lines.append(text)

    tee("\n" + "=" * 70)
    tee("  CONFLUENCE RAG AGENT — EVAL SUMMARY")
    tee(f"  Experiment  : {experiment_name}")
    tee(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee("=" * 70)

    item_results = result.item_results
    total = len(item_results)
    passed = 0

    for r in item_results:
        evs = {e.name: e for e in r.evaluations}

        item_pass = all(
            m in evs and evs[m].value is not None and evs[m].value >= PASS_THRESHOLD
            for m in PASS_METRICS
        )
        if item_pass:
            passed += 1

        status = "PASS" if item_pass else "FAIL"
        question = r.item.input.get("question", "")[:70]
        tee(f"\n  [{status}] Q: {question}...")
        for m in ALL_METRICS:
            ev = evs.get(m)
            if ev is not None and ev.value is not None:
                tee(f"         {m:<15}={ev.value:.2f}  ({ev.comment})")
            else:
                tee(f"         {m:<15}=n/a   (not scored — judge failed or no reference)")

    # Run-level scores
    run_evals = result.run_evaluations if hasattr(result, "run_evaluations") else []
    run_avgs = {e.name: e.value for e in run_evals}

    tee("\n" + "-" * 70)
    tee(f"  Total items : {total}")
    tee(f"  Passed      : {passed} / {total}  "
        f"(all of {', '.join(PASS_METRICS)} >= {PASS_THRESHOLD})")
    for m in ALL_METRICS:
        avg = run_avgs.get(f"avg_{m}")
        if avg is not None:
            tee(f"  Avg {m:<15}: {avg:.2%}")
    tee("=" * 70 + "\n")

    # Write to file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Eval summary written to %s", output_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Confluence RAG Agent eval.")
    parser.add_argument(
        "--experiment",
        default=f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Name of the experiment run in Langfuse (default: timestamped).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading items to Langfuse (use if dataset already exists).",
    )
    args = parser.parse_args()

    # Initialise Langfuse (reads keys from .env automatically)
    langfuse = get_client()

    # Step 1: upload dataset items
    if not args.skip_upload:
        upload_dataset(langfuse)
    else:
        logger.info("Skipping dataset upload (--skip-upload flag set).")

    # Step 2: fetch dataset and run experiment
    logger.info("Fetching dataset '%s' from Langfuse...", DATASET_NAME)
    dataset = langfuse.get_dataset(DATASET_NAME)

    logger.info("Starting experiment: '%s'", args.experiment)
    result = dataset.run_experiment(
        name=args.experiment,
        description=(
            "Automated eval: faithfulness + relevance + correctness + context recall "
            "scoring via llama3.2 judge."
        ),
        task=rag_task,
        evaluators=[
            faithfulness_evaluator,
            relevance_evaluator,
            correctness_evaluator,
            context_recall_evaluator,
        ],
        run_evaluators=[_make_avg_evaluator(m) for m in ALL_METRICS],
        max_concurrency=1,   # keep at 1 for local Ollama to avoid overloading
    )

    # Step 3: print summary (also writes to eval_results_<experiment>.txt)
    print_summary(result, experiment_name=args.experiment)

    # Step 4: flush Langfuse traces
    langfuse.flush()
    logger.info("All traces flushed to Langfuse. Experiment complete.")


if __name__ == "__main__":
    main()
