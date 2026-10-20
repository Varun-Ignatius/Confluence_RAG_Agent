"""
eval.py — Evaluation harness for the Confluence RAG Agent.

Pipeline
--------
1. Loads the local eval_dataset.json
2. Uploads each item to a named Langfuse dataset (creates it if it doesn't exist)
3. Runs a Langfuse experiment that:
   a. Calls ask_agent() for every dataset item
   b. Scores each response with two LLM-as-a-judge evaluators:
      - faithfulness  : does the answer stay within the retrieved context?
      - relevance     : does the answer actually address the question?
4. Computes run-level averages and posts them back to Langfuse
5. Prints a human-readable summary to stdout

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

from RAG_Agent import ask_agent, build_prompt
from retriever import retrievel

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

# ── Judge LLM (reuses the same local Ollama model) ────────────────────────────
judge_llm = OllamaLLM(model="llama3.2")

# ── Judge prompt templates ─────────────────────────────────────────────────────
FAITHFULNESS_PROMPT = """\
You are an objective evaluator. Your job is to assess whether an answer is
faithful to the provided context — i.e. it makes no claims that are not
supported by the context.

Context (retrieved from the knowledge base):
{context}

Question:
{question}

Answer:
{answer}

Score the faithfulness on a scale from 0.0 to 1.0:
  1.0 = every claim in the answer is directly supported by the context
  0.5 = answer is partially supported; some claims are not in the context
  0.0 = answer contradicts or ignores the context entirely

Return ONLY a JSON object with two fields, nothing else:
{{"score": <float>, "reason": "<one sentence>"}}"""

RELEVANCE_PROMPT = """\
You are an objective evaluator. Your job is to assess whether an answer
actually addresses the question that was asked.

Question:
{question}

Answer:
{answer}

Score the relevance on a scale from 0.0 to 1.0:
  1.0 = the answer directly and completely addresses the question
  0.5 = the answer is related but misses key parts of the question
  0.0 = the answer does not address the question at all

Return ONLY a JSON object with two fields, nothing else:
{{"score": <float>, "reason": "<one sentence>"}}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_judge_output(raw: str) -> tuple[float, str]:
    """Extract (score, reason) from the judge LLM's JSON output.

    Falls back gracefully if the model wraps JSON in markdown fences or adds
    extra prose before/after the JSON block.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("Judge output did not contain valid JSON: %r", raw)
        return 0.0, "Could not parse judge output"
    try:
        data = json.loads(match.group())
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))
        return max(0.0, min(1.0, score)), reason  # clamp to [0, 1]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse judge JSON (%s): %r", exc, raw)
        return 0.0, "Parse error"


def _run_judge(prompt_template: str, **kwargs) -> tuple[float, str]:
    """Fill a judge prompt template and call the local LLM."""
    prompt = prompt_template.format(**kwargs)
    try:
        raw = judge_llm.invoke(prompt)
        return _parse_judge_output(raw)
    except Exception as exc:
        logger.error("Judge LLM call failed: %s", exc)
        return 0.0, f"Judge error: {exc}"


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
    the retrieved chunks (needed for the faithfulness judge).
    """
    question = item.input["question"]
    logger.info("Running RAG for: %r", question[:80])

    chunks = retrievel(question, n_results=3)
    answer = ask_agent(question)

    return {
        "answer": answer,
        "chunks": chunks,   # passed through to evaluators via `output`
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def faithfulness_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Score: is the answer grounded in the retrieved context?"""
    question = input["question"]
    answer = output.get("answer", "")
    chunks = output.get("chunks", [])
    context = "\n---\n".join(chunks) if chunks else "(no context retrieved)"

    score, reason = _run_judge(
        FAITHFULNESS_PROMPT,
        context=context,
        question=question,
        answer=answer,
    )
    logger.info("Faithfulness score=%.2f | reason=%s", score, reason)
    return Evaluation(name="faithfulness", value=score, comment=reason)


def relevance_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Score: does the answer actually address the question?"""
    question = input["question"]
    answer = output.get("answer", "")

    score, reason = _run_judge(
        RELEVANCE_PROMPT,
        question=question,
        answer=answer,
    )
    logger.info("Relevance score=%.2f | reason=%s", score, reason)
    return Evaluation(name="relevance", value=score, comment=reason)


# ── Run-level evaluators ───────────────────────────────────────────────────────

def avg_faithfulness(*, item_results, **kwargs) -> Evaluation:
    """Compute average faithfulness across all items in the run."""
    scores = [
        ev.value
        for r in item_results
        for ev in r.evaluations
        if ev.name == "faithfulness" and ev.value is not None
    ]
    if not scores:
        return Evaluation(name="avg_faithfulness", value=None)
    avg = sum(scores) / len(scores)
    return Evaluation(
        name="avg_faithfulness",
        value=round(avg, 4),
        comment=f"Average faithfulness across {len(scores)} item(s): {avg:.2%}",
    )


def avg_relevance(*, item_results, **kwargs) -> Evaluation:
    """Compute average relevance across all items in the run."""
    scores = [
        ev.value
        for r in item_results
        for ev in r.evaluations
        if ev.name == "relevance" and ev.value is not None
    ]
    if not scores:
        return Evaluation(name="avg_relevance", value=None)
    avg = sum(scores) / len(scores)
    return Evaluation(
        name="avg_relevance",
        value=round(avg, 4),
        comment=f"Average relevance across {len(scores)} item(s): {avg:.2%}",
    )


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
        faith_ev = next((e for e in r.evaluations if e.name == "faithfulness"), None)
        rel_ev   = next((e for e in r.evaluations if e.name == "relevance"),    None)

        faith_score = faith_ev.value if faith_ev else None
        rel_score   = rel_ev.value   if rel_ev   else None

        item_pass = (
            (faith_score is not None and faith_score >= PASS_THRESHOLD) and
            (rel_score   is not None and rel_score   >= PASS_THRESHOLD)
        )
        if item_pass:
            passed += 1

        status = "PASS" if item_pass else "FAIL"
        question = r.item.input.get("question", "")[:70]
        tee(f"\n  [{status}] Q: {question}...")
        if faith_score is not None:
            tee(f"         faithfulness={faith_score:.2f}  ({faith_ev.comment})")
        if rel_score is not None:
            tee(f"         relevance   ={rel_score:.2f}  ({rel_ev.comment})")

    # Run-level scores
    run_evals = result.run_evaluations if hasattr(result, "run_evaluations") else []
    avg_f = next((e.value for e in run_evals if e.name == "avg_faithfulness"), None)
    avg_r = next((e.value for e in run_evals if e.name == "avg_relevance"),    None)

    tee("\n" + "-" * 70)
    tee(f"  Total items : {total}")
    tee(f"  Passed      : {passed} / {total}  (threshold >= {PASS_THRESHOLD})")
    if avg_f is not None:
        tee(f"  Avg Faithfulness : {avg_f:.2%}")
    if avg_r is not None:
        tee(f"  Avg Relevance    : {avg_r:.2%}")
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
        description="Automated eval: faithfulness + relevance scoring via llama3.2 judge.",
        task=rag_task,
        evaluators=[faithfulness_evaluator, relevance_evaluator],
        run_evaluators=[avg_faithfulness, avg_relevance],
        max_concurrency=1,   # keep at 1 for local Ollama to avoid overloading
    )

    # Step 3: print summary (also writes to eval_results_<experiment>.txt)
    print_summary(result, experiment_name=args.experiment)

    # Step 4: flush Langfuse traces
    langfuse.flush()
    logger.info("All traces flushed to Langfuse. Experiment complete.")


if __name__ == "__main__":
    main()
