"""CLI entrypoint for the local-first support agent network.

Usage:
    python main.py "Can a read-only user create API credentials?"
    python main.py --mock "Can a read-only user create API credentials?"
    python main.py --file data/sample_questions.json
    python main.py --interactive
    python main.py --require-offline "Can a read-only user create API credentials?"

By default this uses REAL local Hugging Face models (downloads them the
first time, then runs offline). Pass --mock to use the deterministic mock
backend instead (useful for quickly sanity-checking graph routing without
waiting on model downloads, and for CI/automated tests).

Pass --require-offline to force transformers/huggingface_hub to refuse any
network access (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1) — this is the
flag to use when demonstrating that the app works with no internet
connection, since it fails loudly instead of silently succeeding if
anything tries to reach the network.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("support_agent.main")


def print_offline_status_banner():
    """Prints a clear, visual before-every-run check of offline status:
    the two env vars that force transformers/huggingface_hub to refuse
    network access, plus a live reachability probe against the Hugging
    Face Hub host. This exists purely so a reviewer (or a video viewer)
    doesn't have to take "it's offline" on faith — the check is right
    there in the terminal output."""
    hub_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    tf_offline = os.environ.get("TRANSFORMERS_OFFLINE", "1" if hub_offline else "0") == "1"

    try:
        socket.setdefaulttimeout(1.5)
        socket.create_connection(("huggingface.co", 443))
        network_reachable = True
    except OSError:
        network_reachable = False

    print("=" * 70)
    print("OFFLINE STATUS CHECK")
    print(f"  HF_HUB_OFFLINE=1        : {hub_offline}")
    print(f"  TRANSFORMERS_OFFLINE=1  : {tf_offline}")
    print(f"  huggingface.co reachable: {network_reachable}")
    if hub_offline and tf_offline and not network_reachable:
        print("  -> Fully offline: env vars force no network calls, AND no network is reachable.")
    elif hub_offline and tf_offline:
        print("  -> Offline mode forced via env vars (network happens to be reachable right now, "
              "but transformers/huggingface_hub will not use it).")
    else:
        print("  -> Online mode: models may be downloaded/checked against the Hub. "
              "Use --require-offline to force and prove offline operation.")
    print("=" * 70 + "\n")


def run_once(app, question: str, backend=None) -> dict:
    initial_state = {
        "question": question,
        "revision_count": 0,
        "node_trace": [],
    }
    t0 = time.time()
    result = app.invoke(initial_state, config={"recursion_limit": 25})
    latency = time.time() - t0

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("-" * 70)
    print("NODE TRACE:", " -> ".join(result["node_trace"]))
    print(f"LATENCY: {latency:.2f}s")
    print("-" * 70)
    print(json.dumps(result["final_output"], indent=2))
    print("=" * 70)
    if backend is not None:
        # Logged again here (not just at startup) because the generation
        # model hasn't loaded yet the first time load_stats() is called in
        # main() — this line reflects the real, post-load timing.
        logger.info("Model stats (post-run): %s", json.dumps(backend.load_stats()))
    return result


def main():
    parser = argparse.ArgumentParser(description="Local-first support agent network")
    parser.add_argument("question", nargs="?", help="A single question to ask")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic mock model backend")
    parser.add_argument("--file", help="Path to a JSON file of {question: ...} objects to run in batch")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--kb-dir", default="data/kb")
    parser.add_argument("--cases", default="data/resolved_cases.json")
    parser.add_argument("--k", type=int, default=4, help="Number of passages to retrieve")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Answerable-vs-clarification retrieval score threshold "
                              "(defaults: 0.08 for --mock, 0.35 for real HF embeddings)")
    parser.add_argument("--require-offline", action="store_true",
                         help="Force HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before any "
                              "model code runs, so the app errors instead of silently using the "
                              "network. Use this to demonstrate offline operation.")
    args = parser.parse_args()

    # Must happen BEFORE any transformers/huggingface_hub import (including
    # inside src.models), which is why this is the very first thing main()
    # does after parsing args.
    if args.require_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print_offline_status_banner()

    from src.graph import build_graph
    from src.knowledge_base import load_passages
    from src.models import get_backend
    from src.retriever import Retriever

    logger.info("Loading knowledge base from %s ...", args.kb_dir)
    passages = load_passages(args.kb_dir, args.cases)
    logger.info("Loaded %d passages (%d docs + resolved cases)", len(passages),
                len({p.document for p in passages if p.source_type == "kb"}))

    backend = get_backend(use_mock=args.mock)
    retriever = Retriever(backend, passages)

    t0 = time.time()
    retriever.build_index()
    logger.info("Index built in %.2fs", time.time() - t0)
    logger.info("Model stats: %s", json.dumps(backend.load_stats()))

    threshold = args.threshold
    if threshold is None:
        threshold = 0.08 if args.mock else 0.35
    app = build_graph(backend, retriever, k=args.k, answerable_threshold=threshold)

    if args.file:
        items = json.loads(open(args.file).read())
        for item in items:
            run_once(app, item["question"], backend=backend)
    elif args.interactive:
        print("Interactive mode. Ctrl+C to exit.")
        while True:
            try:
                q = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip():
                run_once(app, q.strip(), backend=backend)
    elif args.question:
        run_once(app, args.question, backend=backend)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
