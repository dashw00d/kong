"""Persistent analysis state — save/load between runs.

Stores FunctionResult data as JSON so Kong can resume interrupted runs
or improve on previous results without re-analyzing everything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kong.agent.models import FunctionResult

logger = logging.getLogger(__name__)

STATE_FILENAME = "analysis_state.json"


def save_state(
    results: dict[int, FunctionResult],
    output_dir: Path,
) -> Path:
    """Save analysis results to a state file for resumption."""
    entries = []
    for addr, r in sorted(results.items()):
        entries.append({
            "address": addr,
            "original_name": r.original_name,
            "name": r.name,
            "signature": r.signature,
            "confidence": r.confidence,
            "classification": r.classification,
            "comments": r.comments,
            "reasoning": r.reasoning,
            "error": r.error,
            "llm_calls": r.llm_calls,
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
            "signature_applied": r.signature_applied,
            "variables": [{"old": old, "new": new} for old, new in r.variables],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STATE_FILENAME
    path.write_text(json.dumps({"functions": entries}, indent=2))
    logger.info("Saved analysis state: %d results to %s", len(entries), path)
    return path


def load_state(output_dir: Path) -> dict[int, FunctionResult]:
    """Load previous analysis results. Returns empty dict if none found."""
    path = output_dir / STATE_FILENAME
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load state file %s: %s", path, e)
        return {}

    results: dict[int, FunctionResult] = {}
    for entry in data.get("functions", []):
        addr = entry.get("address", 0)
        if not addr:
            continue
        variables = [
            (v["old"], v["new"])
            for v in entry.get("variables", [])
            if "old" in v and "new" in v
        ]
        results[addr] = FunctionResult(
            address=addr,
            original_name=entry.get("original_name", ""),
            name=entry.get("name", ""),
            signature=entry.get("signature", ""),
            confidence=entry.get("confidence", 0),
            classification=entry.get("classification", ""),
            comments=entry.get("comments", ""),
            reasoning=entry.get("reasoning", ""),
            error=entry.get("error", ""),
            llm_calls=entry.get("llm_calls", 0),
            skipped=entry.get("skipped", False),
            skip_reason=entry.get("skip_reason", ""),
            signature_applied=entry.get("signature_applied", False),
            variables=variables,
        )

    logger.info("Loaded previous state: %d results from %s", len(results), path)
    return results
