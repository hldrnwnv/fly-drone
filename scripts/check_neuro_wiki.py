"""Validate the neuron wiki's CUE claims, sources, and Obsidian links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "docs" / "neuro-wiki"
CLAIMS = WIKI / "claims.jsonl"
SCHEMA = WIKI / "claims.cue"
WIKI_LINK = re.compile(r"\[\[([^\]\n]+)\]\]")
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
CLAIM_ID = re.compile(r"\bN-\d{3,}\b")
NON_NOTE_PAGES = {"AGENTS.md", "TEMPLATE.md", "README.md", "Карта нейронов.md", "Источники.md"}


def prose_only(markdown: str) -> str:
    """Ignore examples inside fenced or inline code when checking links."""
    without_fences = re.sub(r"(?ms)^```.*?^```[^\n]*\n?", "", markdown)
    return re.sub(r"`[^`\n]*`", "", without_fences)


def local_file(path: str, base: Path) -> Path:
    candidate = (base / unquote(path.split("#", 1)[0])).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository: {path}")
    if not candidate.is_file():
        raise ValueError(f"missing local file: {path}")
    return candidate


def check_claims() -> set[str]:
    ids: set[str] = set()
    for line_number, line in enumerate(CLAIMS.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            claim = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"claims.jsonl:{line_number}: {exc}") from exc
        claim_id = claim["id"]
        if claim_id in ids:
            raise ValueError(f"claims.jsonl:{line_number}: duplicate id {claim_id}")
        ids.add(claim_id)
        local_paths: dict[str, Path] = {}
        for field in ("source", "report"):
            value = claim.get(field)
            if value and not value.startswith(("https://", "http://")):
                try:
                    local_paths[field] = local_file(value, ROOT)
                except ValueError as exc:
                    raise ValueError(f"claims.jsonl:{line_number}: {field}: {exc}") from exc
        if claim["evidence_type"] == "model_intervention":
            source_run = local_paths["source"].relative_to(ROOT / "runs").parts[0]
            report_run = local_paths["report"].relative_to(ROOT / "runs").parts[0]
            if source_run != report_run:
                raise ValueError(f"claims.jsonl:{line_number}: model source and report belong to different runs")
    if not ids:
        raise ValueError("claims.jsonl has no claims")
    return ids


def check_notes(ids: set[str]) -> int:
    cited: set[str] = set()
    pages = sorted(WIKI.glob("*.md"))
    for page in pages:
        original = page.read_text(encoding="utf-8")
        text = prose_only(original)
        for raw in WIKI_LINK.findall(text):
            target = raw.split("|", 1)[0].split("#", 1)[0]
            if not target.endswith(".md"):
                target += ".md"
            destination = local_file(target, WIKI)
            if not destination.is_relative_to(WIKI):
                raise ValueError(f"{page.name}: wiki link escapes wiki: {raw}")
        for raw in MARKDOWN_LINK.findall(text):
            target = raw.strip().strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            local_file(target, page.parent)
        references = set(CLAIM_ID.findall(original))
        unknown = references - ids
        if unknown:
            raise ValueError(f"{page.name}: unknown claim ids: {', '.join(sorted(unknown))}")
        if page.name not in NON_NOTE_PAGES:
            cited.update(references)
    orphaned = ids - cited
    if orphaned:
        raise ValueError(f"claims without a neuron note: {', '.join(sorted(orphaned))}")
    return len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue", default="cue", help="path to the CUE CLI")
    args = parser.parse_args()
    cue_binary = shutil.which(args.cue)
    if cue_binary is None and args.cue == "cue":
        project_binary = ROOT / ".venv" / "bin" / "cue"
        if project_binary.is_file():
            cue_binary = str(project_binary)
    if cue_binary is None:
        parser.error("CUE CLI is required; install it from https://cuelang.org/docs/introduction/installation/")
    command = [cue_binary, "vet", "-d", "#Claim", str(SCHEMA), str(CLAIMS)]
    subprocess.run(command, cwd=ROOT, check=True)
    try:
        ids = check_claims()
        pages = check_notes(ids)
    except ValueError as exc:
        parser.exit(1, f"neuro-wiki: {exc}\n")
    print(f"neuro-wiki: {len(ids)} claims and {pages} Markdown pages valid")


if __name__ == "__main__":
    main()
