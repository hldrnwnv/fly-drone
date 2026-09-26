---
name: neuro-wiki-curator
description: Update fly_drone's Obsidian-style neuron and connectome wiki from completed experiments with source-checked KAG claims. Use after new neural results or when asked to curate the wiki; not for running experiments.
---

# Neuro-wiki curator

Use this workflow only in the `fly_drone` repository. Read the root `AGENTS.md`, `docs/neuro-wiki/AGENTS.md`, `README.md`, `Карта нейронов.md`, and `claims.jsonl` before editing.

## Primary research to check

Open the relevant original work, not just this list, when comparing an experimental result with known biology. Use [the project's source index](../../../docs/neuro-wiki/Источники.md) for scope notes and additional papers. These are distinct datasets; never transfer a body ID or measured edge between them without an explicit mapping.

- [MaleCNS v1.0 official data and neuPrint access](https://male-cns.janelia.org/download/) and [the MaleCNS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/) — the source dataset used by this project; check its release, annotation and raw connectivity before recording an anatomical claim.
- [MANC ventral nerve cord connectome](https://elifesciences.org/reviewed-preprints/97769) and [descending-to-motor circuit analysis](https://elifesciences.org/articles/96084) — relevant to wing and other motor pathways, but a separate reconstruction from MaleCNS.
- [FlyWire whole adult brain connectome](https://www.nature.com/articles/s41586-024-07558-y) and [its cell-type annotations](https://www.nature.com/articles/s41586-024-07686-5) — comparative context from a female brain, not a source of MaleCNS body IDs.
- [Hemibrain reconstruction](https://elifesciences.org/articles/57443) — comparative partial central-brain connectome; respect its volume boundary.
- For functional comparisons, use [DNg02 flight experiments](https://pubmed.ncbi.nlm.nih.gov/35090590/), [wing proprioception](https://elifesciences.org/articles/107867), [DNa02 walking experiments](https://elifesciences.org/articles/102230), and [T4/T5 motion experiments](https://www.nature.com/articles/nature12320) only when the tested cells and behavior match the finding.

For any new published or anatomical claim, cite the specific paper or data release in the wiki claim's `source`, give a precise figure, table, query or field in `source_locator`, and state the dataset and experimental conditions. A bibliography entry alone is not evidence for a new claim.

Compare completed `runs/*/results.json` and their protocol reports with existing KAG claims. Select specific neuron or circuit findings that have source-backed observations, including negative results. Verify the exact values, directions, controls, model version, and held-out conditions in the receipt. Keep fly experiments, MaleCNS annotations, `flybrain` model effects, and engineering choices separate.

For each supported finding, add or update a Russian neuron note, one atomic `claims.jsonl` record with a precise `source_locator`, and meaningful `[[wiki links]]` in the note and map. A note link is navigation, not a synapse. Claim an anatomical edge only with compatible body IDs, direction, release, raw synapse count, stated threshold or filter, and a reproducible primary-data query. Leave unresolved interpretations as questions on the note. Do not start a new experiment during curation.

Before finishing, run `python3 scripts/check_neuro_wiki.py` from the repository root (CUE CLI required). It validates `docs/neuro-wiki/claims.jsonl` against `claims.cue`, unique IDs, local source files, claim references, and wiki links. Fix validation failures instead of skipping the check. Independently recheck quoted metrics and anatomical claims against their primary sources, then review `git diff --check`. Report which new claims were added, their evidence type, and what remains unverified.
