# PhyScensis clean-room reproduction

[![CI](https://github.com/Zarad0X/physcensis-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/Zarad0X/physcensis-reproduction/actions/workflows/ci.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

This repository reconstructs the core method from **PhyScensis: Physics-Augmented
LLM Agents for Complex Physical Scene Arrangement** (Wang et al., ICLR 2026).
The target is a stable, high-quality scene-generation demo rather than an exact
copy of the authors' private asset library or robot experiments.

The implementation follows the paper's information flow:

1. an agent emits object descriptions and spatial/physical predicates;
2. a parser validates the predicate program;
3. a convex-polygon spatial solver places objects on the base surface;
4. an occupancy/physics solver realizes `PLACE-IN`, `PLACE-ON`, and
   `PLACE-ANYWHERE`;
5. structured feedback drives another generation round;
6. renderers and evaluation tools expose semantic and physical quality.

## Status

The stable demonstration target is implemented. It includes the complete
predicate language described in the paper, planar optimization, occupancy-based
physical placement, Genesis validation, feedback-driven prompt generation,
procedural rendering, a five-family benchmark, a dedicated dense-container
gate, a browser demo, and a frozen seven-model Objaverse CC BY visual-asset pack.
The dense examples contain up to 31 total objects and use multi-layer support
search, explicit same-asset nesting, and distinct open-container geometries
rather than a single planar packing pass.

The frozen method contract and evidence are recorded in
[`docs/reproduction_spec.md`](docs/reproduction_spec.md); module boundaries and
control flow are in [`docs/architecture.md`](docs/architecture.md).

This is an independent clean-room reproduction. It is not the official code
release of the PhyScensis authors and does not redistribute their private asset
library.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
physcensis generate --program examples/dining_table.json --output output/scenes/dining_table
python -m pytest -q
```

Launch the interactive local demo:

```bash
physcensis demo --backend quasistatic
# Open http://127.0.0.1:8787
```

The deterministic backend is intended for parser, geometry, prompt, and demo
regression. It labels its results `GEOMETRY VALID`; it does not claim physical
simulation. Paper-like physical validation requires Genesis and a CUDA-capable
Linux host:

```bash
python -m pip install -e '.[dev,genesis,render]'
physcensis doctor --backend genesis
CUDA_VISIBLE_DEVICES=0 physcensis prompt \
  --prompt "Create a warm dining table for four people" \
  --output output/scenes/dining_genesis \
  --backend genesis --stability-samples 0
```

`--stability-samples 0` keeps the interactive path fast while still running one
400-step final-scene Genesis validation. Set it to `64` to run the paper-sized
11D perturbation estimate as well.

## Natural-language agent

The repository works offline with a deterministic prompt-family agent. An
optional OpenAI Responses API adapter emits the same strict predicate schema and
participates in the same structured feedback loop:

```bash
python -m pip install -e '.[agent]'
export OPENAI_API_KEY=...  # never store the key in this repository
physcensis demo --backend genesis --agent openai --model o4-mini
```

## Reproduce the acceptance gate

```bash
physcensis benchmark \
  --backend quasistatic \
  --repetitions 20 \
  --report output/evaluation/core_quasistatic_100.json
```

The frozen report contains 100/100 successful generations over dining table,
office desk, workbench, coffee table, and physical-showcase families; every run
contains at least 15 objects. This is explicitly stored as `geometry_only`.
Separate real-physics outputs record a 400-step Genesis settle displacement of
0.032 mm for the 16-object dining scene and 0.074 mm for the 15-object physical
showcase.

Reproduce the separate dense-container gate:

```bash
physcensis benchmark \
  --suite dense \
  --backend quasistatic \
  --repetitions 10 \
  --report output/evaluation/dense_quasistatic_20.json
```

The frozen dense report contains 20/20 successful generations. The grocery
basket contains 22 packed objects across at least three support layers with a
47.9% projected packing fraction; the kitchen sink contains 23 packed objects
across at least four layers with a 31.3% packing fraction. Separate 400-step
Genesis runs on an RTX A6000 settled by 0.224 mm and 4.186 mm respectively.
The semantic-stack sink variant contains two same-model plate stacks, one bowl
stack, and one handleless-cup stack (14 nested members total); its freely
simulated rigid-stack proxies settle by 0.465 mm.

Three additional complex demos exercise different packing regimes: a
31-object dishwashing station with four semantic stacks and 18 nested members,
a 24-object mixed tool crate, and a 28-object office tote with stacked books,
thin devices, and small stationery. Their authoritative Genesis runs settle by
0.472 mm, 2.584 mm, and 8.953 mm respectively, all below the frozen 10 mm gate.

Large assets are intentionally not stored in Git. Asset acquisition produces a
manifest containing source IDs, licenses, hashes, annotations, and derived mesh
paths.

## Licensed visual assets

The frozen `objaverse_cc_by_v2` pack supplies seven manually screened GLB models:
one textured grocery can, one bowl, one handleless stackable tea cup, two plate
variants, and two non-stackable mug variants. The
repository stores only the allowlisted manifest and attribution; downloaded
model bytes stay in an ignored cache. Every file must be CC BY 4.0, match its
frozen SHA-256, score 3 in Objaverse++, be marked non-transparent, and pass both
thumbnail and real Genesis visual QA before a scene can use it.

Install or validate the pack on a machine with network access:

```bash
physcensis assets --action fetch \
  --manifest assets/manifests/objaverse_cc_by_v2.json \
  --cache assets/cache/objaverse_cc_by_v2/original

physcensis assets --action validate \
  --manifest assets/manifests/objaverse_cc_by_v2.json \
  --cache assets/cache/objaverse_cc_by_v2/original
```

Render the dense kitchen sink with the licensed meshes:

```bash
CUDA_VISIBLE_DEVICES=0 physcensis generate \
  --program examples/dense_kitchen_sink.json \
  --output output/scenes/dense_kitchen_sink_semantic_stacks_v3 \
  --backend genesis --stability-samples 0 \
  --asset-manifest assets/manifests/objaverse_cc_by_v2.json \
  --asset-cache assets/cache/objaverse_cc_by_v2/original
```

Render the three complex dense demos with the same physical/visual boundary:

```bash
for scene in dense_dishwashing_station dense_tool_crate dense_office_tote; do
  CUDA_VISIBLE_DEVICES=0 physcensis generate \
    --program "examples/${scene}.json" \
    --output "output/scenes/${scene}_genesis" \
    --backend genesis --stability-samples 0 \
    --asset-manifest assets/manifests/objaverse_cc_by_v2.json \
    --asset-cache assets/cache/objaverse_cc_by_v2/original
done
```

External meshes are presentation overlays only. The deterministic procedural
proxy remains the collision and stability authority, so changing a visual model
cannot silently change the accepted packing result. A `nested` request is
accepted only for manifest entries marked `stackable`; each accepted column is
recorded in `scene.metadata.semantic_stacks` and simulated as one freely moving
rigid-stack proxy. See
[`docs/asset_library.md`](docs/asset_library.md) and
[`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md) for provenance, licensing, and
the exact integration boundary.

## Reproduction boundary

- In scope: predicate language, spatial solver, physical placement, stability
  estimation, feedback loop, asset retrieval interfaces, rendering, evaluation,
  and project-style demos.
- Deferred: the paper's robot demonstration pipeline and exact reproduction of
  its private approximately 800-asset BlenderKit library. BlenderKit remains an
  optional source adapter because its account/API credential and Blender export
  runtime are not redistributable project inputs.
- Not claimed: exact paper metrics before a frozen asset manifest, simulator
  version, API models, prompts, seeds, and evaluation set are available.

Procedural proxies keep the system self-contained; the optional frozen
Objaverse pack improves visible category detail without claiming the breadth or
photorealism of the authors' private asset collection.
