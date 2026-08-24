# Reproduction specification

## Goal

Reproduce the paper's core idea, code path, and visible effect at a level that
resembles the project website and is stable enough for repeated demonstrations.
Robot-learning experiments and exact recovery of the authors' asset library are
not required for completion.

## Authoritative method contract

The reconstruction follows the ICLR 2026 paper and its appendix.

1. The input is a natural-language scene description plus a base-surface
   bounding box.
2. The agent returns asset descriptions and an ordered predicate program.
3. Spatial predicates solve planar position and yaw. The paper uses 2D convex
   hull overlap and boundary distance penalties, with 10 coordinate-descent
   iterations and 40 candidate values per parameter.
4. Physical predicates solve containment, stacking, support, and free
   placement. The paper uses occupancy grids, `scipy.correlate`, support-contact
   convex hulls, projected center-of-mass checks, and Genesis simulation.
5. Stability is evaluated in an 11-dimensional perturbation space: 3D position,
   3D small-angle rotation, 3D center-of-mass shift, friction, and mass.
6. Feedback distinguishes grammar errors, unsolved objects, solver failures,
   empty regions, and successful-scene measurements.

## Predicate coverage

Spatial:

- `LEFT-OF`, `RIGHT-OF`, `FRONT-OF`, `BACK-OF`
- `ALIGN-CENTER-LR`, `ALIGN-CENTER-FB`, `ALIGN-LEFT`, `ALIGN-RIGHT`,
  `ALIGN-FRONT`, `ALIGN-BACK`
- `SYMMETRY-ALONG`
- `FACING-TO`, `FACING-SAME-AS`, `FACING-OPPOSITE-TO`, cardinal facing,
  `RANDOM-ROT`, `ORIENT-BY-RELATIVE-SIDE`
- `PLACE-ON-BASE`, `GROUP`, `COPY-GROUP`

Physical:

- `PLACE-IN`
- `PLACE-ON`
- `PLACE-ANYWHERE`

## Known ambiguities

The paper does not report several thresholds and simulator settings; these are
enumerated under `paper_unspecified` in `configs/paper.yaml`. Reproduction
choices are stored separately and must be tuned through versioned evaluations.

The main method and asset appendix describe an approximately 800-model
BlenderKit library, but the LayoutVLM appendix calls the same database
Objaverse-sourced. This implementation keeps source identity explicit per asset
and evaluates a separate 24-model Objaverse CC BY visual pack without silently
claiming it is the paper's private database. BlenderKit remains an adapter
boundary until an authorized API credential and Blender export runtime are
provided.

## Completion evidence

Completion requires all of the following, not merely a passing smoke test:

- one-command installation and generation on a documented NVIDIA Linux host;
- all predicate families covered by parser and solver tests;
- five representative website-like scene families rendered successfully;
- 100-run core gate meeting the thresholds in `configs/paper.yaml`;
- real Genesis simulation evidence, including 400-step settle-distance metrics;
- a stable interactive demo that accepts a prompt, shows progress/feedback, and
  exports scene state plus images;
- visual inspection showing dense, non-penetrating, naturally oriented scenes;
- frozen commands, seeds, manifests, logs, metrics, and representative outputs.

## Verified reproduction snapshot (2026-08-25)

The stable-demo completion gate has been met on the checked-in implementation:

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Unit/integration tests | 23 passed | Parser, geometry, agent routing, core and complex dense families, pipeline, evaluation, and licensed-asset quality/integrity |
| Ruff | passed | Source and tests |
| Core gate | 100/100, five families | `geometry_only`; 20 runs per family, at least 15 objects each |
| Dining scene | 16 objects, 0 spatial penalty, 0.032 mm settle | Real Genesis, 400 steps |
| Physical showcase | 15 objects, 12 physical placements, 0 spatial penalty, 0.074 mm settle | Real Genesis, 400 steps; containment and stacking exercised |
| Dense-container gate | 20/20, two families | `geometry_only`; 23--24 total objects, at least three support layers, 31.3% minimum packing fraction |
| Grocery basket | 23 total / 22 packed, 3 layers, 47.9% fill, 0.224 mm settle | Real Genesis, 400 steps |
| Kitchen sink | 24 total / 23 packed, 5 post-simulation layers, 31.3% fill, 4.186 mm settle | Real Genesis, 400 steps |
| Dishwashing station | 31 total / 30 packed, 10 layers, 46.2% fill, 4 stacks / 18 nested, 0.472 mm settle | Real Genesis, 400 steps; licensed opaque dishware overlays |
| Tool crate | 24 total / 23 packed, 8 layers, 54.0% fill, 2.584 mm settle | Real Genesis, 400 steps; mixed long, heavy, and cylindrical objects |
| Office tote | 28 total / 27 packed, 7 layers, 88.4% surface coverage, 8.953 mm settle | Real Genesis, 400 steps; stacked thin objects plus small fillers |
| Interactive demo | prompt-to-preview/export verified | Progress states, measurements, predicates, and downloadable scene JSON |
| Licensed asset pack v3 | 24/24 files validated | CC BY 4.0, SHA-256, Objaverse++ score 3, opaque and embedded-texture only, proxy-fit audit plus dense Genesis QA |

The CUDA verification ran with Genesis 1.3.3 and PyTorch 2.11.0+cu128 on an
NVIDIA RTX A6000. The test config used for browser interaction is intentionally
separate from the paper config.

To avoid expensive Genesis kernel reconstruction for every candidate, physical
placements are screened with occupancy/support geometry and the assembled scene
is then subjected to one authoritative 400-step Genesis validation. This keeps
the interactive demo responsive while retaining real final-state physics
evidence. The perturbation estimator remains available independently.

Dense `PLACE-IN` placement searches the container-local frame at both 0 and 90
degree yaw, treats already placed top faces as candidate support layers, checks
support area and collision before accepting each pose, and validates the final
assembled scene in Genesis. Procedural container rims, basket slats, dish rims,
handles, lids, and package labels improve category readability without claiming
the photoreal texture fidelity of the private asset collection.

The optional `objaverse_cc_by_v3` manifest overlays licensed GLB visuals on the
same procedural proxy records. Mesh centering, up-axis conversion, and scale are
derived from frozen raw bounds and target proxy dimensions. Genesis collision,
containment, support, and settle measurements continue to use only the proxies;
this intentionally isolates visible-asset changes from physical acceptance.
Uniform-fit assets preserve aspect ratio and must occupy at least 45% of every
proxy axis and 20% of proxy volume. Repeated-instance face/file budgets prevent
one attractive model from making a dense scene impractical. The v3 dense sink
uses two plate variants, one bowl, two mugs, and one stackable-cup variant; its
24-object Genesis result retains the same five layers, 31.3% fill, and 4.186 mm
settle distance while removing the v1 transparent/empty-looking dishware.

## Non-goals

- Exact robot-policy success rates.
- Redistribution of restricted third-party assets.
- Claiming paper-number reproduction using a different asset/evaluation set.
- Claiming procedural proxy renders reproduce the private asset library's
  photorealism.
