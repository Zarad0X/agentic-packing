# Architecture

## Implementation approach

The repository separates symbolic intent, geometric solving, physical
validation, and presentation. Every backend consumes and returns serializable
scene state so deterministic geometry tests and expensive Genesis simulations
share one contract.

```mermaid
classDiagram
    class ReproductionConfig {
      +from_yaml(path: Path) ReproductionConfig
    }
    class PredicateParser {
      +parse(payload: list) PlacementProgram
    }
    class AssetCatalog {
      +resolve(description: str) AssetRecord
    }
    class ManifestAssetCatalog {
      +load(manifest, cache) ManifestAssetCatalog
      +resolve(description: str) AssetRecord
    }
    class SpatialSolver {
      +solve(program: PlacementProgram, scene: SceneState) SolveReport
    }
    class PhysicalSolver {
      +solve(program: PlacementProgram, scene: SceneState) SolveReport
    }
    class PhysicsBackend {
      <<Protocol>>
      +simulate(scene: SceneState, steps: int) SimulationResult
    }
    class StabilityEstimator {
      +estimate(scene: SceneState, object_id: str) StabilityResult
    }
    class FeedbackEngine {
      +build(report: SolveReport) Feedback
    }
    class ScenePipeline {
      +generate(prompt: str) GenerationResult
      +run_program(program: PlacementProgram) GenerationResult
    }
    class SceneRenderer {
      +render(scene: SceneState, output_dir: Path) RenderArtifacts
    }

    ScenePipeline --> PredicateParser
    ScenePipeline --> AssetCatalog
    ManifestAssetCatalog --|> AssetCatalog
    ScenePipeline --> SpatialSolver
    ScenePipeline --> PhysicalSolver
    ScenePipeline --> FeedbackEngine
    ScenePipeline --> SceneRenderer
    PhysicalSolver --> PhysicsBackend
    PhysicalSolver --> StabilityEstimator
    StabilityEstimator --> PhysicsBackend
    SpatialSolver --> ReproductionConfig
    PhysicalSolver --> ReproductionConfig
```

The Genesis implementation deliberately validates the assembled scene once,
after deterministic occupancy and support checks select physical candidates.
Incrementally rebuilding a Genesis scene for every candidate triggers repeated
kernel compilation and is not suitable for an interactive demonstration. The
final 400-step simulation remains the authority for the reported physical
settle distance.

Licensed external geometry follows a dual-representation contract. A manifest
catalog resolves a frozen, hash-verified GLB plus its provenance and visual
transform, while the primitive catalog still resolves the object's collision
size, mass, friction, and semantic visual tag. Genesis creates the GLB as a
fixed, collision-disabled presentation entity and hides the proxy material; the
proxy body remains the only collision authority. This makes license/model swaps
auditable without changing packing feasibility or stability measurements.
Quality-gated manifests may additionally require an Objaverse++ score threshold,
an explicit non-transparent annotation, and `genesis_pass` for every entry; the
catalog refuses to load when any one of those conditions is missing.

For dense containers, `container_candidates` searches in container-local
coordinates so rotated containers and 90-degree package orientations share the
same bounds logic. Candidate heights come from the container floor and every
previously placed top surface. Each candidate must satisfy containment,
non-overlap, and a minimum support-area ratio before the physical solver can
accept it. This produces genuine multi-layer packing while keeping the
persistent scene representation compact and serializable.

The `organized` strategy adds a household-storage planner above that geometric
search. It first orders the complete mixed batch by footprint and load-bearing
value, but repeatedly scans all remaining objects so smaller pieces can refill
floor holes before any upper layer is opened. Floor candidates minimize a
compactness/contact-gap score. Only after no remaining object fits the floor may
the planner use an upper surface, where it enforces both mass capacity and a
semantic compatibility profile. Paper goods may form paper stacks and workshop
tools may share rigid workshop supports; fragile electronics cannot become
generic shelves. Small loose items remain eligible to backfill safe gaps.

Every accepted container contact is retained in
`scene.metadata.container_supports`, and each global placement order is retained
in `scene.metadata.storage_plans`. Evaluation therefore measures floor coverage,
floor compactness, bottom-layer fraction, load-bearing violations, semantic
support violations, and a composite organization score. These measurements are
independent of visual meshes and are checked before the final assembled-scene
Genesis simulation.

The same support graph also drives a separate presentation-only contact pass.
It recursively preserves floor contact and aligns each upper visual bottom to
the highest visible top among its recorded supporters. This removes apparent
floating caused by a conservative collision proxy being taller than its fitted
GLB, while leaving Genesis collision, stability, and settled poses unchanged.
The evaluator records pre-alignment gap, applied correction, final contact gap,
violations above 5 mm, and unresolved support cycles; organized acceptance
requires zero final-gap violations and zero unresolved supports.

## Program call flow

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant P as ScenePipeline
    participant A as Agent
    participant R as PredicateParser
    participant C as AssetCatalog
    participant S as SpatialSolver
    participant H as PhysicalSolver
    participant B as PhysicsBackend
    participant F as FeedbackEngine
    participant V as SceneRenderer

    U->>P: generate(prompt)
    loop up to generation_round_limit
      P->>A: propose(prompt, previous_program, feedback)
      A-->>P: predicate payload
      P->>R: parse(payload)
      alt grammar invalid
        R-->>P: structured issues
        P->>F: grammar feedback
      else grammar valid
        R-->>P: PlacementProgram
        P->>C: resolve all object descriptions
        C-->>P: AssetRecord map
        P->>S: solve base and planar constraints
        S-->>P: SolveReport
        alt spatial failure
          P->>F: solver and empty-region feedback
        else spatial success
          P->>H: solve physical predicates
          H->>B: simulate candidate placements
          B-->>H: displacement/contact results
          H-->>P: SolveReport
          alt physical failure
            P->>F: support/fall/penetration feedback
          else success
            P->>V: render and export scene
            V-->>P: artifacts
            P-->>U: GenerationResult
          end
        end
      end
    end
```

## Dependency order

1. `config`, `types`, `geometry`
2. `predicates`, `assets`, `asset_library`
3. `spatial_solver`
4. `occupancy`, `physics`, `stability`, `physical_solver`
5. `feedback`, `agent`, `pipeline`
6. `render`, `evaluation`, `cli`, demo service
