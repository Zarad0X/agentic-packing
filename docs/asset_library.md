# Licensed asset library

## Delivered pack

`assets/manifests/objaverse_cc_by_v2.json` is the current frozen visual-asset
pack for the dense grocery-basket and kitchen-sink demos. It contains seven GLB
files: one grocery can, one bowl, one handleless stackable tea cup, two plate
variants, and two non-stackable mug variants. Each
manifest entry records:

- the Objaverse UID and original Sketchfab page;
- title, author, license, and license URL;
- an immutable download URL and expected SHA-256;
- raw mesh bounds and source up-axis used to derive the visual transform.
- Objaverse++ quality score, transparency annotation, and visual-QA status.
- an explicit `stackable` gate and, where applicable, a measured visual
  `stacking_step_ratio`.

The general loader accepts only CC0 1.0, CC BY 4.0, and CC BY-SA 4.0. The v2
pack is stricter: every entry is CC BY 4.0, has Objaverse++ score 3, is marked
non-transparent, and has passed both thumbnail review and a real Genesis
contact-sheet render. Attribution remains required when renders or redistributed
model files are published; generated scene JSON retains author, source URL,
license, hash, quality score, transparency flag, and QA status per object.

## Install and validate

```bash
physcensis assets --action fetch \
  --manifest assets/manifests/objaverse_cc_by_v2.json \
  --cache assets/cache/objaverse_cc_by_v2/original
```

`fetch` downloads to a `.part` file, verifies SHA-256, and atomically installs
the file. `validate` performs the same integrity check without network access:

```bash
physcensis assets --action validate \
  --manifest assets/manifests/objaverse_cc_by_v2.json \
  --cache assets/cache/objaverse_cc_by_v2/original
```

The cache is ignored by Git. Copy it between offline rendering hosts together
with the same manifest, then run `validate` on the destination.

## Physics boundary

The external GLB is deliberately visual-only. Object dimensions, mass,
friction, containment, support, collision, and final Genesis settle distance
come from the deterministic procedural proxy. A mesh is centered and scaled to
that proxy, added as a fixed collision-disabled visual, and never becomes the
physical body. This avoids fragile collision decomposition and keeps the frozen
geometry benchmark comparable with and without external assets.

For `nested` placement, the solver first checks the manifest's `stackable`
flag, assigns every upper member to a deterministic same-asset column, and uses
the declared stacking increment for collision search. Genesis then represents
the already nested column as one freely simulated rigid-stack proxy while each
licensed mesh remains individually visible. This preserves gravity, container
collision, and settle-distance validation without inventing unstable layer-wise
sliding between interlocked dishes.

## BlenderKit boundary

BlenderKit's official client supports authenticated asset search and download,
but a reproducible export also needs a user-authorized account/API key and a
Blender runtime that can resolve the downloaded asset into a portable GLB. No
credential or BlenderKit binary asset is included here. A later BlenderKit
adapter should emit the same manifest fields and must pass the same license,
provenance, hash, and visual-only integration checks before use.

## Quality boundary

The v2 curation gate first intersects LVIS category candidates with Objaverse++
score 3, `is_transparent=false`, `is_scene=false`, and
`is_multi_object=false`. Source metadata, license, texture count, and thumbnail
are reviewed next. GLBs that survive are audited for embedded textures and
opaque material modes, then rendered together in
`examples/dishware_contact_sheet.json`. Only correct scale, orientation,
normals, material, and opacity earn `visual_qa_status=genesis_pass`.

The semantic-stack dense-sink render contains 26 objects: two three-plate
same-model stacks, one four-bowl stack, one four-member handleless tea-cup
stack, two individually placed non-stackable mugs, and nine utensils. It has
eight collision layers, 35.8% fill, and 0.465 mm Genesis settle distance. It is
a materially stronger licensed baseline, but it still does not claim the
breadth or photorealism of the paper's private approximately 800-model library.
