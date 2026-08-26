# External asset attribution

The binary meshes are downloaded on demand and are not stored in Git. The
current quality-gated manifest is `assets/manifests/objaverse_cc_by_v3.json`;
the earlier v1 manifest remains available for provenance and comparison. Each records the
source model URL, author, CC license, download URL, geometry metadata, source
orientation, and SHA-256 for every accepted file.

`assets/manifests/blenderkit_dense_v1.json` is a separate ledger for 19
user-authorized BlenderKit candidates. It records the exact asset ID, author,
source URL, BlenderKit license label, hash, and QA status. Royalty-free BlenderKit
binaries remain in the authorized user's private cache and must not be copied into
this repository or redistributed as a standalone model pack. CC0 entries retain
their explicit source and hash here even though attribution is not required.

All models in this manifest are distributed under Creative Commons Attribution
4.0. Any render, scene bundle, or redistributed mesh that uses them must retain
the corresponding author and source URL from the manifest. The procedural
physics proxies and PhyScensis code remain separate from these third-party
works.

The v2 pack contains Green Beans by House Doctor; Peruvian Bowl by Beth
Fischer; Momo-gata Japanese Tea Cup by Melina.Thadea; Polychrome Dish by Thomas
Flynn; Dish by bogdanzloy20280; Le Creuset Style Mug by phillips.kieran; and
Good Morning Coffee-Cup by KonstantinAg. The v3 pack also adds 17 attributed
books, notebooks, tools, a motor, keyboards, a phone, a mouse, pens, and
pencils. Exact titles, authors, source pages, licenses, and hashes are retained
per entry in the v3 manifest.
