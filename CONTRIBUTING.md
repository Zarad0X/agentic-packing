# Contributing

Contributions that improve Agentic Packing's planning loop, dense placement,
physical validation, documentation, or redistributable asset integrations are
welcome.

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
agentic-packing doctor --backend quasistatic
python -m ruff check src tests
python -m pytest -q
```

Keep deterministic geometry results distinct from real Genesis physics
evidence. New physical claims should include the simulator version, command,
device, scene artifact, and settle measurement.

Do not commit downloaded BlenderKit/Objaverse model bytes, credentials, caches,
or generated outputs. Redistributable manifests must retain source identifiers,
license metadata, hashes, and visual-QA status.

Please keep changes focused and add or update tests for behavior changes.
