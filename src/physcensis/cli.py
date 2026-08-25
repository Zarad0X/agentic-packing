"""Command-line interface for generation and environment diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from physcensis.agent import OpenAIPredicateAgent, TemplatePredicateAgent
from physcensis.asset_library import (
    AssetManifestError,
    LicensedAssetManifest,
    ManifestAssetCatalog,
)
from physcensis.config import ConfigError, ReproductionConfig
from physcensis.evaluation import run_core_gate, run_dense_gate, run_organized_gate
from physcensis.physics import GenesisBackend, QuasiStaticBackend
from physcensis.pipeline import ScenePipeline


def _backend(name: str, config: ReproductionConfig):
    if name == "genesis":
        return GenesisBackend(headless=True)
    if name == "quasistatic":
        return QuasiStaticBackend(config.physical.displacement_threshold_m)
    raise ValueError(f"Unknown backend: {name}")


def _config(args: argparse.Namespace) -> ReproductionConfig:
    config = ReproductionConfig.from_yaml(args.config)
    samples = getattr(args, "stability_samples", None)
    if samples is not None:
        if samples < 0:
            raise ValueError("--stability-samples cannot be negative")
        config = replace(config, stability=replace(config.stability, sample_count=samples))
    return config


def _catalog(args: argparse.Namespace):
    manifest = getattr(args, "asset_manifest", None)
    if not manifest:
        return None
    cache = args.asset_cache or f"assets/cache/{Path(manifest).stem}/original"
    return ManifestAssetCatalog.load(manifest, cache)


def _generate(args: argparse.Namespace) -> int:
    config = _config(args)
    backend = _backend(args.backend, config)
    with Path(args.program).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    pipeline = ScenePipeline(config, backend, catalog=_catalog(args))
    result = pipeline.run_payload(payload, output_dir=args.output)
    summary = {
        "success": result.success,
        "feedback": result.feedback.summary,
        "measurements": dict(result.feedback.measurements),
        "issues": [issue.message for issue in result.feedback.issues],
        "output": str(args.output) if result.artifacts else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.success else 2


def _arrange(args: argparse.Namespace) -> int:
    config = _config(args)
    backend = _backend(args.backend, config)
    inventory_path = Path(args.inventory)
    with inventory_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    pipeline = ScenePipeline(config, backend, catalog=_catalog(args))
    result = pipeline.run_inventory(
        payload,
        base_dir=inventory_path.parent,
        output_dir=args.output,
    )
    supplied_ids = result.scene.metadata.get("inventory_input_object_ids", [])
    summary = {
        "success": result.success,
        "inventory_object_count": len(supplied_ids),
        "arranged_object_ids": result.scene.metadata.get("physical_placement_order", []),
        "feedback": result.feedback.summary,
        "measurements": dict(result.feedback.measurements),
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "details": dict(issue.details),
            }
            for issue in result.feedback.issues
        ],
        "output": str(args.output) if result.artifacts else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.success else 2


def _prompt(args: argparse.Namespace) -> int:
    config = _config(args)
    backend = _backend(args.backend, config)
    agent = (
        OpenAIPredicateAgent(model=args.model)
        if args.agent == "openai"
        else TemplatePredicateAgent(args.examples)
    )
    result = ScenePipeline(config, backend, catalog=_catalog(args)).generate(
        args.prompt,
        agent,
        output_dir=args.output,
    )
    summary = {
        "success": result.success,
        "rounds": result.rounds,
        "feedback": result.feedback.summary,
        "measurements": dict(result.feedback.measurements),
        "issues": [issue.message for issue in result.feedback.issues],
        "output": str(args.output) if result.artifacts else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.success else 2


def _demo(args: argparse.Namespace) -> int:
    from physcensis.demo import serve_demo

    config = _config(args)
    serve_demo(
        host=args.host,
        port=args.port,
        config=config,
        backend_name=args.backend,
        output_root=Path(args.output),
        examples_dir=Path(args.examples),
        use_openai=args.agent == "openai",
        model=args.model,
        catalog=_catalog(args),
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    config = _config(args)
    backend = _backend(args.backend, config)
    gate_runner = {
        "core": run_core_gate,
        "dense": run_dense_gate,
        "organized": run_organized_gate,
    }[args.suite]
    report = gate_runner(
        config,
        backend,
        repetitions_per_family=args.repetitions,
        examples_dir=args.examples,
        catalog=_catalog(args),
    )
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


def _doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    report = {
        "config": "ok",
        "backend": args.backend,
        "backend_import": "pending",
    }
    try:
        backend = _backend(args.backend, config)
        report["backend_import"] = "ok"
        report["backend_name"] = backend.name
    except (ImportError, RuntimeError, ValueError) as exc:
        report["backend_import"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["backend_import"] == "ok" else 2


def _assets(args: argparse.Namespace) -> int:
    manifest = LicensedAssetManifest.load(args.manifest)
    hashes = (
        manifest.fetch(args.cache)
        if args.action == "fetch"
        else manifest.validate_files(args.cache)
    )
    report = {
        "manifest": manifest.name,
        "action": args.action,
        "asset_count": len(manifest.entries),
        "cache": str(Path(args.cache).resolve()),
        "licenses": sorted({entry.license for entry in manifest.entries}),
        "minimum_quality_score": manifest.minimum_quality_score,
        "require_opaque": manifest.require_opaque,
        "required_visual_qa_status": manifest.required_visual_qa_status,
        "required_dense_scene_fit_status": manifest.required_dense_scene_fit_status,
        "verified_sha256": hashes,
    }
    print(json.dumps(report, indent=2))
    return 0


def _add_asset_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-manifest")
    parser.add_argument(
        "--asset-cache",
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physcensis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="run one predicate program")
    generate.add_argument("--config", default="configs/paper.yaml")
    generate.add_argument("--program", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--backend", choices=("quasistatic", "genesis"), default="quasistatic")
    generate.add_argument("--stability-samples", type=int)
    _add_asset_catalog_args(generate)
    generate.set_defaults(handler=_generate)

    arrange = subparsers.add_parser(
        "arrange",
        help="arrange one explicit fixed-object inventory",
    )
    arrange.add_argument("--config", default="configs/paper.yaml")
    arrange.add_argument("--inventory", required=True)
    arrange.add_argument("--output", required=True)
    arrange.add_argument(
        "--backend",
        choices=("quasistatic", "genesis"),
        default="quasistatic",
    )
    arrange.add_argument("--stability-samples", type=int)
    _add_asset_catalog_args(arrange)
    arrange.set_defaults(handler=_arrange)

    prompt = subparsers.add_parser("prompt", help="generate a scene from natural language")
    prompt.add_argument("--config", default="configs/paper.yaml")
    prompt.add_argument("--prompt", required=True)
    prompt.add_argument("--output", required=True)
    prompt.add_argument("--backend", choices=("quasistatic", "genesis"), default="quasistatic")
    prompt.add_argument("--agent", choices=("template", "openai"), default="template")
    prompt.add_argument("--model", default="o4-mini")
    prompt.add_argument("--examples", default="examples")
    prompt.add_argument("--stability-samples", type=int)
    _add_asset_catalog_args(prompt)
    prompt.set_defaults(handler=_prompt)

    demo = subparsers.add_parser("demo", help="serve the interactive reproduction demo")
    demo.add_argument("--config", default="configs/paper.yaml")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8787)
    demo.add_argument("--output", default="output/demo")
    demo.add_argument("--backend", choices=("quasistatic", "genesis"), default="quasistatic")
    demo.add_argument("--agent", choices=("template", "openai"), default="template")
    demo.add_argument("--model", default="o4-mini")
    demo.add_argument("--examples", default="examples")
    demo.add_argument("--stability-samples", type=int, default=0)
    _add_asset_catalog_args(demo)
    demo.set_defaults(handler=_demo)

    benchmark = subparsers.add_parser("benchmark", help="run the frozen core acceptance gate")
    benchmark.add_argument("--config", default="configs/paper.yaml")
    benchmark.add_argument("--backend", choices=("quasistatic", "genesis"), default="quasistatic")
    benchmark.add_argument(
        "--suite", choices=("core", "dense", "organized"), default="core"
    )
    benchmark.add_argument("--repetitions", type=int, default=20)
    benchmark.add_argument("--examples", default="examples")
    benchmark.add_argument("--report")
    benchmark.add_argument("--stability-samples", type=int)
    _add_asset_catalog_args(benchmark)
    benchmark.set_defaults(handler=_benchmark)

    doctor = subparsers.add_parser("doctor", help="validate runtime dependencies")
    doctor.add_argument("--config", default="configs/paper.yaml")
    doctor.add_argument("--backend", choices=("quasistatic", "genesis"), default="quasistatic")
    doctor.set_defaults(handler=_doctor)

    assets = subparsers.add_parser(
        "assets", help="fetch or verify a frozen licensed-asset manifest"
    )
    assets.add_argument("--action", choices=("validate", "fetch"), default="validate")
    assets.add_argument("--manifest", default="assets/manifests/objaverse_cc_by_v3.json")
    assets.add_argument("--cache", default="assets/cache/objaverse_cc_by_v3/original")
    assets.set_defaults(handler=_assets)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except (
        AssetManifestError,
        ConfigError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
