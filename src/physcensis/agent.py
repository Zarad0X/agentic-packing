"""Predicate and fixed-inventory agents for offline and API-backed generation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Protocol

from physcensis.types import Feedback


class PredicateAgent(Protocol):
    def propose(
        self,
        prompt: str,
        *,
        previous_payload: list[Any] | None = None,
        feedback: Feedback | None = None,
    ) -> list[Any]:
        """Return a JSON-compatible predicate payload."""


class InventoryPlanningAgent(Protocol):
    """Provider-neutral agent that plans a fixed, already-resolved inventory."""

    last_call_metadata: dict[str, Any]

    def propose_inventory(
        self,
        prompt: str,
        inventory_context: dict[str, Any],
        *,
        previous_plan: dict[str, Any] | None = None,
        feedback: Feedback | None = None,
    ) -> dict[str, Any]:
        """Return one strict fixed-inventory arrangement plan."""


def _inventory_plan_schema() -> dict[str, Any]:
    group_id = {"type": "string"}
    object_ids = {
        "type": "array",
        "items": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {
            "placement_order": {
                "type": "array",
                "items": {"type": "string"},
            },
            "stack_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": group_id,
                        "bottom_to_top_object_ids": object_ids,
                    },
                    "required": ["group_id", "bottom_to_top_object_ids"],
                    "additionalProperties": False,
                },
            },
            "adjacency_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"group_id": group_id, "object_ids": object_ids},
                    "required": ["group_id", "object_ids"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["placement_order", "stack_groups", "adjacency_groups", "rationale"],
        "additionalProperties": False,
    }


def _feedback_payload(feedback: Feedback | None) -> dict[str, Any] | None:
    if feedback is None:
        return None
    return {
        "category": feedback.category,
        "summary": feedback.summary,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "object_id": issue.object_id,
                "details": dict(issue.details),
            }
            for issue in feedback.issues
        ],
        "measurements": dict(feedback.measurements),
    }


def _inventory_instructions() -> str:
    return (
        "You are the semantic planning agent in a physics-backed fixed-inventory arrangement "
        "pipeline. Plan only the supplied object IDs; placement_order must contain every supplied "
        "ID exactly once and must never include the container. Put broad, heavy, load-bearing "
        "objects early and small gap-fillers later. Use stack_groups only for two or more objects "
        "whose stackable field is true and whose asset_id or category is identical; list each stack "
        "strictly bottom-to-top. Prefer stacks for repeated plates, bowls, and explicitly stackable "
        "cups. Use adjacency_groups to keep same-use or same-category objects together without "
        "forcing unsafe support. Do not stack ordinary handled cups, tools, cans, fragile electronics, "
        "or objects marked stackable=false. The deterministic geometry and physics solver is the "
        "safety authority. If solver feedback is present, repair the plan without changing inventory "
        "identity or count. Return only the requested JSON object."
    )


class DeterministicInventoryAgent:
    """Offline reference planner used for tests and explicit API-free fallback."""

    def __init__(self) -> None:
        self.last_call_metadata: dict[str, Any] = {"provider": "deterministic"}

    def propose_inventory(
        self,
        prompt: str,
        inventory_context: dict[str, Any],
        *,
        previous_plan: dict[str, Any] | None = None,
        feedback: Feedback | None = None,
    ) -> dict[str, Any]:
        del prompt, previous_plan, feedback
        objects = list(inventory_context["objects"])
        objects.sort(
            key=lambda item: (
                -float(item["footprint_m2"]),
                -float(item["mass_kg"]),
                str(item["object_id"]),
            )
        )
        stack_groups = []
        stackable: dict[str, list[str]] = {}
        adjacency: dict[str, list[str]] = {}
        for item in objects:
            object_id = str(item["object_id"])
            category = str(item["category"])
            adjacency.setdefault(category, []).append(object_id)
            if item["stackable"]:
                stackable.setdefault(category, []).append(object_id)
        for index, members in enumerate(stackable.values()):
            if len(members) >= 2:
                stack_groups.append(
                    {
                        "group_id": f"stack_{index}",
                        "bottom_to_top_object_ids": members,
                    }
                )
        adjacency_groups = [
            {"group_id": f"near_{index}", "object_ids": members}
            for index, members in enumerate(adjacency.values())
            if len(members) >= 2
        ]
        return {
            "placement_order": [str(item["object_id"]) for item in objects],
            "stack_groups": stack_groups,
            "adjacency_groups": adjacency_groups,
            "rationale": "Heavy broad bases first, repeated stackable assets stacked, peers grouped.",
        }


class CodexInventoryAgent:
    """Use the authenticated local Codex CLI as the inventory planning agent."""

    def __init__(
        self,
        model: str | None = None,
        *,
        executable: str | None = None,
        timeout_s: float = 180.0,
        working_directory: str | Path = ".",
    ):
        self.executable = executable or shutil.which("codex")
        if not self.executable:
            raise RuntimeError("Codex CLI is not installed or not available on PATH")
        self.model = model
        self.timeout_s = timeout_s
        self.working_directory = str(Path(working_directory).resolve())
        self.last_call_metadata: dict[str, Any] = {"provider": "codex_cli"}

    def propose_inventory(
        self,
        prompt: str,
        inventory_context: dict[str, Any],
        *,
        previous_plan: dict[str, Any] | None = None,
        feedback: Feedback | None = None,
    ) -> dict[str, Any]:
        request = {
            "task": prompt,
            "inventory": inventory_context,
            "previous_plan": previous_plan,
            "solver_feedback": _feedback_payload(feedback),
        }
        with tempfile.TemporaryDirectory(prefix="physcensis-codex-") as directory:
            root = Path(directory)
            schema_path = root / "inventory_plan.schema.json"
            output_path = root / "inventory_plan.json"
            schema_path.write_text(
                json.dumps(_inventory_plan_schema(), ensure_ascii=False), encoding="utf-8"
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--cd",
                self.working_directory,
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")
            try:
                completed = subprocess.run(
                    command,
                    input=_inventory_instructions()
                    + "\n\nPlanning request:\n"
                    + json.dumps(request, ensure_ascii=False, indent=2),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex inventory planning exceeded {self.timeout_s:.0f} seconds"
                ) from exc
            self.last_call_metadata = {
                "provider": "codex_cli",
                "model": self.model or "codex_config_default",
                "returncode": completed.returncode,
            }
            if completed.returncode != 0:
                error = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"Codex inventory planning failed: {error[-2000:]}")
            if not output_path.is_file():
                raise RuntimeError("Codex inventory planning produced no final JSON response")
            plan = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise TypeError("Codex inventory plan must be a JSON object")
        return plan


class OpenAIInventoryAgent:
    """Paper-faithful o4-mini fixed-inventory planner through the Responses API."""

    def __init__(self, model: str = "o4-mini", client: Any | None = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the 'agent' extra to use OpenAIInventoryAgent") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.last_call_metadata: dict[str, Any] = {"provider": "openai", "model": model}

    def propose_inventory(
        self,
        prompt: str,
        inventory_context: dict[str, Any],
        *,
        previous_plan: dict[str, Any] | None = None,
        feedback: Feedback | None = None,
    ) -> dict[str, Any]:
        context = {
            "task": prompt,
            "inventory": inventory_context,
            "previous_plan": previous_plan,
            "solver_feedback": _feedback_payload(feedback),
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=_inventory_instructions(),
            input=json.dumps(context, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "fixed_inventory_arrangement_plan",
                    "strict": True,
                    "schema": _inventory_plan_schema(),
                }
            },
        )
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        self.last_call_metadata = {
            "provider": "openai",
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "usage": usage if isinstance(usage, dict) else None,
        }
        plan = json.loads(response.output_text)
        if not isinstance(plan, dict):
            raise TypeError("OpenAI inventory plan must be a JSON object")
        return plan


class ProgramFileAgent:
    def __init__(self, program_path: str | Path):
        self.program_path = Path(program_path)

    def propose(
        self,
        prompt: str,
        *,
        previous_payload: list[Any] | None = None,
        feedback: Feedback | None = None,
    ) -> list[Any]:
        del prompt, previous_payload, feedback
        with self.program_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a list in {self.program_path}")
        return payload


class TemplatePredicateAgent:
    """Deterministic prompt router used by the stable, API-free demo."""

    def __init__(self, examples_dir: str | Path = "examples"):
        self.examples_dir = Path(examples_dir)

    def propose(
        self,
        prompt: str,
        *,
        previous_payload: list[Any] | None = None,
        feedback: Feedback | None = None,
    ) -> list[Any]:
        del previous_payload, feedback
        normalized = prompt.lower()
        routes = (
            (
                ("dishwashing station", "dish tub", "洗碗站", "餐具清洗"),
                "dense_dishwashing_station.json",
            ),
            (
                ("tool crate", "repair crate", "工具筐", "维修箱"),
                "dense_tool_crate.json",
            ),
            (
                ("office tote", "office storage", "办公收纳箱", "文件收纳"),
                "dense_office_tote.json",
            ),
            (("sink", "dishes", "dishware", "水槽", "洗碗", "碗碟"), "dense_kitchen_sink.json"),
            (
                ("grocery", "groceries", "dense basket", "购物篮", "收纳篮", "密集篮筐"),
                "dense_grocery_basket.json",
            ),
            (("workbench", "tool", "workshop", "工具", "工作台"), "workbench.json"),
            (("desk", "office", "电脑", "办公", "书桌"), "office_desk.json"),
            (("stack", "basket", "physical", "收纳", "堆叠", "物理"), "physical_showcase.json"),
            (("coffee", "living", "茶几", "客厅"), "coffee_table.json"),
        )
        filename = "dining_table.json"
        for keywords, candidate in routes:
            if any(keyword in normalized for keyword in keywords):
                filename = candidate
                break
        return ProgramFileAgent(self.examples_dir / filename).propose(prompt)


class OpenAIPredicateAgent:
    """Generate validated predicate programs through the Responses API.

    The API emits a normalized object/predicate representation. This class then
    translates it into the paper-style heterogeneous list consumed by the core
    parser, keeping all downstream solving independent of the model provider.
    """

    _KINDS: ClassVar[list[str]] = [
        "LEFT-OF",
        "RIGHT-OF",
        "FRONT-OF",
        "BACK-OF",
        "ALIGN-CENTER-LR",
        "ALIGN-CENTER-FB",
        "ALIGN-LEFT",
        "ALIGN-RIGHT",
        "ALIGN-FRONT",
        "ALIGN-BACK",
        "SYMMETRY-ALONG",
        "FACING-TO",
        "FACING-SAME-AS",
        "FACING-OPPOSITE-TO",
        "FACING-FRONT",
        "FACING-BACK",
        "FACING-LEFT",
        "FACING-RIGHT",
        "RANDOM-ROT",
        "ORIENT-BY-RELATIVE-SIDE",
        "PLACE-ON-BASE",
        "PLACE-ON",
        "PLACE-IN",
        "PLACE-ANYWHERE",
        "GROUP",
        "COPY-GROUP",
    ]

    def __init__(self, model: str = "o4-mini", client: Any | None = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the 'agent' extra to use OpenAIPredicateAgent") from exc
            client = OpenAI()
        self.client = client
        self.model = model

    @classmethod
    def _schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["id", "description"],
                        "additionalProperties": False,
                    },
                },
                "predicates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "kind": {"type": "string", "enum": cls._KINDS},
                            "reference": {"type": "string"},
                            "params_json": {"type": "string"},
                        },
                        "required": ["subject", "kind", "reference", "params_json"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["objects", "predicates"],
            "additionalProperties": False,
        }

    def propose(
        self,
        prompt: str,
        *,
        previous_payload: list[Any] | None = None,
        feedback: Feedback | None = None,
    ) -> list[Any]:
        context = {
            "request": prompt,
            "previous_program": previous_payload,
            "solver_feedback": None
            if feedback is None
            else {
                "category": feedback.category,
                "summary": feedback.summary,
                "issues": [issue.message for issue in feedback.issues],
                "measurements": dict(feedback.measurements),
            },
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Design a physically plausible tabletop scene. Use only common categories from: "
                "plate, cup, fork, knife, spoon, book, notebook, laptop, monitor, keyboard, "
                "mouse, phone, bottle, can, jar, bowl, vase, lamp, pen, pencil, basket, "
                "grocery basket, grocery carton, pantry box, sink, box, toolbox, "
                "drill, wrench, hammer, motor, saw, plant. Every object needs exactly one placement "
                "predicate. PLACE-ON-BASE is for direct tabletop placement. Use PLACE-ON, PLACE-IN, "
                "or PLACE-ANYWHERE only when physical reasoning is intended. Reference root for the "
                "tabletop. Add a yaw predicate for every non-special placement and keep all predicates "
                "for each subject contiguous. A special placement must be its subject's only predicate. "
                "params_json must be a valid JSON object string. On solver feedback, repair "
                "the smallest possible part of the previous program. Produce 12-24 objects."
            ),
            input=json.dumps(context, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "predicate_plan",
                    "strict": True,
                    "schema": self._schema(),
                }
            },
        )
        plan = json.loads(response.output_text)
        payload: list[Any] = []
        payload.extend([item["id"], item["description"]] for item in plan["objects"])
        for item in plan["predicates"]:
            params = json.loads(item["params_json"] or "{}")
            if not isinstance(params, dict):
                raise TypeError("params_json must decode to an object")
            payload.append([item["subject"], item["kind"], item["reference"], params])
        return payload
