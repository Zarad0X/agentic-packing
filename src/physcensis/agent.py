"""Predicate-program agents for offline demos and API-backed generation."""

from __future__ import annotations

import json
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
