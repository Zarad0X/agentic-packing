from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import TestCase

from physcensis.agent import (
    OpenAIInventoryAgent,
    OpenAIPredicateAgent,
    TemplatePredicateAgent,
)
from physcensis.predicates import PredicateParser


class AgentTest(TestCase):
    def test_template_router_covers_demo_families(self) -> None:
        agent = TemplatePredicateAgent("examples")
        parser = PredicateParser()
        prompts = (
            "dinner for four",
            "office desk",
            "workbench tools",
            "living room coffee table",
            "physical basket stack",
            "dense grocery basket",
            "kitchen sink full of dishes",
            "dense dishwashing station",
            "heavy tool crate",
            "office storage tote",
        )
        for prompt in prompts:
            program = parser.parse(agent.propose(prompt))
            self.assertGreaterEqual(len(program.descriptions), 1)

    def test_openai_agent_translates_normalized_plan(self) -> None:
        plan = {
            "objects": [{"id": "cup_0", "description": "a ceramic cup"}],
            "predicates": [
                {
                    "subject": "cup_0",
                    "kind": "PLACE-ANYWHERE",
                    "reference": "root",
                    "params_json": "{}",
                }
            ],
        }

        class Responses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(output_text=json.dumps(plan))

        responses = Responses()
        client = SimpleNamespace(responses=responses)
        payload = OpenAIPredicateAgent(client=client).propose("put a cup somewhere")
        self.assertEqual(payload[-1], ["cup_0", "PLACE-ANYWHERE", "root", {}])
        self.assertEqual(responses.kwargs["text"]["format"]["type"], "json_schema")
        PredicateParser().parse(payload)

    def test_openai_inventory_agent_uses_fixed_identity_schema(self) -> None:
        plan = {
            "placement_order": ["plate_1", "plate_2"],
            "stack_groups": [
                {
                    "group_id": "plates",
                    "bottom_to_top_object_ids": ["plate_1", "plate_2"],
                }
            ],
            "adjacency_groups": [],
            "rationale": "Keep identical plates in one ordinary stack.",
        }

        class Responses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    id="resp_inventory",
                    output_text=json.dumps(plan),
                    usage=None,
                )

        responses = Responses()
        agent = OpenAIInventoryAgent(client=SimpleNamespace(responses=responses))
        result = agent.propose_inventory(
            "organize the dishes",
            {"container": {"object_id": "sink"}, "objects": []},
        )

        self.assertEqual(result, plan)
        self.assertEqual(
            responses.kwargs["text"]["format"]["name"],
            "fixed_inventory_arrangement_plan",
        )
        self.assertEqual(agent.last_call_metadata["response_id"], "resp_inventory")
