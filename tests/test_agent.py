from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import TestCase

from physcensis.agent import OpenAIPredicateAgent, TemplatePredicateAgent
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
