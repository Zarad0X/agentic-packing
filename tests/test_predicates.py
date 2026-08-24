from __future__ import annotations

import unittest

from physcensis.predicates import PredicateParser, ProgramValidationError
from physcensis.types import PredicateKind


class PredicateParserTest(unittest.TestCase):
    def test_valid_base_program(self) -> None:
        payload = [
            ["book_0", "a book"],
            ["book_0", "PLACE-ON-BASE", "root", {"x": 0.0, "y": 0.0}],
            ["book_0", "FACING-FRONT", "root", {}],
        ]
        program = PredicateParser().parse(payload)
        self.assertEqual(program.predicates[0].kind, PredicateKind.PLACE_ON_BASE)

    def test_unknown_reference_is_rejected(self) -> None:
        payload = [
            ["book_0", "a book"],
            ["book_0", "LEFT-OF", "book_1", {"distance": 0.1}],
            ["book_0", "PLACE-ON-BASE", "root", {}],
            ["book_0", "FACING-FRONT", "root", {}],
        ]
        with self.assertRaises(ProgramValidationError) as context:
            PredicateParser().parse(payload)
        self.assertIn("unknown_reference", {issue.code for issue in context.exception.issues})

    def test_place_anywhere_must_be_alone(self) -> None:
        payload = [
            ["phone_0", "a phone"],
            ["phone_0", "PLACE-ANYWHERE", "root", {}],
            ["phone_0", "FACING-FRONT", "root", {}],
        ]
        with self.assertRaises(ProgramValidationError) as context:
            PredicateParser().parse(payload)
        codes = {issue.code for issue in context.exception.issues}
        self.assertIn("special_predicate_not_alone", codes)


if __name__ == "__main__":
    unittest.main()
