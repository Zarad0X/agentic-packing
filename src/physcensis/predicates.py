"""Parser and grammar checks for the paper's ordered predicate language."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from physcensis.types import Issue, PlacementProgram, Predicate, PredicateKind


class ProgramValidationError(ValueError):
    """Raised when an agent payload violates the predicate grammar."""

    def __init__(self, issues: Iterable[Issue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


_ROTATION_KINDS = {
    PredicateKind.FACING_TO,
    PredicateKind.FACING_SAME_AS,
    PredicateKind.FACING_OPPOSITE_TO,
    PredicateKind.FACING_FRONT,
    PredicateKind.FACING_BACK,
    PredicateKind.FACING_LEFT,
    PredicateKind.FACING_RIGHT,
    PredicateKind.RANDOM_ROT,
    PredicateKind.ORIENT_BY_RELATIVE_SIDE,
}
_HEIGHT_KINDS = {
    PredicateKind.PLACE_ON_BASE,
    PredicateKind.PLACE_ON,
    PredicateKind.PLACE_IN,
    PredicateKind.PLACE_ANYWHERE,
}
_SPECIAL_KINDS = {
    PredicateKind.PLACE_IN,
    PredicateKind.PLACE_ANYWHERE,
}


def _is_description(record: Sequence[Any]) -> bool:
    return len(record) == 2 and all(isinstance(value, str) for value in record)


def _parse_batch_subject(value: Any, index: int, issues: list[Issue]) -> str | tuple[tuple[str, int], ...]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        issues.append(Issue("invalid_subject", "Predicate subject must be an object id or batch", predicate_index=index))
        return "__invalid__"
    parsed: list[tuple[str, int]] = []
    for entry in value:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], int)
            or entry[1] <= 0
        ):
            issues.append(
                Issue(
                    "invalid_batch_subject",
                    "PLACE-IN batch entries must be [category, positive integer quantity]",
                    predicate_index=index,
                )
            )
            continue
        parsed.append((entry[0], entry[1]))
    return tuple(parsed)


@dataclass(frozen=True)
class PredicateParser:
    """Parse the JSON-compatible list emitted by the paper prompt."""

    require_fully_solved: bool = True

    def parse(self, payload: Any) -> PlacementProgram:
        if not isinstance(payload, list):
            raise ProgramValidationError([Issue("not_a_list", "Program must be a plain list")])

        descriptions: dict[str, str] = {}
        predicates: list[Predicate] = []
        introduced: set[str] = set()
        issues: list[Issue] = []
        last_subject: str | None = None
        closed_subjects: set[str] = set()

        for index, record in enumerate(payload):
            if not isinstance(record, list):
                issues.append(Issue("invalid_record", "Every program record must be a list", predicate_index=index))
                continue
            if _is_description(record):
                object_id, description = record
                if object_id == "root" or object_id.startswith("group_"):
                    issues.append(
                        Issue(
                            "invalid_object_id",
                            f"Description id is reserved: {object_id}",
                            object_id=object_id,
                            predicate_index=index,
                        )
                    )
                elif object_id in descriptions:
                    issues.append(
                        Issue(
                            "duplicate_description",
                            f"Object description repeated: {object_id}",
                            object_id=object_id,
                            predicate_index=index,
                        )
                    )
                else:
                    descriptions[object_id] = description.strip()
                    introduced.add(object_id)
                continue
            if len(record) != 4:
                issues.append(
                    Issue(
                        "invalid_arity",
                        "Predicate records must contain [subject, predicate, reference, params]",
                        predicate_index=index,
                    )
                )
                continue

            subject_raw, kind_raw, reference_raw, params_raw = record
            try:
                kind = PredicateKind(kind_raw)
            except (TypeError, ValueError):
                issues.append(
                    Issue("unknown_predicate", f"Unknown predicate: {kind_raw!r}", predicate_index=index)
                )
                continue
            subject = _parse_batch_subject(subject_raw, index, issues)
            if not isinstance(params_raw, Mapping):
                issues.append(Issue("invalid_params", "Predicate params must be a mapping", predicate_index=index))
                params: Mapping[str, Any] = {}
            else:
                params = dict(params_raw)

            if kind is PredicateKind.GROUP:
                if not isinstance(subject, str) or not subject.startswith("group_"):
                    issues.append(Issue("invalid_group_id", "GROUP subject must start with group_", predicate_index=index))
                if not isinstance(reference_raw, list) or not all(isinstance(v, str) for v in reference_raw):
                    issues.append(Issue("invalid_group_members", "GROUP reference must be an object-id list", predicate_index=index))
                    reference: str | tuple[str, ...] = ()
                else:
                    reference = tuple(reference_raw)
                    missing = [value for value in reference if value not in introduced]
                    if missing:
                        issues.append(
                            Issue(
                                "unknown_group_member",
                                f"GROUP refers to objects not introduced earlier: {missing}",
                                predicate_index=index,
                            )
                        )
                anchor = params.get("anchor")
                if anchor not in reference:
                    issues.append(Issue("invalid_group_anchor", "GROUP anchor must be one of its members", predicate_index=index))
                if isinstance(subject, str):
                    introduced.add(subject)
            else:
                if not isinstance(reference_raw, str):
                    issues.append(Issue("invalid_reference", "Predicate reference must be an object id", predicate_index=index))
                    reference = "__invalid__"
                else:
                    reference = reference_raw

                if isinstance(subject, str):
                    if kind is PredicateKind.COPY_GROUP:
                        if not subject.startswith("group_"):
                            issues.append(Issue("invalid_group_id", "COPY-GROUP subject must start with group_", predicate_index=index))
                        if reference not in introduced:
                            issues.append(Issue("unknown_reference", f"Unknown source group: {reference}", predicate_index=index))
                        introduced.add(subject)
                    elif subject not in introduced:
                        issues.append(
                            Issue(
                                "missing_description",
                                f"Object must be described before use: {subject}",
                                object_id=subject,
                                predicate_index=index,
                            )
                        )
                    if subject in closed_subjects:
                        issues.append(
                            Issue(
                                "noncontiguous_subject",
                                f"Predicates for {subject} are not contiguous",
                                object_id=subject,
                                predicate_index=index,
                            )
                        )
                    if last_subject is not None and last_subject != subject:
                        closed_subjects.add(last_subject)
                    last_subject = subject

                if (
                    kind not in {PredicateKind.PLACE_IN, PredicateKind.PLACE_ANYWHERE}
                    and reference != "root"
                    and reference not in introduced
                ):
                    issues.append(
                        Issue(
                            "unknown_reference",
                            f"Reference must be root or introduced earlier: {reference}",
                            predicate_index=index,
                        )
                    )

            predicates.append(Predicate(subject, kind, reference, params, index))

        if not issues:
            issues.extend(self._semantic_checks(descriptions, predicates))
        if issues:
            raise ProgramValidationError(issues)
        return PlacementProgram(descriptions, tuple(predicates), tuple(payload))

    def _semantic_checks(
        self, descriptions: Mapping[str, str], predicates: Sequence[Predicate]
    ) -> list[Issue]:
        issues: list[Issue] = []
        by_subject: dict[str, list[Predicate]] = {key: [] for key in descriptions}
        seen_place_anywhere = False
        for predicate in predicates:
            if predicate.kind is PredicateKind.PLACE_ANYWHERE:
                seen_place_anywhere = True
            elif seen_place_anywhere and predicate.kind is not PredicateKind.GROUP:
                issues.append(
                    Issue(
                        "place_anywhere_order",
                        "PLACE-ANYWHERE predicates must appear at the end of a placement round",
                        predicate_index=predicate.source_index,
                    )
                )
            if isinstance(predicate.subject, str) and predicate.subject in by_subject:
                by_subject[predicate.subject].append(predicate)

        for object_id, object_predicates in by_subject.items():
            kinds = {predicate.kind for predicate in object_predicates}
            special = kinds & _SPECIAL_KINDS
            if special and len(object_predicates) != 1:
                issues.append(
                    Issue(
                        "special_predicate_not_alone",
                        f"Special placement for {object_id} must be its only predicate",
                        object_id=object_id,
                    )
                )
            if not self.require_fully_solved:
                continue
            if not (kinds & _HEIGHT_KINDS):
                issues.append(Issue("height_unsolved", f"No height placement for {object_id}", object_id=object_id))
            if not special and not (kinds & _ROTATION_KINDS):
                issues.append(Issue("yaw_unsolved", f"No yaw predicate for {object_id}", object_id=object_id))
            if PredicateKind.PLACE_ON in kinds and kinds & {
                PredicateKind.LEFT_OF,
                PredicateKind.RIGHT_OF,
                PredicateKind.FRONT_OF,
                PredicateKind.BACK_OF,
                PredicateKind.ALIGN_CENTER_LR,
                PredicateKind.ALIGN_CENTER_FB,
                PredicateKind.ALIGN_LEFT,
                PredicateKind.ALIGN_RIGHT,
                PredicateKind.ALIGN_FRONT,
                PredicateKind.ALIGN_BACK,
            }:
                issues.append(
                    Issue(
                        "place_on_with_planar_relation",
                        f"PLACE-ON object {object_id} cannot also use planar position predicates",
                        object_id=object_id,
                    )
                )
        return issues
