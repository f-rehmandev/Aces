"""Tests that the no-website intent is detected from the prompt/filters."""

import pytest

from src.core.task_spec import (
    Constraints, EntitySpec, Quality, Target, TaskSpec, FieldSpec,
)
from src.ui.runner import _wants_no_website


def _spec(prompt="", filters=""):
    return TaskSpec(
        natural_language_prompt=prompt,
        target=Target(),
        objective="lead_gen",
        entities=[EntitySpec(entity_name="pizza shop")],
        fields=[FieldSpec(name="business_name")],
        constraints=Constraints(filters=filters, geography="Lahore"),
        quality=Quality(min_records=50),
    )


@pytest.mark.parametrize("prompt", [
    "find 50 pizza shops in Lahore without website",
    "pizza shops in Lahore without a website",
    "pizza shops in Lahore no website",
    "pizza shops in Lahore without web presence",
    "give me pizzerias that don't have a website in Lahore",
    "list pizzerias with no websites in Lahore",
])
def test_no_website_intent_in_prompt(prompt):
    assert _wants_no_website(_spec(prompt=prompt)) is True


@pytest.mark.parametrize("filters", [
    "without website",
    "no website only",
    "without a website",
])
def test_no_website_intent_in_filters(filters):
    assert _wants_no_website(_spec(prompt="pizza in Lahore", filters=filters)) is True


def test_no_intent_when_neither_mentions_website():
    spec = _spec(prompt="find 50 pizza shops in Lahore", filters="best rated")
    assert _wants_no_website(spec) is False


def test_empty_spec_no_intent():
    assert _wants_no_website(TaskSpec()) is False


def test_case_insensitive():
    spec = _spec(prompt="PIZZA SHOPS WITHOUT WEBSITE IN LAHORE")
    assert _wants_no_website(spec) is True


def test_intent_in_filters_only():
    spec = _spec(prompt="pizza in Lahore", filters="without website")
    assert _wants_no_website(spec) is True