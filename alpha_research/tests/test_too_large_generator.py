"""Run rs20261126060436 (2026-11-26): the Groq models' generator prompt was too large for
their per-minute limit. A too-large request does not take a model down, so its arms kept
getting the round's lost slots back and the round looped through 9 refill passes. Now each
refusal halves the fields the model's prompts list; at the floor the model writes no alphas
for the rest of the run, so its arms get no slots and the round ends."""

import dataclasses

from alpha_research.llm import RequestTooLarge
from alpha_research.research import MIN_FIELDS_WHEN_TOO_LARGE, Budget

from test_research import RouterLLM, make_run


class TooLargeGenerator(RouterLLM):
    """The generator's request never fits; every other agent answers normally."""

    def __init__(self):
        super().__init__()
        self.refused = 0

    def complete(self, system, user, *, max_output=4096):
        if "Critic in a quant" not in system and "judge in a debate" not in system \
                and "research strategist" not in system and "You write the lessons" not in system \
                and "repair WorldQuant" not in system:
            self.refused += 1
            raise RequestTooLarge("claude-test", "Limit 7000, Requested 9000", 9000, 7000)
        return super().complete(system, user, max_output=max_output)


def test_a_prompt_that_never_fits_ends_the_round(memory, config, playbook):
    llm = TooLargeGenerator()
    run = make_run(memory, config, playbook, llm=llm,
                   budget=Budget(max_simulations=8, per_round=4, max_llm_calls=100))
    report = run.run(2)
    # 60 (generator.yaml) -> 30 -> 20 fields, then no more alphas from this model
    assert report["generator_field_cap"] == {"claude-test": MIN_FIELDS_WHEN_TOO_LARGE}
    assert "claude-test" in report["generator_no_fit"]
    assert llm.refused == 3
    assert report["simulations_used"] == 0
    assert report["stop_reason"].startswith("every generator model is down")
    # no refill loop: at most one refill pass in the round
    assert len(report["rounds"][0].get("refills", [])) <= 1


class FitsAfterShrinking(RouterLLM):
    """The generator's request is refused until its prompt lists 30 fields or fewer."""

    def __init__(self, run_ref):
        super().__init__()
        self.run_ref = run_ref

    def complete(self, system, user, *, max_output=4096):
        run = self.run_ref[0]
        cap = run.field_cap.get("claude-test", 120) if run else 120
        if "Critic in a quant" not in system and "judge in a debate" not in system \
                and "research strategist" not in system and "You write the lessons" not in system \
                and "repair WorldQuant" not in system and cap > 30:
            raise RequestTooLarge("claude-test", "Limit 7000, Requested 9000", 9000, 7000)
        return super().complete(system, user, max_output=max_output)


def test_fewer_fields_let_the_model_write_again(memory, config, playbook):
    ref = [None]
    run = make_run(memory, config, playbook, llm=FitsAfterShrinking(ref))
    ref[0] = run
    report = run.run(1)
    assert report["generator_field_cap"] == {"claude-test": 30}
    assert report["generator_no_fit"] == {}
    assert report["simulations_used"] > 0
    gen = next(iter(run.generators.values()))
    assert gen.config.max_datafields_in_prompt == 30
