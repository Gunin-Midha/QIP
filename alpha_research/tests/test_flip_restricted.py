"""repair.flip_short_restricted (owner 2026-11-27): try flip_sign in CHN too, one
simulation per negative near miss, to measure whether the opposite alpha mirrors there.
Off by default (BRAIN's guide); the real config turns it on."""

import dataclasses

from alpha_research.brain_client import FixtureBrainClient
from alpha_research.config import load_config
from alpha_research.pipeline import Pipeline
from alpha_research.repair_agent import MECHANICAL, RepairAgent
from alpha_research.schemas import ResultRecord

from conftest import FIXTURES, ROOT
from test_rescue_lessons import SPEC

CHN = dict(region="CHN", universe="TOP2000U", neutralization="INDUSTRY")


def _result():
    return ResultRecord("c1", "complete", sharpe=-2.23, fitness=-2.13, turnover=0.3,
                        checks={"LOW_SHARPE": False},
                        check_values={"LOW_SHARPE": {"value": -2.23, "limit": 2.07}})


def _flip_on(config):
    return dataclasses.replace(config, repair=dataclasses.replace(
        config.repair, flip_short_restricted=True))


def test_default_off_and_real_config_on(config):
    assert config.repair.flip_short_restricted is False
    assert not config.repair.sign_mirrors("CHN") and config.repair.sign_mirrors("USA")
    real = load_config(ROOT / "config" / "thresholds.yaml")
    assert real.repair.sign_mirrors("CHN")


def test_chn_negative_near_miss_goes_to_repair_when_on(config, playbook, memory, candidate):
    on = Pipeline(_flip_on(config), playbook, FixtureBrainClient(FIXTURES / "brain_results"),
                  memory)
    d = on._evaluate(candidate(**CHN), _result()).decision_record
    assert (d.decision, d.failure_class) == ("repair", "sharpe")
    assert any(r.startswith("sign first") or "negative" in r for r in d.reasons)


def test_flip_lever_applies_in_chn_only_when_allowed(candidate, playbook, memory):
    r = _result()
    c = candidate(**CHN)
    flipped = MECHANICAL["flip_sign"](c, {"allow_restricted": True}, SPEC, r)
    assert flipped.expression.startswith("-")
    agent = RepairAgent(playbook, SPEC, memory)
    agent.flip_restricted = True
    lever = next(l for l in playbook.entries["sharpe"] if l.name == "flip_sign")
    out = agent._apply(c, lever, None, r, None, [], type("O", (), {"notes": []})())
    assert out.expression.startswith("-")


def test_reevaluate_brings_back_chn_rejects(config, playbook, memory, candidate):
    c = candidate(**CHN)
    memory.put_candidate(c)
    memory.put_result(dataclasses.replace(_result(), candidate_id=c.candidate_id))
    off = Pipeline(config, playbook, FixtureBrainClient(FIXTURES / "brain_results"), memory)
    d = off._evaluate(c, dataclasses.replace(_result(), candidate_id=c.candidate_id)).decision_record
    memory.put_decision(d)
    assert d.decision == "reject" and off.reevaluate_rejected() == []
    on = Pipeline(_flip_on(config), playbook, FixtureBrainClient(FIXTURES / "brain_results"),
                  memory)
    outs = on.reevaluate_rejected()
    assert [o.decision_record.decision for o in outs] == ["repair"]
