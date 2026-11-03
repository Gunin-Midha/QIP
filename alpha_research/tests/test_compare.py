"""Model comparison tables from memory: generator credit by arm, critic scoring."""

import json

from alpha_research import cli
from alpha_research.compare import arm_models, critic_comparison, generator_comparison
from alpha_research.schemas import DecisionRecord


def review(memory, cid, critic, role, verdict, gen, source="model"):
    memory.put_critic_review(run_id="rs1", candidate_id=cid, critic_model=critic, role=role,
                             verdict=verdict, source=source, generator_model=gen)


def seed(memory, candidate):
    # claude-a arm: g1 accepted; g2 rejected, its repair child g2_r1 accepted
    # gemini-b arm: h1 rejected; h2 revised by the critic into h2_c1 (h2 itself never stored)
    memory.put_candidate(candidate(candidate_id="g1", model="claude-a"))
    memory.put_candidate(candidate(candidate_id="g2", model="claude-a", expression="rank(close)"))
    memory.put_candidate(candidate(candidate_id="g2_r1", model="claude-a", parent_id="g2",
                                   lever="increase_decay", generation=1, decay=8,
                                   expression="rank(close)"))
    memory.put_candidate(candidate(candidate_id="h1", model="gemini-b", expression="rank(open)"))
    memory.put_candidate(candidate(candidate_id="h2_c1", model="claude-a", parent_id="h2",
                                   lever="critic_revision", generation=1, expression="rank(high)"))
    for cid, d in [("g1", "accept"), ("g2", "repair"), ("g2_r1", "accept"), ("h1", "reject"),
                   ("h2_c1", "accept")]:
        memory.put_decision(DecisionRecord(cid, d, "none" if d == "accept" else "turnover"))
    # primary critics come from the other provider; the shadow is the generator's own provider
    for cid, gen, primary, shadow, pv, sv in [
            ("g1", "claude-a", "gemini-b", "claude-a", "simulate", "simulate"),
            ("g2", "claude-a", "gemini-b", "claude-a", "simulate", "drop"),
            ("g2_r1", "claude-a", "gemini-b", "claude-a", "simulate", "simulate"),
            ("h1", "gemini-b", "claude-a", "gemini-b", "simulate", "simulate"),
            ("h2", "gemini-b", "claude-a", "gemini-b", "revise", "simulate")]:
        review(memory, cid, primary, "primary", pv, gen)
        review(memory, cid, shadow, "shadow", sv, gen)


def test_arm_models_follow_lineage_and_critic_records(memory, candidate):
    seed(memory, candidate)
    models = arm_models(memory.candidate_outcomes(), memory.critic_reviews())
    # the critic revision was written by claude-a but belongs to the gemini-b arm
    assert models == {"g1": "claude-a", "g2": "claude-a", "g2_r1": "claude-a",
                      "h1": "gemini-b", "h2_c1": "gemini-b"}


def test_arm_models_fall_back_to_the_root_candidate(memory, candidate):
    memory.put_candidate(candidate(candidate_id="x", model="claude-a"))
    memory.put_candidate(candidate(candidate_id="x_r1", model="claude-a", parent_id="x",
                                   lever="increase_decay", generation=1, decay=8))
    assert arm_models(memory.candidate_outcomes(), []) == {"x": "claude-a", "x_r1": "claude-a"}


def test_generator_comparison(memory, candidate):
    seed(memory, candidate)
    table = generator_comparison(memory.candidate_outcomes(), memory.critic_reviews())
    a, b = table["claude-a"], table["gemini-b"]
    assert (a["candidates"], a["accept"], a["repair"], a["accept_rate"]) == (3, 2, 1, 0.6667)
    assert (b["candidates"], b["accept"], b["reject"], b["accept_rate"]) == (2, 1, 1, 0.5)


def test_critic_comparison(memory, candidate):
    seed(memory, candidate)
    table = critic_comparison(memory.critic_reviews(), memory.candidate_outcomes())
    g = table["gemini-b"]   # primary on g1, g2, g2_r1; shadow on h1, h2
    assert (g["reviews"], g["as_primary"], g["as_shadow"]) == (5, 3, 2)
    assert g["shadow_agreements"] == 1 and g["shadow_agreement_rate"] == 0.5   # h1 agrees
    # simulated and decided: g1 accept, g2 repair, g2_r1 accept, h1 reject (h2 never ran)
    assert (g["cleared_accepted"], g["cleared_not_accepted"]) == (2, 2)
    assert g["cleared_accept_rate"] == 0.5 and g["flag_precision"] is None
    c = table["claude-a"]   # shadow on g1, g2, g2_r1; primary on h1, h2
    assert c["shadow_agreements"] == 2 and c["shadow_agreement_rate"] == 0.6667
    # the shadow's drop on g2 was right (repair, not accept)
    assert (c["flagged_accepted"], c["flagged_not_accepted"], c["flag_precision"]) == (0, 1, 1.0)
    assert (c["simulate"], c["revise"], c["drop"]) == (3, 1, 1)


def test_rule_fallbacks_and_rules_only_critic_are_counted(memory, candidate):
    memory.put_candidate(candidate(candidate_id="g1", model="claude-a"))
    review(memory, "g1", "gemini-b", "primary", "simulate", "claude-a", source="rules")
    review(memory, "g1", None, "primary", "simulate", "claude-a", source="rules")
    table = critic_comparison(memory.critic_reviews(), memory.candidate_outcomes())
    assert table["gemini-b"]["rule_fallback"] == 1 and table["gemini-b"]["from_model"] == 0
    assert table["rules only"]["reviews"] == 1


def test_cli_compare(tmp_path, memory, candidate, capsys):
    seed(memory, candidate)
    out = tmp_path / "cmp.json"
    assert cli.main(["compare", "--data-dir", str(memory.data_dir), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert set(data) == {"generators", "critics"}
    assert data["generators"]["claude-a"]["accept"] == 2
    assert json.loads(capsys.readouterr().out) == data
