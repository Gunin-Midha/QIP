import json

from alpha_research.llm import ScriptedLLMClient
from alpha_research.schemas import DecisionRecord, ResultRecord
from alpha_research.strategy import StrategyConfig, StrategyEvolution, rule_guidance, summarize


def store(memory, cand, decision, failure_class="none", sharpe=None):
    memory.put_candidate(cand)
    memory.put_result(ResultRecord(candidate_id=cand.candidate_id, status="complete", sharpe=sharpe))
    memory.put_decision(DecisionRecord(cand.candidate_id, decision, failure_class))


def seed(memory, candidate):
    store(memory, candidate(candidate_id="a1", family="reversal"), "accept", sharpe=1.9)
    store(memory, candidate(candidate_id="a2", family="reversal", expression="rank(volume)"),
          "accept", sharpe=2.4)
    store(memory, candidate(candidate_id="t1", family="momentum", expression="rank(returns)"),
          "repair", "turnover", sharpe=1.6)
    store(memory, candidate(candidate_id="t2", family="momentum", expression="rank(vwap)"),
          "reject", "turnover", sharpe=1.3)
    memory.put_candidate(candidate(candidate_id="undecided", expression="rank(open)"))


def test_summarize_counts(memory, candidate):
    seed(memory, candidate)
    s = summarize(memory.candidate_outcomes())
    assert (s["decided"], s["accepted"]) == (4, 2)
    assert s["failure_classes"] == {"turnover": 2}
    assert s["families"]["reversal"] == {"decided": 2, "accepted": 2, "failures": {}}
    assert s["best_accepted"][0]["expression"] == "rank(volume)"
    assert s["arms"]["pv1/USA/none"] == {"decided": 4, "accepted": 2}
    g = rule_guidance(s)
    assert "2 of 4" in g and "turnover is the most common failure" in g and "reversal" in g


def test_nothing_to_learn_from(memory):
    assert StrategyEvolution(StrategyConfig()).evolve(memory.candidate_outcomes(), "gen-v1") is None


def test_rule_only_evolution_versions_increment(memory, candidate):
    seed(memory, candidate)
    evo = StrategyEvolution(StrategyConfig())
    first = evo.evolve(memory.candidate_outcomes(), "gen-v1")
    second = evo.evolve(memory.candidate_outcomes(), "gen-v1")
    assert (first.prompt_version, second.prompt_version) == ("gen-v1+evo1", "gen-v1+evo2")
    assert first.source == "rules" and first.hypotheses == []


def test_model_evolution(memory, candidate):
    seed(memory, candidate)
    reply = json.dumps({"guidance": ["slow momentum down", "keep reversal"],
                        "hypotheses": [{"hypothesis": "h1", "dataset": "pv1"}, "h2",
                                       {"hypothesis": "h3"}, {"hypothesis": "h4"}, {"bad": 1}]})
    llm = ScriptedLLMClient([reply])
    evo = StrategyEvolution(StrategyConfig(max_hypotheses=3), llm=llm).evolve(
        memory.candidate_outcomes(), "gen-v1")
    assert evo.source == "model" and evo.guidance == "- slow momentum down\n- keep reversal"
    assert evo.hypotheses == [("pv1", "h1"), (None, "h2"), (None, "h3")]
    assert "<records>" in llm.calls[0]["user"] and evo.llm_calls == 1


def test_unusable_model_reply_keeps_rule_guidance(memory, candidate):
    seed(memory, candidate)
    evo = StrategyEvolution(StrategyConfig(), llm=ScriptedLLMClient(["?", "?"])).evolve(
        memory.candidate_outcomes(), "gen-v1")
    assert evo.source == "rules" and "unusable" in evo.notes[0]


def test_due_and_guidance_cap():
    s = StrategyEvolution(StrategyConfig(every_rounds=2, max_guidance_chars=10))
    assert [s.due(r) for r in (1, 2, 3, 4)] == [False, True, False, True]
    assert not StrategyEvolution(StrategyConfig(every_rounds=0)).due(2)
