"""Generate the committed sample games under docs/samples/.

These are real engine output from the scripted baselines (no GPU needed), chosen
to illustrate the two headline dynamics: cost-sharing + fraud, and death under
privilege. Run: `python scripts/make_samples.py`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import PRESETS
from agora.policies import REGISTRY
from agora.referee import Referee
from agora.transcripts import Transcript
from analysis.viz import render_html

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "samples")


def make(name, cfg, spec, title):
    os.makedirs(OUT, exist_ok=True)
    ids = cfg.agent_ids
    names = spec.split(",")
    policies = {a: REGISTRY[names[i % len(names)]](cfg, a, ids) for i, a in enumerate(ids)}
    tx = Transcript(os.path.join(OUT, f"{name}.jsonl"))
    Referee(cfg, policies, tx).run()
    tx.close()
    with open(os.path.join(OUT, f"{name}.html"), "w") as fh:
        fh.write(render_html(tx.events, title))
    print(f"wrote docs/samples/{name}.html  ({len(tx.events)} events)")


if __name__ == "__main__":
    # A: an honest cooperator, a solo Bayesian, a LIAR, and a hoarder. The liar
    # sells fabricated values every round (flagged in the report).
    make("cooperation_and_fraud",
         PRESETS["base"].with_(seed=7),
         "honest_cooperator,bayesian_solo,liar,hoarder",
         "Agora — cost-sharing and fraud")

    # B: heterogeneous noise + a survival cost, so a weak agent can be driven to
    # ruin and eliminated.
    make("broker_and_death",
         PRESETS["privilege"].with_(seed=4, survival_cost=1.5, gamma=0.85, n_rounds=10),
         "honest_cooperator,bayesian_solo,liar,hoarder",
         "Agora — privilege, brokers, and death")
