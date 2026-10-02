"""Facts from the project knowledge graph (Phase 2-5) as short citable statements: herb-drug research risk scores.
Always framed as research scores, never advice."""
import pandas as pd

CAVEAT = ("This is a research risk score from predicted CYP inhibition by the herb's compounds versus the drug's known CYP substrate status; "
          "it is a mechanistic hypothesis, not medical advice.")


class KGFacts:
    def __init__(self, risk: pd.DataFrame, max_facts=4):
        """risk columns: herb, drug, risk_rf, risk_silver, n_compounds (see phase5.build_risk)."""
        self.risk = risk
        self.max_facts = max_facts

    def _line(self, r) -> str:
        return (f"Research risk score for {r.herb} with {r.drug}: predicted interaction probability {r.risk_rf:.2f} "
                f"(mechanistic silver score {r.risk_silver:.2f}; {int(r.n_compounds)} compounds). {CAVEAT}")

    def facts(self, entities: list[str]) -> list[dict]:
        herbs = [e.split(":", 1)[1] for e in entities if e.startswith("herb:")]
        drugs = [e.split(":", 1)[1] for e in entities if e.startswith("drug:")]
        out = []
        if herbs and drugs:
            sub = self.risk[self.risk["herb"].isin(herbs) & self.risk["drug"].isin(drugs)].sort_values("risk_rf", ascending=False)
            out = [{"text": self._line(r), "source": "KG: herb-drug research risk"} for r in sub.itertuples()]
        elif herbs:
            for h in herbs:
                top = self.risk[self.risk["herb"] == h].sort_values("risk_rf", ascending=False).head(3)
                if len(top):
                    lst = "; ".join(f"{r.drug} {r.risk_rf:.2f}" for r in top.itertuples())
                    out.append({"text": f"Highest research risk scores for {h} against the scoped drugs: {lst}. {CAVEAT}", "source": "KG: herb-drug research risk"})
        elif drugs:
            for d in drugs:
                top = self.risk[self.risk["drug"] == d].sort_values("risk_rf", ascending=False).head(3)
                if len(top):
                    lst = "; ".join(f"{r.herb} {r.risk_rf:.2f}" for r in top.itertuples())
                    out.append({"text": f"Herbs with the highest research risk scores against {d}: {lst}. {CAVEAT}", "source": "KG: herb-drug research risk"})
        return out[:self.max_facts]
