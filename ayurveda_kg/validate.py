"""KG quality gate. Every check returns a list of human-readable problems; empty list = clean."""
from ayurveda_kg.kg import SCHEMA


def _type_of(nodes):
    return {i: t for t, df in nodes.items() for i in df["id"]}


def check_nodes(nodes) -> list[str]:
    out = []
    for t, df in nodes.items():
        for i in df["id"][df["id"].duplicated()].unique():
            out.append(f"duplicate node id in {t}: {i}")
    seen = {}
    for t, df in nodes.items():
        for i in df["id"].unique():
            if i in seen and seen[i] != t:
                out.append(f"node id used by two types ({seen[i]}, {t}): {i}")
            seen[i] = t
    return out


def check_edges(nodes, edges) -> list[str]:
    tmap, out = _type_of(nodes), []
    for et, df in edges.items():
        if et not in SCHEMA:
            out.append(f"unknown edge type: {et}")
            continue
        s_t, d_t = SCHEMA[et]
        for r in df.itertuples():
            if r.src not in tmap or r.dst not in tmap:
                out.append(f"dangling {et} edge: {r.src} -> {r.dst}")
            elif (tmap[r.src], tmap[r.dst]) != (s_t, d_t):
                out.append(f"type mismatch on {et}: {tmap[r.src]}({r.src}) -> {tmap[r.dst]}({r.dst}), expected {s_t}->{d_t}")
    return out


def check_scope(nodes, edges, scope) -> list[str]:
    out = []
    herbs = set(nodes["Herb"]["name"]) if "Herb" in nodes else set()
    drugs = set(nodes["Drug"]["name"]) if "Drug" in nodes else set()
    out += [f"herb missing from KG: {h['imppat_name']}" for h in scope["herbs"] if h["imppat_name"] not in herbs]
    out += [f"drug missing from KG: {d['name']}" for d in scope["drugs"] if d["name"] not in drugs]
    has = set(edges["contains"]["src"]) if "contains" in edges else set()
    if "Herb" in nodes:
        out += [f"herb has no compounds: {n}" for i, n in zip(nodes["Herb"]["id"], nodes["Herb"]["name"]) if i not in has]
    return out


def validate_all(nodes, edges, scope=None) -> list[str]:
    return check_nodes(nodes) + check_edges(nodes, edges) + (check_scope(nodes, edges, scope) if scope else [])
