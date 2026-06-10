"""Table 3 (tab:operating-point) — cross-dataset threshold transfer.

Rows = target dataset; columns = tau=0 (no threshold) plus each dataset as the
SOURCE of a borrowed threshold. For every (model, source) the MCC-optimal
cutoff tau* is determined on the source, then applied unchanged to the target;
each cell reports MCC and accept-error AE at that operating point,
mean±std over the eight paper models. Shaded diagonal = in-dataset oracle.
Deltas vs tau=0 are computed before rounding (as in the paper caption).

Run:  python -m src.analysis.make_table3_operating_point
Out:  results/tables/table3_operating_point.md  +  .tex (paper cell macros)  +  console
"""
from __future__ import annotations

from . import common as C


def main() -> None:
    M = C.models()
    rows = {(m, d): C.load(m, d) for m in M for d in C.DATASETS}
    for k, v in rows.items():
        assert v is not None, f"missing primary-cell predictions: {k}"
    tau_mcc = {(m, d): C.best_tau(rows[(m, d)], C.mcc_at) for m in M for d in C.DATASETS}

    def cell(target: str, tau_of) -> tuple[tuple, tuple]:
        """((MCC mean, std), (AE mean, std)) on `target` at tau_of(model)."""
        mc = [C.mcc_at(rows[(m, target)], tau_of(m)) for m in M]
        ae = [C.ae_at(rows[(m, target)], tau_of(m)) for m in M]
        return C.mean_std(mc), C.mean_std(ae)

    md = ["# Table 3 — Cross-dataset transfer (MCC / accept-error at the "
          "source's MCC-optimal cutoff)",
          "",
          f"mean±std over {len(M)} models; `[oracle]` = in-dataset diagonal; "
          "Δ vs τ=0 computed before rounding.", ""]
    hdr = ["target", "τ=0"] + [f"τ from {C.SHORT[s]}" for s in C.DATASETS]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))

    tex = ["% Table 3 (tab:operating-point) rows — regenerated with the paper's cell macros",
           "% (two-decimal cells, matching the manuscript's rendering)",
           "% \\bcell{v}{s} (tau0)  \\ocell{v}{s}{d} (oracle diag)  \\gcell/\\rcell{v}{s}{d}",
           "% (better/worse MCC)  \\zcell{v}{s} (delta rounds to 0)",
           "% \\aeb{v}{s} (tau0 AE)  \\aeo{v}{s}{d} (oracle AE)  \\aed{v}{s}{d} (AE down=better)"]

    def f2(x: float) -> str:
        return f"{x:.2f}".lstrip("0") if x < 1 else f"{x:.2f}"

    for t in C.DATASETS:
        (m0, s0), (a0, as0) = cell(t, lambda m: 0.0)
        md_cells = [C.SHORT[t], f"{m0:.2f}±{s0:.2f} / AE {a0:.2f}±{as0:.2f}"]
        tex_cells = [f"\\bcell{{{f2(m0)}}}{{{f2(s0)}}} & \\aeb{{{f2(a0)}}}{{{f2(as0)}}}"]
        for s in C.DATASETS:
            (mc, mcs), (ae, aes) = cell(t, lambda m, s=s: tau_mcc[(m, s)])
            dm, da = mc - m0, ae - a0
            oracle = (s == t)
            tag = " [oracle]" if oracle else ""
            md_cells.append(f"{mc:.2f}±{mcs:.2f} (Δ{dm:+.2f}) / "
                            f"AE {ae:.2f}±{aes:.2f} (Δ{da:+.2f}){tag}")
            if oracle:
                mcell = f"\\ocell{{{f2(mc)}}}{{{f2(mcs)}}}{{{abs(dm):.2f}}}"
                acell = f"\\aeo{{{f2(ae)}}}{{{f2(aes)}}}{{{abs(da):.2f}}}"
            else:
                if round(dm, 2) == 0.0:
                    mcell = f"\\zcell{{{f2(mc)}}}{{{f2(mcs)}}}"
                else:
                    mac = "gcell" if dm > 0 else "rcell"
                    mcell = f"\\{mac}{{{f2(mc)}}}{{{f2(mcs)}}}{{{abs(dm):.2f}}}"
                if da <= 0:   # AE down = improvement (the regular case)
                    acell = f"\\aed{{{f2(ae)}}}{{{f2(aes)}}}{{{abs(da):.2f}}}"
                else:         # AE up = worse — no paper macro exists; flag loudly
                    acell = f"\\aed{{{f2(ae)}}}{{{f2(aes)}}}{{{abs(da):.2f}}} % WARNING: AE INCREASED"
            tex_cells.append(f"{mcell} & {acell}")
        md.append("| " + " | ".join(md_cells) + " |")
        tex.append(f"{C.SHORT[t]} & " + " & ".join(tex_cells) + r" \\")

    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "table3_operating_point.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (C.OUT_DIR / "table3_operating_point.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {C.OUT_DIR / 'table3_operating_point.md'} (+ .tex)")


if __name__ == "__main__":
    main()
