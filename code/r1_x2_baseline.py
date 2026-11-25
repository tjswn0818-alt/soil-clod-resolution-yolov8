"""기준선: test20 seed 0, 이 코드(r1_x2_infer base)의 TP/FP/FN·KS D vs unified_4cm (unified_4cm_260914.py 출력).
출력 x2_baseline_check.csv"""
import csv
import math

import r1_x2_common as X
import r1_x2_infer as I

UN = I.U.OUT
cells = {(r["model"], int(r["eval_res"])): r for r in csv.DictReader(
    open(UN / "unified_4cm_cells_260914.csv", encoding="utf-8-sig"))
    if r["dataset"] == "test" and float(r["seed"]) == 0}
frag = {(r["model"], int(float(r["eval_res"]))): r for r in csv.DictReader(
    open(UN / "unified_4cm_frag_seed_metrics_260914.csv", encoding="utf-8-sig"))
    if r["dataset"] == "test" and float(r["seed"]) == 0}
rows, nok = [], 0
conds = [(l, ev) for l, _ in I.MR for ev in I.EVALS] + [(l, ev) for l, _, ev in I.SR + I.SR_AUX]
for lbl, ev in conds:
    files, A, ped, ged = X.per_image(X.load("test20", lbl, ev, 0))
    tp, fn, fp, _ = A.sum(0)
    u = cells[(lbl, ev)]
    ks = X.ks_d(ped, ged)
    uk = float(frag[(lbl, ev)]["ED_KS_D"]) if (lbl, ev) in frag and frag[(lbl, ev)]["ED_KS_D"] else float("nan")
    ok = (tp, fp, fn) == (int(u["TP"]), int(u["FP"]), int(u["FN"])) and (
        abs(ks - uk) < 1e-4 if not math.isnan(uk) else True)
    nok += ok
    rows.append([lbl, ev, tp, int(u["TP"]), fp, int(u["FP"]), fn, int(u["FN"]),
                 round(ks, 5), round(uk, 5), int(ok)])
    print(rows[-1], flush=True)
with open(I.OUT / "x2_baseline_check.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "eval_res", "TP_this", "TP_unified", "FP_this", "FP_unified", "FN_this", "FN_unified",
                "KS_D_this", "KS_D_unified", "match"])
    w.writerows(rows)
print(f"match {nok}/{len(rows)}")
