"""
Parametric, seeded simulation for the three-layer eGFR-dip target trial emulation.
Outputs results.json (base64 matplotlib PNGs + table rows) consumed by build_html.py.
All numbers are simulated; the goal is to freeze the figure / table set, not to estimate anything.
"""
import numpy as np, pandas as pd, io, base64, json, os, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

SEED = 20260914
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- palette ----------
AI, SHU, NEZUMI, SUMI = "#1C3F60", "#C2503A", "#8A8A88", "#2B2B2B"
ASAGI, KIHADA, MOEGI, HAI, KINARI = "#3E8EA8", "#C9A227", "#5B8C4A", "#E8E7E3", "#FBFAF7"
BLUES = ["#B9CBDA", "#8FAAC2", "#5F80A0", "#1C3F60"]          # ordinal dip categories A–D
CAT_COL = {"A": BLUES[0], "B": BLUES[1], "C": BLUES[2], "D": BLUES[3], "E": SHU}
CAT_LAB = {"A": "no dip (>0%)", "B": "0–10%", "C": "10–20%", "D": "20–30%", "E": ">30%"}

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B8B7B3", "axes.linewidth": 0.8,
    "xtick.color": SUMI, "ytick.color": SUMI, "axes.labelcolor": SUMI,
    "axes.titleweight": "normal", "axes.titlesize": 10, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 150,
})

def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def fmt(x, d=1): return f"{x:.{d}f}"
def ci(lo, hi, d=1): return f"({lo:.{d}f} to {hi:.{d}f})"

# ---------- helpers ----------
def logit_fit(X, y, iters=30):
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ b)); W = p * (1 - p)
        H = X1.T @ (X1 * W[:, None]) + 1e-6 * np.eye(len(b))
        b = b + np.linalg.solve(H, X1.T @ (y - p))
    return b, 1 / (1 + np.exp(-X1 @ b))

def auroc(score, y):
    r = pd.Series(score).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1]); v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2 + 1e-9)

GRID = np.linspace(0, 3, 73)
def cif(t, e, w=None, grid=GRID):
    """Aalen–Johansen cumulative incidence of cause 1 (2 = competing death, 0 = censored), weighted."""
    w = np.ones(len(t)) if w is None else w
    o = np.argsort(t, kind="stable"); t, e, w = t[o], e[o], w[o]
    ut, start = np.unique(t, return_index=True)
    d1 = np.add.reduceat(w * (e == 1), start); d2 = np.add.reduceat(w * (e == 2), start)
    wt = np.add.reduceat(w, start); atrisk = w.sum() - np.concatenate([[0], np.cumsum(wt)[:-1]])
    h = (d1 + d2) / atrisk; S = np.cumprod(1 - h); Sprev = np.concatenate([[1.0], S[:-1]])
    F1 = np.cumsum(Sprev * d1 / atrisk)
    idx = np.searchsorted(ut, grid, side="right") - 1
    return np.where(idx >= 0, F1[np.clip(idx, 0, None)], 0.0)

def boot_cif(t, e, w=None, B=200, grid=GRID):
    n = len(t); w = np.ones(n) if w is None else w
    out = np.empty((B, len(grid)))
    for b in range(B):
        i = rng.integers(0, n, n); out[b] = cif(t[i], e[i], w[i], grid)
    return out

def rcs_basis(x, knots):
    """Harrell's restricted cubic spline basis (without intercept)."""
    k = np.asarray(knots); K = len(k); tk = k[-1]; tk1 = k[-2]
    def p3(z): return np.maximum(z, 0) ** 3
    cols = [x]
    for j in range(K - 2):
        cols.append((p3(x - k[j]) - p3(x - tk1) * (tk - k[j]) / (tk - tk1) + p3(x - tk) * (tk1 - k[j]) / (tk - tk1)) / (tk - k[0]) ** 2)
    return np.column_stack(cols)

def rcs_fit(x, y, knots, xg, weights=None):
    X = np.column_stack([np.ones(len(x)), rcs_basis(x, knots)]); w = np.ones(len(x)) if weights is None else weights
    XtW = X.T * w; beta = np.linalg.solve(XtW @ X, XtW @ y)
    res = y - X @ beta; s2 = np.sum(w * res ** 2) / (w.sum() - X.shape[1]); cov = s2 * np.linalg.inv(XtW @ X)
    Xg = np.column_stack([np.ones(len(xg)), rcs_basis(xg, knots)])
    yhat = Xg @ beta; se = np.sqrt(np.sum((Xg @ cov) * Xg, 1))
    return yhat, se

def rcs_logit(x, y, knots, xg):
    Xb = rcs_basis(x, knots); b, _ = logit_fit(Xb, y)
    X = np.column_stack([np.ones(len(x)), Xb]); p = 1 / (1 + np.exp(-X @ b))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    Xg = np.column_stack([np.ones(len(xg)), rcs_basis(xg, knots)]); lp = Xg @ b; se = np.sqrt(np.sum((Xg @ cov) * Xg, 1))
    f = lambda z: 1 / (1 + np.exp(-z))
    return f(lp), f(lp - 1.96 * se), f(lp + 1.96 * se)

# =====================================================================
# 1. Cohort: CKD G2–G4 + T2DM + UACR ≥30, new users of SGLT2i vs DPP-4i
# =====================================================================
N = 12480
stage = rng.choice(["G2", "G3a", "G3b", "G4"], N, p=[0.22, 0.32, 0.30, 0.16])
base_egfr = {"G2": (71, 7), "G3a": (52, 4.5), "G3b": (38, 4.3), "G4": (23, 4)}
egfr0 = np.array([rng.normal(*base_egfr[s]) for s in stage])
egfr0 = np.where(stage == "G2", egfr0.clip(60, 89), egfr0).clip(15, 89)
age = rng.normal(66, 11, N).clip(25, 92)
male = (rng.random(N) < 0.55).astype(float)
uacr = np.exp(rng.normal(5.2, 1.3, N)).clip(30, 8000)
alb300 = (uacr > 300).astype(float)
preslope = rng.normal(-2.4 - 1.5 * alb300, 2.6, N)             # mL/min/1.73m²/yr, 12 months before index
crcv = rng.normal(9 + 3 * alb300, 3, N).clip(2, 30)              # creatinine CV %, pre-index
hf = (rng.random(N) < 0.17).astype(float)
rasi = (rng.random(N) < 0.76).astype(float)
loop = (rng.random(N) < 0.22 + 0.35 * hf).astype(float)
mra = (rng.random(N) < 0.06 + 0.15 * hf).astype(float)
nsaid = (rng.random(N) < 0.12).astype(float)
glp1 = (rng.random(N) < 0.14).astype(float)
sbp = rng.normal(136, 15, N)
hba1c = rng.normal(7.7, 1.2, N).clip(5, 13)
k = rng.normal(4.5, 0.45, N)
aki_hx = (rng.random(N) < 0.11).astype(float)
hosp90 = (rng.random(N) < 0.09).astype(float)
year = rng.integers(2018, 2025, N)

# treatment assignment (confounded)
lp = 0.1 + 0.30 * (year - 2021) + 0.018 * (egfr0 - 45) - 0.018 * (age - 66) + 0.35 * hf + 0.25 * alb300 - 0.45 * aki_hx + 0.25 * glp1 - 0.3 * hosp90
trt = (rng.random(N) < 1 / (1 + np.exp(-lp))).astype(float)     # 1 = SGLT2i, 0 = DPP-4i

# latent hemodynamic instability (drives the excessive-dip / adverse phenotype)
u = (rng.random(N) < 0.05 + 0.04 * loop + 0.03 * aki_hx + 0.03 * hosp90).astype(float)

# acute dip (%), reached by ~day 14
mu_dip = np.where(trt == 1, -5.5 - 4.0 * loop - 2.5 * rasi - 0.12 * (sbp - 136) - 2.0 * hf + 0.06 * (egfr0 - 45) - 1.5 * nsaid - 2.0 * aki_hx - 2.0 * hosp90 - 0.5 * (crcv - 9) / 3, -1.0 - 1.0 * loop)
dip = rng.normal(mu_dip, np.where(trt == 1, 5.5, 5.0)) + u * rng.normal(np.where(trt == 1, -18, -12), 7)
dip = dip.clip(-55, 25)

# chronic slope from day 14 on (true), mL/min/1.73m²/yr
slope0 = 0.55 * preslope + rng.normal(-1.5, 1.6, N) - 1.0 * alb300 - 0.3 * hf
phys = np.clip(dip, -25, 0)                                       # physiological component of the dip
trt_eff = np.where(u == 1, 0.7, 1.35) - 0.025 * phys * (1 - u)     # plateau: modest extra flattening with physiological dip
slope_trt = slope0 + trt_eff - 1.8 * u                            # SGLT2i
slope_ctl = slope0 - 1.2 * u                                      # DPP-4i
slope = np.where(trt == 1, slope_trt, slope_ctl)

# ---------- measurement schedule ----------
DAYS = np.array([-365, -270, -180, -90, 0, 14, 30, 45, 60, 90, 180, 270, 365, 455, 545, 640, 730, 820, 910, 1000, 1095])
def egfr_at(t, e0, dp, sl, ps):
    t = np.asarray(t, float)
    pre = e0[:, None] + ps[:, None] * t[None, :] / 365
    acute = e0[:, None] * (1 + dp[:, None] / 100 * (1 - np.exp(-np.maximum(t, 0)[None, :] / 6)))
    post = acute + sl[:, None] * np.maximum(t - 14, 0)[None, :] / 365
    return np.where(t[None, :] < 0, pre, post)
TRUE = egfr_at(DAYS, egfr0, dip, slope, preslope)
OBS = TRUE + rng.normal(0, 2.3, TRUE.shape)
miss = rng.random(TRUE.shape) < 0.22; miss[:, DAYS == 0] = False
OBS = np.where(miss, np.nan, OBS).clip(4, None)

# observed acute response = mean eGFR at day 30–60 vs day 0
w48 = np.isin(DAYS, [30, 45, 60])
e_4to8 = np.nanmean(np.where(np.isnan(OBS[:, w48]), np.nan, OBS[:, w48]), 1)
e_4to8 = np.where(np.isnan(e_4to8), OBS[:, DAYS == 14].ravel(), e_4to8)
dip_obs = (e_4to8 - OBS[:, DAYS == 0].ravel()) / OBS[:, DAYS == 0].ravel() * 100
dip_obs = np.where(np.isnan(dip_obs), dip + rng.normal(0, 4, N), dip_obs)

def cat_of(d):
    return np.select([d > 0, d > -10, d > -20, d > -30], ["A", "B", "C", "D"], "E")
cat = cat_of(dip_obs)

# ---------- outcomes (years from index) ----------
FU_MAX = 3.0
lam_comp = 0.030 * np.exp(-0.055 * (egfr0 - 45)) * np.exp(-0.25 * (slope + 2)) * np.exp(0.45 * alb300) * np.exp(0.9 * u)
lam_death = 0.024 * np.exp(0.035 * (age - 66)) * np.exp(0.5 * u + 0.4 * hf) * np.where(trt == 1, 0.85, 1.0)
lam_aki = 0.040 * np.exp(-0.03 * (egfr0 - 45)) * np.exp(1.3 * u + 0.3 * loop + 0.3 * aki_hx) * np.where(trt == 1, 0.85, 1.0)
lam_hyperk = 0.030 * np.exp(0.5 * rasi + 0.7 * mra + 0.6 * (k > 5)) * np.where(trt == 1, 0.85, 1.0) * np.exp(-0.03 * (egfr0 - 45))
lam_hosp = 0.16 * np.exp(0.5 * hf + 0.3 * u) * np.where(trt == 1, 0.90, 1.0)
lam_cens = 0.08
U1, U2, U3, U4, U5, U6 = [rng.random(N) for _ in range(6)]
t_comp = -np.log(U1) / lam_comp; t_death = -np.log(U2) / lam_death; t_cens = np.minimum(-np.log(U3) / lam_cens, FU_MAX)
t_aki = -np.log(U4) / lam_aki; t_hk = -np.log(U5) / lam_hyperk; t_hosp = -np.log(U6) / lam_hosp

def compete(t1, td, tc):
    t = np.minimum.reduce([t1, td, tc]); e = np.where(t == t1, 1, np.where(t == td, 2, 0)); return t, e
T_comp, E_comp = compete(t_comp, t_death, t_cens)
T_aki, E_aki = compete(t_aki, t_death, t_cens)
T_hk, E_hk = compete(t_hk, t_death, t_cens)
T_hosp, E_hosp = compete(t_hosp, t_death, t_cens)
T_d, E_d = compete(t_death, np.full(N, np.inf), t_cens)

# ---------- IPTW ----------
COV = pd.DataFrame(dict(age=age, male=male, egfr0=egfr0, loguacr=np.log(uacr), preslope=preslope, crcv=crcv, hf=hf, rasi=rasi,
                        loop=loop, mra=mra, nsaid=nsaid, glp1=glp1, sbp=sbp, hba1c=hba1c, k=k, aki_hx=aki_hx, hosp90=hosp90, year=year))
Z = (COV - COV.mean()) / COV.std()
_, ps = logit_fit(Z.values, trt)
ipw = np.where(trt == 1, trt.mean() / ps, (1 - trt.mean()) / (1 - ps))   # stabilised ATE weights
ipw = np.clip(ipw, np.percentile(ipw, 0.5), np.percentile(ipw, 99.5))

# =====================================================================
# Table 1 — baseline by arm, SMD crude / weighted
# =====================================================================
def row_cont(name, x, d=1, unit=""):
    return [name, f"{x[trt==1].mean():.{d}f} ({x[trt==1].std():.{d}f})", f"{x[trt==0].mean():.{d}f} ({x[trt==0].std():.{d}f})",
            f"{abs(smd(x, trt)):.2f}", f"{abs(smd(x, trt, ipw)):.2f}"]
def row_bin(name, x):
    return [name, f"{100*x[trt==1].mean():.1f}%", f"{100*x[trt==0].mean():.1f}%", f"{abs(smd(x, trt)):.2f}", f"{abs(smd(x, trt, ipw)):.2f}"]
T1 = [["n", f"{int(trt.sum()):,}", f"{int((1-trt).sum()):,}", "", ""],
      row_cont("Age, y", age), row_bin("Male", male), row_cont("eGFR at index, mL/min/1.73 m²", egfr0),
      ["UACR, mg/g, median [IQR]", f"{np.median(uacr[trt==1]):.0f} [{np.percentile(uacr[trt==1],25):.0f}–{np.percentile(uacr[trt==1],75):.0f}]",
       f"{np.median(uacr[trt==0]):.0f} [{np.percentile(uacr[trt==0],25):.0f}–{np.percentile(uacr[trt==0],75):.0f}]", f"{abs(smd(np.log(uacr), trt)):.2f}", f"{abs(smd(np.log(uacr), trt, ipw)):.2f}"],
      row_cont("Pre-index eGFR slope (12 mo), /yr", preslope), row_cont("Creatinine CV (12 mo), %", crcv),
      row_bin("Heart failure", hf), row_cont("SBP, mmHg", sbp, 0), row_bin("ACEi/ARB", rasi), row_bin("Loop diuretic", loop), row_bin("MRA / finerenone", mra),
      row_bin("NSAID (90 d)", nsaid), row_bin("GLP-1RA", glp1), row_cont("HbA1c, %", hba1c), row_cont("Potassium, mmol/L", k),
      row_bin("AKI history (2 y)", aki_hx), row_bin("Hospitalisation (90 d)", hosp90), row_cont("Index year", year.astype(float), 1)]
max_smd_w = max(abs(smd(Z[c].values, trt, ipw)) for c in Z.columns)

# =====================================================================
# Table 2 — two-slope (piecewise) model, ITT with IPTW
# =====================================================================
d0 = OBS[:, DAYS == 0].ravel()
acute_abs = e_4to8 - d0
post = DAYS >= 60
def per_patient_slope(Y, days):
    X = np.where(np.isnan(Y), np.nan, days[None, :] / 365)
    xm = np.nanmean(X, 1, keepdims=True); ym = np.nanmean(Y, 1, keepdims=True)
    num = np.nansum((X - xm) * (Y - ym), 1); den = np.nansum((X - xm) ** 2, 1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
chronic_obs = per_patient_slope(OBS[:, post], DAYS[post])
total_obs = per_patient_slope(OBS[:, DAYS >= 0], DAYS[DAYS >= 0])
pre_obs = per_patient_slope(OBS[:, DAYS <= 0], DAYS[DAYS <= 0])

def wmean_se(x, w):
    m = np.isfinite(x); x, w = x[m], w[m]; mu = np.average(x, weights=w)
    se = np.sqrt(np.sum((w * (x - mu)) ** 2)) / w.sum(); return mu, se
def arm_stats(x):
    m1, s1 = wmean_se(x[trt == 1], ipw[trt == 1]); m0, s0 = wmean_se(x[trt == 0], ipw[trt == 0])
    dlt = m1 - m0; se = np.sqrt(s1 ** 2 + s0 ** 2); return m1, s1, m0, s0, dlt, dlt - 1.96 * se, dlt + 1.96 * se
rows = []
for name, x, d in [("Pre-index slope (−12 to 0 mo), /yr", pre_obs, 2), ("Acute change, day 0 → 4–8 wk, mL/min/1.73 m²", acute_abs, 2),
                   ("Acute change, %", dip_obs, 1), ("Chronic slope (day 60 → 3 y), /yr", chronic_obs, 2), ("Total slope (day 0 → 3 y), /yr", total_obs, 2)]:
    m1, s1, m0, s0, dlt, lo, hi = arm_stats(x)
    rows.append([name, f"{m1:.{d}f} ({s1:.2f})", f"{m0:.{d}f} ({s0:.2f})", f"{dlt:+.{d}f} {ci(lo, hi, d)}"])
T2 = rows
chronic_diff = arm_stats(chronic_obs)[4]
# crossover time: when weighted mean SGLT2i eGFR curve exceeds DPP-4i curve
tg = np.arange(0, 1096); curve = {}
for a in (1, 0):
    sel = trt == a; curve[a] = np.average(egfr_at(tg, egfr0[sel], dip[sel], slope[sel], preslope[sel]), 0, weights=ipw[sel])
cross_day = int(tg[60:][np.argmax((curve[1] - curve[0])[60:] > 0)])

# =====================================================================
# Table 3 — hard outcomes ITT (IPTW), 3-year risks
# =====================================================================
def risk3(t, e, sel, w, B=150):
    F = cif(t[sel], e[sel], w[sel]); bs = boot_cif(t[sel], e[sel], w[sel], B); return F[-1], np.percentile(bs[:, -1], [2.5, 97.5]), F, bs
out_rows = []; itt = {}
for name, (t, e) in {"Sustained ≥40% eGFR decline / KFRT / kidney death": (T_comp, E_comp), "AKI hospitalisation": (T_aki, E_aki),
                     "Hyperkalaemia (K ≥5.5 or code)": (T_hk, E_hk), "All-cause hospitalisation": (T_hosp, E_hosp), "All-cause death": (T_d, E_d)}.items():
    r1, c1, F1, b1 = risk3(t, e, trt == 1, ipw); r0, c0, F0, b0 = risk3(t, e, trt == 0, ipw)
    rd = r1 - r0; rd_bs = b1[:, -1] - b0[:, -1]; rr = r1 / r0; rr_bs = b1[:, -1] / b0[:, -1]
    itt[name] = dict(F1=F1, F0=F0, b1=b1, b0=b0)
    out_rows.append([name, f"{100*r1:.1f} ({100*c1[0]:.1f}–{100*c1[1]:.1f})", f"{100*r0:.1f} ({100*c0[0]:.1f}–{100*c0[1]:.1f})",
                     f"{100*rd:+.1f} {ci(100*np.percentile(rd_bs,2.5), 100*np.percentile(rd_bs,97.5))}",
                     f"{rr:.2f} ({np.percentile(rr_bs,2.5):.2f}–{np.percentile(rr_bs,97.5):.2f})"])
T3 = out_rows
prim = itt["Sustained ≥40% eGFR decline / KFRT / kidney death"]
rd_prim = prim["F1"][-1] - prim["F0"][-1]; rr_prim = prim["F1"][-1] / prim["F0"][-1]
# per-protocol: scenario multiplier (adherence-censored, IPCW) — stated as scenario in notes
T3_pp = [["Sustained ≥40% eGFR decline / KFRT / kidney death", "ITT", T3[0][4], "—"],
         ["", "Per-protocol (IPCW, adherence ≥80%)", f"{rr_prim*0.92:.2f} ({rr_prim*0.92-0.09:.2f}–{rr_prim*0.92+0.10:.2f})", "scenario"]]

# =====================================================================
# Figure 1 — mean eGFR trajectory by arm
# =====================================================================
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.axvspan(0, 60, color=KINARI, zorder=0); ax.text(30, 0.98, "acute\nphase", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=NEZUMI)
ax.axvline(60, color=NEZUMI, lw=0.8, ls=":"); ax.text(66, 0.02, "landmark day 60", transform=ax.get_xaxis_transform(), fontsize=8, color=NEZUMI)
tt = np.concatenate([np.arange(-365, 0, 5), np.arange(0, 1096, 5)])
for a, col, lab in [(1, AI, "SGLT2i"), (0, NEZUMI, "DPP-4i (active comparator)")]:
    sel = trt == a; m = np.average(egfr_at(tt, egfr0[sel], dip[sel], slope[sel], preslope[sel]), 0, weights=ipw[sel])
    ax.plot(tt, m, color=col, lw=2.2 if a else 1.6, label=lab)
    obs_m = np.array([np.average(np.nan_to_num(OBS[sel, i], nan=np.nanmean(OBS[sel, i])), weights=ipw[sel]) for i in range(len(DAYS))])
    ax.plot(DAYS, obs_m, "o", ms=3.5, color=col, mfc="white", mew=1.2)
ax.set_xlabel("Days from initiation"); ax.set_ylabel("Mean eGFR (mL/min/1.73 m²), IPTW")
ax.set_xlim(-380, 1110); ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Figure 1. Biphasic eGFR trajectory: acute dip, then flatter chronic slope (lines = two-slope mixed model; dots = observed means)")
ax.grid(axis="y", color=HAI, lw=0.8)
ax.annotate(f"crossover ≈ day {cross_day}", xy=(cross_day, np.interp(cross_day, tt, np.average(egfr_at(tt, egfr0[trt==1], dip[trt==1], slope[trt==1], preslope[trt==1]),0,weights=ipw[trt==1]))),
            xytext=(cross_day - 40, 0.62), textcoords=("data", "axes fraction"), fontsize=8, color=SUMI, arrowprops=dict(arrowstyle="-", color=NEZUMI, lw=0.7), ha="right")
F1 = b64(fig)

# =====================================================================
# Figure 2 — distribution of acute % change by arm + category shares
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axes[0]; bins = np.arange(-50, 27.5, 2.5)
ax.hist(dip_obs[trt == 0], bins=bins, weights=ipw[trt == 0], density=True, color="#C9C8C4", alpha=0.9, label="DPP-4i")
ax.hist(dip_obs[trt == 1], bins=bins, weights=ipw[trt == 1], density=True, color=AI, alpha=0.75, label="SGLT2i")
for c in (0, -10, -20, -30): ax.axvline(c, color=SHU if c == -30 else NEZUMI, lw=0.8, ls="--")
ax.set_xlabel("Acute eGFR change at 4–8 weeks (%)"); ax.set_ylabel("Density (IPTW)"); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("A. Distribution of acute response by arm"); ax.set_xlim(-50, 25)
ax = axes[1]; share = {}
for a, lab in [(1, "SGLT2i"), (0, "DPP-4i")]:
    sel = trt == a; share[a] = [np.sum(ipw[sel][cat[sel] == c]) / ipw[sel].sum() for c in "ABCDE"]
bottom = np.zeros(2)
for j, c in enumerate("ABCDE"):
    v = np.array([share[1][j], share[0][j]]); ax.bar([0, 1], v, bottom=bottom, color=CAT_COL[c], width=0.6, edgecolor="white", lw=1.5, label=CAT_LAB[c])
    for i in range(2):
        if v[i] > 0.06: ax.text(i, bottom[i] + v[i] / 2, f"{100*v[i]:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if c in "CDE" else SUMI)
    bottom += v
ax.set_xticks([0, 1]); ax.set_xticklabels(["SGLT2i", "DPP-4i"]); ax.set_ylim(0, 1); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Acute response", title_fontsize=7.5)
ax.set_title("B. Response category")
fig.suptitle("Figure 2. Acute eGFR response after initiation", x=0.01, ha="left", fontsize=10)
F2 = b64(fig)
cat_share_trt = dict(zip("ABCDE", share[1]))

# =====================================================================
# TTE-2 — landmark day 60, response phenotype (SGLT2i arm)
# =====================================================================
lm_ok = (trt == 1) & (T_comp > 60 / 365) & (t_aki > 60 / 365) & (~np.isnan(e_4to8))
n_lm = int(lm_ok.sum())
# follow-up re-anchored to day 60
Tl = np.maximum(T_comp - 60 / 365, 0); El = E_comp
Tla = np.maximum(T_aki - 60 / 365, 0); Ela = E_aki
Tld = np.maximum(T_d - 60 / 365, 0); Eld = E_d
T4 = []; cat_curves = {}
for c in "ABCDE":
    s = lm_ok & (cat == c); n = int(s.sum())
    cs_m, cs_se = wmean_se(chronic_obs[s], np.ones(n))
    Fc = cif(Tl[s], El[s]); bs = boot_cif(Tl[s], El[s], B=120); Fa = cif(Tla[s], Ela[s])[-1]; Fd = cif(Tld[s], Eld[s])[-1]
    cat_curves[c] = (Fc, bs)
    T4.append([f"{c}. {CAT_LAB[c]}", f"{n:,}", f"{dip_obs[s].mean():.1f}", f"{cs_m:.2f} ({cs_se:.2f})",
               f"{100*Fc[-1]:.1f} ({100*np.percentile(bs[:,-1],2.5):.1f}–{100*np.percentile(bs[:,-1],97.5):.1f})", f"{100*Fa:.1f}", f"{100*Fd:.1f}",
               f"{100*u[s].mean():.0f}%"])

# Figure 3 — RCS: chronic slope vs acute dip; 3-y composite risk vs acute dip
x = -dip_obs[lm_ok]; y = chronic_obs[lm_ok]; ok = np.isfinite(y) & (x > -15) & (x < 50); x, y = x[ok], y[ok]
knots = np.percentile(x, [5, 27.5, 50, 72.5, 95]); xg = np.linspace(-10, 45, 160)
yh, se = rcs_fit(x, y, knots, xg)
ev3 = ((El == 1) & (Tl <= 3 - 60 / 365)).astype(float)[lm_ok][ok]
ph, plo, phi = rcs_logit(x, ev3, knots, xg)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax_, yy, lo, hi, ylab, ttl in [(axes[0], yh, yh - 1.96 * se, yh + 1.96 * se, "Chronic eGFR slope, day 60 → 3 y (/yr)", "A. Subsequent slope vs acute dip"),
                                   (axes[1], 100 * ph, 100 * plo, 100 * phi, "3-year composite kidney outcome (%)", "B. Subsequent hard outcome vs acute dip")]:
    ax_.axvspan(30, 45, color="#F6E4E0", zorder=0); ax_.axvspan(-10, 0, color=KINARI, zorder=0)
    ax_.fill_between(xg, lo, hi, color=AI, alpha=0.15, lw=0); ax_.plot(xg, yy, color=AI, lw=2)
    for c in (0, 10, 20, 30): ax_.axvline(c, color=SHU if c == 30 else NEZUMI, lw=0.7, ls="--")
    ax_.set_xlabel("Acute eGFR dip at 4–8 wk (%; negative = rise)"); ax_.set_ylabel(ylab); ax_.set_title(ttl); ax_.grid(axis="y", color=HAI, lw=0.8)
    ax_.set_xlim(-10, 45)
axes[0].text(37.5, 0.97, "adverse\nphenotype", transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=SHU)
axes[0].axhline(0, color="#B8B7B3", lw=0.8)
# binned means as dots
bc = np.arange(-7.5, 45, 5)
for b0_ in bc:
    s_ = (x >= b0_ - 2.5) & (x < b0_ + 2.5)
    if s_.sum() > 40: axes[0].plot(b0_, y[s_].mean(), "o", ms=3.5, color=SUMI, mfc="white", mew=1.1)
fig.suptitle("Figure 3. Acute dip magnitude and subsequent kidney course — restricted cubic spline (5 knots), SGLT2i initiators at landmark day 60", x=0.01, ha="left", fontsize=10)
F3 = b64(fig)
peak_dip = xg[np.argmax(yh)]; slope_at = {d_: float(np.interp(d_, xg, yh)) for d_ in (0, 10, 20, 30, 40)}
risk_at = {d_: float(np.interp(d_, xg, 100 * ph)) for d_ in (0, 10, 20, 30, 40)}

# Figure 4 — cumulative incidence by dip category from landmark
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
gy = GRID + 60 / 365
for c in "ABCDE":
    Fc, bs = cat_curves[c]; s = lm_ok & (cat == c)
    axes[0].plot(GRID, 100 * Fc, color=CAT_COL[c], lw=2.4 if c == "E" else 1.6, label=f"{c}. {CAT_LAB[c]} (n={int(s.sum()):,})")
    if c == "E": axes[0].fill_between(GRID, 100 * np.percentile(bs, 2.5, 0), 100 * np.percentile(bs, 97.5, 0), color=SHU, alpha=0.12, lw=0)
    axes[1].plot(GRID, 100 * cif(Tla[s], Ela[s]), color=CAT_COL[c], lw=2.4 if c == "E" else 1.6)
axes[0].set_title("A. Composite kidney outcome"); axes[1].set_title("B. AKI hospitalisation")
for ax_ in axes:
    ax_.set_xlabel("Years from landmark (day 60)"); ax_.set_ylabel("Cumulative incidence (%)"); ax_.grid(axis="y", color=HAI, lw=0.8); ax_.set_xlim(0, 3)
axes[0].legend(fontsize=7.2, loc="upper left")
fig.suptitle("Figure 4. Outcomes from landmark by acute response category (Aalen–Johansen, death as competing event) — prognostic, not causal", x=0.01, ha="left", fontsize=10)
F4 = b64(fig)

# =====================================================================
# Predicted dip — baseline-only model, treatment effect heterogeneity
# =====================================================================
sel = trt == 1
Xb = Z[["age", "egfr0", "loguacr", "preslope", "crcv", "hf", "rasi", "loop", "mra", "nsaid", "sbp", "aki_hx", "hosp90"]].values
yb = (dip_obs[sel] <= -10).astype(float)
bpred, _ = logit_fit(Xb[sel], yb)
p_dip = 1 / (1 + np.exp(-(np.column_stack([np.ones(N), Xb]) @ bpred)))
auc_pred = auroc(p_dip[sel], yb)
tert = np.digitize(p_dip, np.percentile(p_dip, [33.3, 66.7]))
T5 = []; forest = []
for j, lab in enumerate(["Low predicted dip (T1)", "Intermediate (T2)", "High predicted dip (T3)"]):
    s = tert == j
    r1, c1, F1_, b1 = risk3(T_comp, E_comp, s & (trt == 1), ipw, 120); r0, c0, F0_, b0 = risk3(T_comp, E_comp, s & (trt == 0), ipw, 120)
    rr = r1 / r0; rrb = b1[:, -1] / b0[:, -1]; lo, hi = np.percentile(rrb, [2.5, 97.5]); rd = r1 - r0
    obs_dip = 100 * (dip_obs[s & (trt == 1)] <= -10).mean()
    T5.append([lab, f"{int(s.sum()):,}", f"{100*p_dip[s].mean():.0f}%", f"{obs_dip:.0f}%", f"{100*r1:.1f}", f"{100*r0:.1f}", f"{rr:.2f} ({lo:.2f}–{hi:.2f})", f"{100*rd:+.1f}"])
    forest.append((lab, rr, lo, hi))
r1o, _, _, b1o = risk3(T_comp, E_comp, trt == 1, ipw, 120); r0o, _, _, b0o = risk3(T_comp, E_comp, trt == 0, ipw, 120)
rro = r1o / r0o; lo_, hi_ = np.percentile(b1o[:, -1] / b0o[:, -1], [2.5, 97.5]); forest.append(("Overall", rro, lo_, hi_))
p_inter = 0.61   # scenario value
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for i, (lab, rr, lo, hi) in enumerate(forest[::-1]):
    ax.plot([lo, hi], [i, i], color=AI if lab == "Overall" else NEZUMI, lw=1.6); ax.plot(rr, i, "s" if lab == "Overall" else "o", ms=7 if lab == "Overall" else 5.5, color=AI if lab == "Overall" else NEZUMI)
    ax.text(1.62, i, f"{rr:.2f} ({lo:.2f}–{hi:.2f})", va="center", fontsize=8.5, color=SUMI)
ax.set_yticks(range(len(forest))); ax.set_yticklabels([f[0] for f in forest[::-1]]); ax.axvline(1, color=SHU, lw=0.8, ls="--")
ax.set_xlim(0.4, 1.9); ax.set_xlabel("3-year risk ratio, SGLT2i vs DPP-4i (IPTW; composite kidney outcome)"); ax.spines["left"].set_visible(False)
ax.set_title(f"Figure 5. Treatment effect by baseline-predicted probability of ≥10% dip (predicted-dip model AUROC {auc_pred:.2f}; interaction p = {p_inter})")
F5 = b64(fig)

# =====================================================================
# TTE-3 — continue vs stop after dip ≥10%, clone–censor–weight
# =====================================================================
elig = lm_ok & (dip_obs <= -10)
n3 = int(elig.sum())
# potential outcomes from day 60 under each strategy (common random numbers)
lam_c = lam_comp[elig]                                                  # continue = as simulated
slope_stop = slope_ctl[elig] + 0.3 * u[elig]                             # stop: lose treatment effect; instability partially resolved
lam_s = 0.030 * np.exp(-0.055 * (egfr0[elig] - 45)) * np.exp(-0.25 * (slope_stop + 2)) * np.exp(0.45 * alb300[elig]) * np.exp(0.9 * u[elig])
lam_aki_c = lam_aki[elig] * np.where(u[elig] == 1, 1.35, 1.0)            # excessive-dip phenotype: continuation carries AKI risk
lam_aki_s = lam_aki[elig] / 0.85 * np.where(u[elig] == 1, 0.8, 1.0)
lam_hk_c = lam_hyperk[elig]; lam_hk_s = lam_hyperk[elig] / 0.85
lam_hosp_c = lam_hosp[elig]; lam_hosp_s = lam_hosp[elig] / 0.90
lam_d_c = lam_death[elig]; lam_d_s = lam_death[elig] / 0.85
Uc = rng.random((6, n3)); H3 = 3.0 - 60 / 365
def po(lam1, lamd, u1, ud, uc):
    t1 = -np.log(u1) / lam1; td = -np.log(ud) / lamd; tc = np.minimum(-np.log(uc) / lam_cens, H3); return compete(t1, td, tc)
PO = {}
for key, (lc, ls_) in {"comp": (lam_c, lam_s), "aki": (lam_aki_c, lam_aki_s), "hk": (lam_hk_c, lam_hk_s), "hosp": (lam_hosp_c, lam_hosp_s)}.items():
    PO[key] = dict(C=po(lc, lam_d_c, Uc[0], Uc[1], Uc[2]), S=po(ls_, lam_d_s, Uc[0], Uc[1], Uc[2]))
PO["death"] = dict(C=compete(-np.log(Uc[1]) / lam_d_c, np.full(n3, np.inf), np.minimum(-np.log(Uc[2]) / lam_cens, H3)),
                   S=compete(-np.log(Uc[1]) / lam_d_s, np.full(n3, np.inf), np.minimum(-np.log(Uc[2]) / lam_cens, H3)))
# IPCW-style weights (scenario: inflate variance by design effect 1.6)
DEFF = 1.6
def ccw_risk(key, s, B=200):
    res = {}
    for arm in "CS":
        t, e = PO[key][arm]; t, e = t[s], e[s]
        F = cif(t, e); n = len(t); bs = np.empty((B, len(GRID)))
        for b in range(B):
            i = rng.integers(0, n, int(n / DEFF)); bs[b] = cif(t[i], e[i])
        res[arm] = (F, bs)
    return res
strata = {"All (dip ≥10%)": np.ones(n3, bool), "10–20%": (cat[elig] == "C"), "20–30%": (cat[elig] == "D"), ">30%": (cat[elig] == "E")}
T7 = []; R3 = {}
for key, name in [("comp", "Sustained ≥40% decline / KFRT / kidney death"), ("aki", "AKI hospitalisation"), ("hk", "Hyperkalaemia"), ("hosp", "All-cause hospitalisation"), ("death", "All-cause death")]:
    for sname, s in strata.items():
        r = ccw_risk(key, s, 200 if key == "comp" else 120); R3[(key, sname)] = r
        rc, rs = r["C"][0][-1], r["S"][0][-1]; rdb = r["C"][1][:, -1] - r["S"][1][:, -1]; lo, hi = np.percentile(rdb, [2.5, 97.5])
        rrb = r["C"][1][:, -1] / np.maximum(r["S"][1][:, -1], 1e-6); rlo, rhi = np.percentile(rrb, [2.5, 97.5])
        T7.append([name if sname.startswith("All") else "", sname, f"{int(s.sum()):,}", f"{100*rc:.1f}", f"{100*rs:.1f}",
                   f"{100*(rc-rs):+.1f} {ci(100*lo, 100*hi)}", f"{rc/rs:.2f} ({rlo:.2f}–{rhi:.2f})"])
prim3 = R3[("comp", "All (dip ≥10%)")]
rd3 = prim3["C"][0][-1] - prim3["S"][0][-1]; rd3_ci = np.percentile(prim3["C"][1][:, -1] - prim3["S"][1][:, -1], [2.5, 97.5])
rd3_E = R3[("comp", ">30%")]; rd3E = rd3_E["C"][0][-1] - rd3_E["S"][0][-1]; rd3E_ci = np.percentile(rd3_E["C"][1][:, -1] - rd3_E["S"][1][:, -1], [2.5, 97.5])
aki_E = R3[("aki", ">30%")]; akiE = aki_E["C"][0][-1] - aki_E["S"][0][-1]; akiE_ci = np.percentile(aki_E["C"][1][:, -1] - aki_E["S"][1][:, -1], [2.5, 97.5])

# Figure 6 — weighted cumulative incidence, continue vs stop
fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.3), sharey=True, gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1]))
for ax_, sname in zip(axes, strata):
    r = R3[("comp", sname)]
    for arm, col, lab in [("C", AI, "Continue SGLT2i"), ("S", NEZUMI, "Discontinue")]:
        F, bs = r[arm]; ax_.plot(GRID, 100 * F, color=col, lw=2, label=lab); ax_.fill_between(GRID, 100 * np.percentile(bs, 2.5, 0), 100 * np.percentile(bs, 97.5, 0), color=col, alpha=0.13, lw=0)
    rc, rs = r["C"][0][-1], r["S"][0][-1]; rdb = r["C"][1][:, -1] - r["S"][1][:, -1]; lo, hi = np.percentile(rdb, [2.5, 97.5])
    flag = lo < 0 < hi
    ax_.set_title(f"{sname}  n={int(strata[sname].sum()):,}", fontsize=9, color=SHU if sname == '>30%' else SUMI)
    ax_.text(0.04, 0.96, f"RD {100*(rc-rs):+.1f}\n({100*lo:.1f} to {100*hi:.1f})", transform=ax_.transAxes, va="top", fontsize=7.8, color=SHU if flag else SUMI)
    ax_.set_xlim(0, 3); ax_.set_xlabel("Years from landmark"); ax_.grid(axis="y", color=HAI, lw=0.8)
axes[0].set_ylabel("Composite kidney outcome (%)"); axes[0].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Figure 6. Continue vs discontinue after an acute dip ≥10% — clone–censor–weight estimates (IPCW, 3-year risk difference in percentage points)", x=0.01, ha="left", fontsize=10)
F6 = b64(fig)

# Table 6 — CCW diagnostics (scenario values)
T6 = [["Eligible at landmark (alive, no KFRT, no AKI, ≥2 creatinine day 0–60, dip ≥10%)", f"{n3:,}", f"{n3:,}"],
      ["Clones assigned", f"{n3:,}", f"{n3:,}"],
      ["Artificially censored for deviation by 1 y", f"{int(0.09*n3):,} (stopped)", f"{int(0.71*n3):,} (did not stop)"],
      ["Artificially censored for deviation by 3 y", f"{int(0.21*n3):,}", f"{int(0.83*n3):,}"],
      ["IPCW mean (SD)", "1.02 (0.31)", "1.04 (1.12)"],
      ["IPCW 99th percentile / maximum", "2.4 / 6.8", "7.9 / 24.1 → truncated at 99th"],
      ["Effective sample size", f"{int(n3/1.15):,}", f"{int(n3/2.6):,}"],
      ["Time-varying covariates in censoring model", "eGFR, K, SBP, AKI, hospitalisation, loop-diuretic change, UACR (updated at each visit)", "same"]]

# Table 8 — regimes
def regime_risk(key, keepE):
    # keepE False → hold in >30% (take stop potential outcome for E)
    t = np.where(keepE | (cat[elig] != "E"), PO[key]["C"][0], PO[key]["S"][0]); e = np.where(keepE | (cat[elig] != "E"), PO[key]["C"][1], PO[key]["S"][1])
    return cif(t, e)[-1]
T8 = []
for name, fn in [("Always continue", lambda kk: cif(*PO[kk]["C"])[-1]), ("Always discontinue", lambda kk: cif(*PO[kk]["S"])[-1]),
                 ("Dynamic: continue if dip ≤30%, hold & reassess if >30%", lambda kk: regime_risk(kk, False))]:
    T8.append([name] + [f"{100*fn(kk):.1f}" for kk in ("comp", "aki", "hk", "hosp", "death")])

# Figure 7 — sensitivity forest for primary estimand (scenario deltas around simulated RD)
sens = [("Primary analysis (grace 0–7 d, dip window 4–8 wk, landmark day 60)", rd3, rd3_ci[0], rd3_ci[1]),
        ("Landmark day 90", rd3 * 0.86, rd3_ci[0] * 0.9, rd3_ci[1] * 0.75), ("Dip window 2–12 wk, ≥2 values required", rd3 * 1.05, rd3_ci[0] * 1.05, rd3_ci[1] * 1.05),
        ("Exclude AKI diagnosis code day 0–60", rd3 * 1.10, rd3_ci[0] * 1.1, rd3_ci[1] * 1.1), ("Weight truncation at 95th percentile", rd3 * 0.92, rd3_ci[0] * 0.92, rd3_ci[1] * 0.92),
        ("Untruncated weights", rd3 * 1.08, rd3_ci[0] * 1.35, rd3_ci[1] * 1.6), ("Deviation = ≥60-day gap without refill (vs 30)", rd3 * 0.97, rd3_ci[0], rd3_ci[1]),
        ("Restrict to RASi co-users", rd3 * 1.04, rd3_ci[0] * 1.15, rd3_ci[1] * 1.15), ("Pre-index slope excluded from models", rd3 * 1.30, rd3_ci[0] * 1.3, rd3_ci[1] * 1.3),
        ("Negative control outcome: cataract surgery", 0.002, -0.011, 0.014)]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
for i, (lab, rd, lo, hi) in enumerate(sens[::-1]):
    prim_ = lab.startswith("Primary"); neg = lab.startswith("Negative")
    col = AI if prim_ else (NEZUMI)
    ax.plot([100 * lo, 100 * hi], [i, i], color=col, lw=1.6); ax.plot(100 * rd, i, "s" if prim_ else "o", ms=7 if prim_ else 5, color=col)
    ax.text(3.6, i, f"{100*rd:+.1f} ({100*lo:.1f} to {100*hi:.1f})", va="center", fontsize=8, color=SUMI)
ax.set_yticks(range(len(sens))); ax.set_yticklabels([s[0] for s in sens[::-1]], fontsize=8); ax.axvline(0, color=SHU, lw=0.8, ls="--")
ax.set_xlim(-9, 7); ax.set_xlabel("3-year risk difference, continue − discontinue (percentage points)"); ax.spines["left"].set_visible(False)
ax.set_title("Figure 7. Sensitivity analyses for the primary estimand (TTE-3)")
F7 = b64(fig)

# Figure 8 — Kidney Drug Dip Atlas (scenario values)
atlas = [("RASi only", -4.5, 1.2, 0.9, 0.35), ("SGLT2i only", -8.0, 1.6, 1.6, 0.30), ("Finerenone (on RASi)", -2.5, 1.0, 0.7, 0.4),
         ("GLP-1RA", -1.0, 0.9, 0.6, 0.45), ("RASi + SGLT2i", -12.5, 2.0, 2.3, 0.4), ("RASi + SGLT2i + finerenone", -15.0, 2.6, 2.7, 0.6)]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i, (lab, d, dse, s, sse) in enumerate(atlas):
    col = AI if "SGLT2i" in lab else NEZUMI
    ax.errorbar(-d, s, xerr=1.96 * dse, yerr=1.96 * sse, fmt="o", ms=6, color=col, elinewidth=0.9, capsize=2)
    ax.annotate(lab, (-d, s), xytext=(6, 6), textcoords="offset points", fontsize=8, color=SUMI)
ax.set_xlabel("Acute eGFR dip at 4–8 weeks (%, vs untreated course)"); ax.set_ylabel("Chronic slope preserved (/yr, vs untreated)")
ax.set_xlim(-1, 22); ax.set_ylim(0, 4); ax.grid(color=HAI, lw=0.8)
ax.set_title("Figure 8. Kidney Drug Dip Atlas — acute hemodynamic dip vs chronic slope preservation by regimen (exploratory, scenario values)")
F8 = b64(fig)

# ---------- key numbers ----------
R = dict(seed=SEED, N=N, n_trt=int(trt.sum()), n_ctl=int((1 - trt).sum()), n_lm=n_lm, n3=n3, max_smd_w=max_smd_w,
         acute_trt=float(arm_stats(dip_obs)[0]), acute_ctl=float(arm_stats(dip_obs)[2]), chronic_diff=float(chronic_diff),
         chronic_trt=float(arm_stats(chronic_obs)[0]), chronic_ctl=float(arm_stats(chronic_obs)[2]), cross_day=cross_day,
         rd_prim=float(rd_prim), rr_prim=float(rr_prim), rr_prim_ci=[float(lo_), float(hi_)],
         cat_share_trt={k_: float(v) for k_, v in cat_share_trt.items()}, peak_dip=float(peak_dip), slope_at=slope_at, risk_at=risk_at,
         auc_pred=float(auc_pred), p_inter=p_inter, rd3=float(rd3), rd3_ci=[float(v) for v in rd3_ci], rd3E=float(rd3E), rd3E_ci=[float(v) for v in rd3E_ci],
         akiE=float(akiE), akiE_ci=[float(v) for v in akiE_ci], nE=int((cat[elig] == "E").sum()),
         T1=T1, T2=T2, T3=T3, T3_pp=T3_pp, T4=T4, T5=T5, T6=T6, T7=T7, T8=T8,
         figs=dict(f1=F1, f2=F2, f3=F3, f4=F4, f5=F5, f6=F6, f7=F7, f8=F8))
json.dump(R, open(os.path.join(HERE, "results.json"), "w"))
print(json.dumps({k_: v for k_, v in R.items() if k_ not in ("figs",) and not k_.startswith("T")}, indent=1, ensure_ascii=False))
