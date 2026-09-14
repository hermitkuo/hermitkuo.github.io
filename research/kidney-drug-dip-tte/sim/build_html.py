"""Render results.json → ../index.html (single file, base64 figures)."""
import json, os, html as H
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
F = R["figs"]

def table(headers, rows, caption, note="", cls="", rowclass=None):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for r in rows:
        rc = f' class="{rowclass(r)}"' if rowclass and rowclass(r) else ""
        trs += f"<tr{rc}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    n = f'<p class="note">{note}</p>' if note else ""
    return f'<figure class="tbl {cls}"><figcaption>{caption}</figcaption><div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>{n}</figure>'

def fig(key, note=""):
    n = f'<p class="note">{note}</p>' if note else ""
    return f'<figure class="img"><img src="data:image/png;base64,{F[key]}" alt="">{n}</figure>'

pct = lambda v, d=1: f"{100*v:.{d}f}"
T1 = table(["Characteristic", "SGLT2i", "DPP-4i", "SMD crude", "SMD IPTW"], R["T1"],
           "Table 1. Baseline characteristics of new users by treatment strategy, before and after inverse-probability-of-treatment weighting",
           f"Mean (SD) unless stated. Pre-index eGFR slope and creatinine variability are computed from the 12 months before index and are mandatory confounders: a 10% dip from a declining trajectory and from a stable one are biologically different events. Maximum weighted SMD {R['max_smd_w']:.3f}.")
T2 = table(["Estimand (IPTW, two-slope mixed model)", "SGLT2i, mean (SE)", "DPP-4i, mean (SE)", "Difference (95% CI)"], R["T2"],
           "Table 2. Acute and chronic eGFR response — piecewise linear mixed model with a knot at day 60",
           "Acute change = mean of eGFR values at weeks 4–8 minus index eGFR; chronic slope = patient-level slope from day 60 to 3 years; pre-index slope shown as a balance check. Model: eGFR(t) = β₀ + β_acute·t·1(t≤60 d) + β_chronic·(t−60)·1(t>60 d), random intercept and slopes.")
T3 = table(["Outcome (3-year cumulative incidence, %)", "SGLT2i (95% CI)", "DPP-4i (95% CI)", "Risk difference, pp (95% CI)", "Risk ratio (95% CI)"], R["T3"],
           "Table 3. Intention-to-treat hard outcomes, IPTW Aalen–Johansen with death as competing event",
           "Grace period 0–7 days; time zero = first prescription. Per-protocol (IPCW for adherence) is scenario-valued in this simulation: composite RR " + R["T3_pp"][1][2] + ".", "primary-row-1")
T4 = table(["Acute response category", "n at landmark", "Mean dip, %", "Chronic slope /yr (SE)", "3-y composite, % (95% CI)", "3-y AKI, %", "3-y death, %", "Latent instability*"], R["T4"],
           "Table 4. Outcomes from landmark day 60 by acute response category — SGLT2i initiators (prognostic / response-phenotype analysis)",
           "*Simulation-only column: proportion of patients carrying the latent hemodynamic-instability variable that generates the excessive-dip phenotype; it is invisible in real data, which is exactly why this table must not be read causally. Category E is flagged as the adverse phenotype.",
           rowclass=lambda r: "alert" if r[0].startswith("E.") else "")
T5 = table(["Predicted-dip tertile (baseline-only model)", "n", "Mean P(dip ≥10%)", "Observed dip ≥10%, SGLT2i", "3-y composite SGLT2i, %", "DPP-4i, %", "Risk ratio (95% CI)", "RD, pp"], R["T5"],
           "Table 5. Treatment-effect heterogeneity by baseline-predicted probability of an acute dip ≥10%",
           f"Predicted dip is a pre-treatment characteristic, so this comparison is a valid causal contrast, unlike Table 4. Predicted-dip model AUROC {R['auc_pred']:.2f}; interaction p = {R['p_inter']} (scenario). Benefit is present in every tertile, with no evidence that a larger expected dip is required for protection.")
T6 = table(["Clone–censor–weight diagnostic", "Continue strategy", "Discontinue strategy"], R["T6"],
           "Table 6. Clone–censor–weight diagnostics for TTE-3 (scenario values)", cls="text", note=
           "Deviation = a refill gap ≥30 days (continue arm) or any refill after landmark (discontinue arm). Weights are truncated at the 99th percentile; effective sample size drives the width of the intervals in Figure 6.")
T7 = table(["Outcome", "Stratum (acute dip)", "n", "Continue, %", "Discontinue, %", "Risk difference, pp (95% CI)", "Risk ratio (95% CI)"], R["T7"],
           "Table 7. Three-year risks under continuation vs discontinuation after an acute dip ≥10% — overall and by dip stratum",
           "IPCW-weighted Aalen–Johansen estimates from landmark day 60; intervals from a clustered bootstrap with the design effect of the weights. Red rows: the >30% stratum, where the primary contrast is uninformative and AKI trends against continuation.",
           rowclass=lambda r: "alert" if r[1] == ">30%" else "")
T8 = table(["Strategy from landmark day 60", "Composite kidney, %", "AKI hosp., %", "Hyperkalaemia, %", "Hospitalisation, %", "Death, %"], R["T8"],
           "Table 8. Static vs dynamic treatment regimes — 3-year risks (IPCW, all patients with dip ≥10%)",
           "The dynamic regime holds therapy only in the >30% stratum (4% of initiators); it keeps essentially all of the kidney benefit of always continuing while lowering AKI admissions. This is the decision rule the study is designed to inform.",
           rowclass=lambda r: "hl" if r[0].startswith("Dynamic") else "")
BIAS = table(["Threat", "Where it enters", "Design defence in this protocol"], [
    ["Conditioning on a post-treatment variable", "Grouping by observed dip and comparing outcomes from day 0", "Dip categories are used only in TTE-2, explicitly labelled prognostic; the causal contrast in TTE-3 is a treatment strategy, not the dip"],
    ["Collider / selection bias", "Dip → discontinuation → who remains on drug", "Cloning at landmark; censoring at deviation; IPCW with time-varying eGFR, K, AKI, SBP"],
    ["Immortal time bias", "Classifying patients by ever-stopping after day 60", "Follow-up starts at landmark for both clones; deviation censors, it never re-classifies"],
    ["Informative measurement", "Sicker patients get more creatinine tests, so dips are found more often", "Eligibility requires ≥2 creatinine values day 0–60 for everyone; sensitivity: fixed 4–8 wk window only"],
    ["Regression to the mean", "Index eGFR sampled high by chance → apparent dip", "Pre-index slope and creatinine CV as confounders; index eGFR = mean of last 2 values; sensitivity excluding index within 30 d of hospitalisation"],
    ["AKI misclassified as dip", "Concurrent volume depletion, NSAID, contrast", "AKI codes / creatinine ≥1.5× within day 0–60 excluded from landmark set; sensitivity includes them"],
    ["Unmeasured confounding", "Frailty, volume status", "Negative control outcome (cataract surgery); E-value; active comparator with the same indication"]],
    "Table 9. Bias audit — why the naïve 'dip vs no-dip Cox model' is not run", cls="text")

svg = """<svg viewBox="0 0 900 330" role="img" aria-label="Protocol timeline with three target-trial layers" style="width:100%;height:auto;display:block">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6F6F6D"/></marker></defs>
<rect x="0" y="0" width="900" height="330" fill="#fff"/>
<rect x="300" y="58" width="90" height="14" fill="#FBFAF7" stroke="#E8E7E3"/>
<line x1="40" y1="65" x2="880" y2="65" stroke="#2B2B2B" stroke-width="1.2" marker-end="url(#ar)"/>
<g font-family="IBM Plex Sans, Noto Sans TC, sans-serif" font-size="11.5" fill="#2B2B2B">
<line x1="60" y1="58" x2="60" y2="72" stroke="#2B2B2B"/><text x="60" y="88" text-anchor="middle">−12 mo</text>
<text x="180" y="52" text-anchor="middle" fill="#6F6F6D">pre-index slope · creatinine CV</text>
<line x1="300" y1="52" x2="300" y2="78" stroke="#1C3F60" stroke-width="2"/><text x="300" y="92" text-anchor="middle" font-weight="600" fill="#1C3F60">Day 0 · index</text>
<text x="345" y="52" text-anchor="middle" fill="#6F6F6D">acute 0–60 d</text>
<line x1="390" y1="46" x2="390" y2="84" stroke="#C2503A" stroke-width="2"/><text x="390" y="98" text-anchor="middle" font-weight="600" fill="#C2503A">Landmark day 60</text>
<text x="640" y="52" text-anchor="middle" fill="#6F6F6D">chronic phase · day 60 → year 3</text>
<line x1="860" y1="58" x2="860" y2="72" stroke="#2B2B2B"/><text x="860" y="88" text-anchor="middle">Year 3</text>
<text x="330" y="118" font-size="10.5" fill="#6F6F6D">eGFR₀ → d14 → d30 → d45 → d60 : acute response = (eGFR₄₋₈wk − eGFR₀)/eGFR₀</text>
<!-- TTE-1 -->
<text x="40" y="160" font-weight="600" fill="#1C3F60">TTE-1</text><text x="40" y="176" font-size="10.5" fill="#6F6F6D">drug → biphasic course</text>
<line x1="300" y1="156" x2="850" y2="156" stroke="#1C3F60" stroke-width="2.5"/><text x="310" y="150" font-size="10.5" fill="#1C3F60">SGLT2i initiation</text>
<line x1="300" y1="172" x2="850" y2="172" stroke="#8A8A88" stroke-width="2.5"/><text x="310" y="186" font-size="10.5" fill="#6F6F6D">DPP-4i initiation (active comparator) — ITT + per-protocol</text>
<!-- TTE-2 -->
<text x="40" y="222" font-weight="600" fill="#1C3F60">TTE-2</text><text x="40" y="238" font-size="10.5" fill="#6F6F6D">response phenotype</text>
<rect x="300" y="210" width="90" height="22" fill="#FBFAF7" stroke="#D6D5D0"/><text x="345" y="225" text-anchor="middle" font-size="10.5">dip A–E</text>
<line x1="390" y1="221" x2="850" y2="221" stroke="#5F80A0" stroke-width="2.5"/><text x="400" y="240" font-size="10.5" fill="#6F6F6D">spline: dip magnitude → chronic slope / hard outcomes (prognostic)</text>
<!-- TTE-3 -->
<text x="40" y="284" font-weight="600" fill="#1C3F60">TTE-3</text><text x="40" y="300" font-size="10.5" fill="#6F6F6D">continue vs stop · CCW</text>
<rect x="300" y="268" width="90" height="22" fill="#F6E4E0" stroke="#C2503A"/><text x="345" y="283" text-anchor="middle" font-size="10.5" fill="#C2503A">dip ≥10%</text>
<line x1="390" y1="279" x2="410" y2="272" stroke="#2B2B2B"/><line x1="390" y1="279" x2="410" y2="294" stroke="#2B2B2B"/>
<line x1="410" y1="272" x2="850" y2="272" stroke="#1C3F60" stroke-width="2.5"/><text x="420" y="266" font-size="10.5" fill="#1C3F60">clone A · continue</text>
<line x1="410" y1="294" x2="850" y2="294" stroke="#8A8A88" stroke-width="2.5"/><text x="420" y="310" font-size="10.5" fill="#6F6F6D">clone B · discontinue — censor at deviation, inverse-probability-of-censoring weight</text>
</g></svg>"""

rd3, lo3, hi3 = 100 * R["rd3"], 100 * R["rd3_ci"][0], 100 * R["rd3_ci"][1]
rdE, loE, hiE = 100 * R["rd3E"], 100 * R["rd3E_ci"][0], 100 * R["rd3E_ci"][1]
akiE, aloE, ahiE = 100 * R["akiE"], 100 * R["akiE_ci"][0], 100 * R["akiE_ci"][1]
sa, ra = R["slope_at"], R["risk_at"]
cs = R["cat_share_trt"]

html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kidney Drug Dip Atlas</title>
<meta name="description" content="Simulated results page for a three-layer target trial emulation of the acute eGFR dip after kidney-protective drug initiation.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--ai:#1C3F60;--shu:#C2503A;--sumi:#2B2B2B;--nezumi:#6F6F6D;--hai:#E8E7E3;--kinari:#FBFAF7;--rule:#D6D5D0;--shu-bg:#F9ECE9}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth;color-scheme:light}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
body{{margin:0;background:#fff;color:var(--sumi);font-family:"IBM Plex Sans","Noto Sans TC",system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.65}}
.wrap{{max-width:900px;margin:0 auto;padding-block:0 96px;padding-inline:24px}}
header{{padding:56px 0 28px;border-bottom:2px solid var(--ai)}}
.sim{{display:inline-block;border:1px solid var(--shu);color:var(--shu);padding:2px 10px;font-size:12.5px;font-weight:500;margin-bottom:22px;letter-spacing:.02em}}
h1{{font-size:28px;line-height:1.3;font-weight:600;margin:0 0 10px;color:var(--ai);letter-spacing:-.01em;text-wrap:balance}}
h1 span{{display:block;font-size:16px;font-weight:400;color:var(--nezumi);margin-top:6px;letter-spacing:0}}
.meta{{font-size:13.5px;color:var(--nezumi);margin:0}}
nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--rule);padding:10px 0;margin-bottom:8px;z-index:2}}
nav .row{{overflow-x:auto;white-space:nowrap;scrollbar-width:none}}
nav a{{font-size:13px;color:var(--nezumi);text-decoration:none;margin-right:18px;white-space:nowrap}}
nav a:hover,nav a:focus-visible{{color:var(--ai);outline:none;text-decoration:underline}}
.key{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--rule);margin:26px 0 8px}}
.key div{{padding:14px 12px 16px 0;border-right:1px solid var(--rule)}}
.key div:last-child{{border-right:0}}
.key b{{display:block;font-size:26px;font-weight:600;color:var(--ai);line-height:1.1;font-variant-numeric:tabular-nums}}
.key b.warn{{color:var(--shu)}}
.key small{{color:var(--nezumi);font-size:12.5px;line-height:1.35;display:block;margin-top:6px}}
section{{padding-top:44px}}
.layer{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--nezumi);margin:0 0 4px}}
h2{{font-size:19px;font-weight:600;color:var(--ai);margin:0 0 6px;text-wrap:balance}}
h2 small{{font-weight:400;color:var(--nezumi);font-size:13px;margin-left:10px}}
h3{{font-size:15px;font-weight:600;color:var(--sumi);margin:26px 0 4px}}
.lead{{margin:0 0 18px;max-width:72ch}}
.lead em{{font-style:normal;background:linear-gradient(transparent 62%,rgba(28,63,96,.14) 62%)}}
figure{{margin:22px 0}}
figure.img img{{width:100%;height:auto;display:block;border:1px solid var(--hai)}}
figcaption{{font-size:13.5px;font-weight:500;color:var(--sumi);margin-bottom:8px}}
.note{{font-size:12.5px;color:var(--nezumi);margin:8px 0 0;max-width:80ch;line-height:1.5}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-weight:500;color:var(--nezumi);border-top:1px solid var(--ai);border-bottom:1px solid var(--rule);padding:8px 10px 6px;white-space:nowrap}}
td{{padding:7px 10px;border-bottom:1px solid var(--hai);vertical-align:top;white-space:nowrap}}
td:first-child{{white-space:normal;min-width:16ch}}
tbody tr:last-child td{{border-bottom:1px solid var(--rule)}}
.primary-row-1 tbody tr:first-child td{{background:var(--kinari);font-weight:500;color:var(--ai)}}
tr.alert td{{color:var(--shu);background:var(--shu-bg)}}
.text td{{white-space:normal;min-width:0}}
tr.hl td{{background:var(--kinari);font-weight:500;color:var(--ai)}}
.schema{{border:1px solid var(--hai);padding:8px;margin:18px 0 6px}}
.estimand{{border-left:3px solid var(--ai);padding:4px 0 4px 18px;margin:18px 0}}
.estimand p{{margin:6px 0}}
.estimand code{{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--kinari);padding:1px 5px}}
.limits{{background:var(--kinari);padding:18px 22px;border:1px solid var(--hai);font-size:13.5px}}
.limits ul{{margin:8px 0 0;padding-left:18px}}
.dont{{border:1px solid var(--shu);padding:14px 20px;font-size:13.5px;margin:18px 0}}
.dont p{{margin:0 0 6px;color:var(--shu);font-weight:500}}
.dont ul{{margin:0;padding-left:18px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
footer{{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--nezumi)}}
@media(max-width:640px){{.key{{grid-template-columns:1fr 1fr}}.key div:nth-child(2){{border-right:0}}.two{{grid-template-columns:1fr}}h1{{font-size:23px}}header{{padding-top:36px}}}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <span class="sim">Simulated results — not real patient data</span>
  <h1>Acute eGFR Dip After Kidney-Protective Drug Initiation: A Three-Layer Target Trial Emulation
  <span>Kidney Drug Dip Atlas · 模擬結果報告，用於凍結 figure / table 架構與 estimand，真實分析僅替換資料層</span></h1>
  <p class="meta">CKD G2–G4 with T2DM and UACR ≥30 mg/g · new users of SGLT2i vs DPP-4i, 2018–2024 · landmark day 60 · follow-up 3 years · Generated 2026-09-14, seed {R['seed']}</p>
</header>

<nav><div class="row">
<a href="#protocol">Protocol</a><a href="#tte1">TTE-1 Drug → dip → protection</a><a href="#tte2">TTE-2 Dip phenotype</a><a href="#pred">Predicted dip</a><a href="#tte3">TTE-3 Continue vs stop</a><a href="#atlas">Dip Atlas</a><a href="#bias">Bias audit</a><a href="#plan">Analysis plan</a><a href="#notes">Simulation notes</a>
</div></nav>

<div class="key">
  <div><b>{R['N']:,}</b><small>new users, SGLT2i {R['n_trt']:,} vs DPP-4i {R['n_ctl']:,} (TTE-1)</small></div>
  <div><b>{R['acute_trt']:.1f}%</b><small>acute eGFR change at 4–8 wk on SGLT2i (DPP-4i {R['acute_ctl']:.1f}%); chronic slope {R['chronic_diff']:+.2f} mL/min/1.73 m²/yr vs comparator</small></div>
  <div><b>{rd3:+.1f} pp</b><small>primary estimand: 3-year risk difference of ≥40% decline/KFRT, continue vs discontinue after dip ≥10% ({lo3:.1f} to {hi3:.1f}), n = {R['n3']:,}</small></div>
  <div><b class="warn">{rdE:+.1f} pp</b><small>same contrast in the &gt;30% dip stratum ({loE:.1f} to {hiE:.1f}), n = {R['nE']}: uninformative, AKI {akiE:+.1f} pp against continuation</small></div>
</div>

<section id="protocol">
<p class="layer">Protocol</p>
<h2>三層 target trial：一個 index、一個 landmark、三個 estimand</h2>
<p class="lead">研究不驗證 “the greater the dip, the greater the protection”。三層設計分別回答：藥物是否造成 <em>acute loss + chronic preservation</em> 的雙相軌跡（TTE-1）；dip 幅度到哪裡仍是 pharmacodynamic response、從哪裡開始變成 adverse phenotype（TTE-2，明確標示為 prognostic）；以及臨床真正的決策——看到 creatinine 上升後，<em>continue 還是 stop</em>（TTE-3，clone–censor–weight）。</p>
<div class="schema">{svg}</div>
<div class="estimand">
<p><strong>Primary estimand（TTE-3）</strong>　3-year risk difference of sustained ≥40% eGFR decline / KFRT / kidney death, <code>continue</code> vs <code>discontinue</code> SGLT2i, among patients with an acute eGFR dip ≥10% who are alive, free of KFRT and AKI at landmark day 60.</p>
<p><strong>Secondary</strong>　chronic eGFR slope from day 60 onward（TTE-1, TTE-3）；AKI、hyperkalaemia、hospitalisation、death。</p>
<p><strong>Exploratory</strong>　restricted cubic spline of chronic outcome on acute dip magnitude（TTE-2）；treatment × predicted-dip interaction；dynamic regime 「continue if dip ≤30%, hold if &gt;30%」。</p>
</div>
</section>

<section id="tte1">
<p class="layer">TTE-1 · Drug → dip → long-term protection</p>
<h2>藥物造成雙相軌跡：acute dip 之後 chronic slope 變平<small>Tables 1–3 · Figures 1–2</small></h2>
<p class="lead">這是最接近真正 target trial 的一層。Time zero 為第一次處方，grace period 0–7 天，IPTW 後所有 baseline 變項 SMD &lt; 0.01，含 <em>pre-index eGFR slope 與 creatinine variability</em>——沒有這兩項，任何 dip 分析都無法區分「藥物反應」與「原本就在下降」。</p>
{T1}
{T2}
{fig('f1', f'SGLT2i arm loses about {abs(R["acute_trt"]):.0f}% in the first two weeks and then declines more slowly; the two mean curves cross at roughly day {R["cross_day"]} (month {R["cross_day"]/30.4:.0f}). Crossover time is the single most useful number for counselling a patient who asks why the creatinine went up.')}
{fig('f2', f'On SGLT2i, {pct(cs["A"],0)}% show no dip, {pct(cs["B"],0)}% a 0–10% dip, {pct(cs["C"],0)}% 10–20%, {pct(cs["D"],0)}% 20–30% and only {pct(cs["E"],1)}% exceed 30%. The comparator distribution is centred near zero with a thinner left tail, consistent with EMPA-REG / CREDENCE reports that a >30% dip is uncommon and merits re-evaluation rather than reassurance.')}
{T3}
</section>

<section id="tte2">
<p class="layer">TTE-2 · Landmark day 60 · Response phenotype</p>
<h2>多少 dip 是 physiological，從哪裡開始 pathological<small>Figure 3 (thesis figure) · Table 4 · Figure 4</small></h2>
<p class="lead">只有在 day 0–60 存活、未 KFRT、無 AKI、且有 ≥2 次 creatinine 的 SGLT2i 使用者（n = {R['n_lm']:,}）進入這一層；follow-up 從 day 60 重新起算。Observed dip 是 post-treatment variable，因此本層的輸出是 <em>response-phenotype prognosis，不是 causal effect</em>。以 5-knot restricted cubic spline 取代任意 10% cut-off。</p>
{fig('f3', f'Predicted chronic slope is flat from a small rise through a 20% dip (≈ {sa["0"]:.1f}, {sa["10"]:.1f}, {sa["20"]:.1f} mL/min/1.73 m²/yr at 0 / 10 / 20%), then turns down beyond ~25% ({sa["30"]:.1f} at 30%, {sa["40"]:.1f} at 40%). The 3-year composite risk follows the same shape ({ra["10"]:.0f}% at a 10% dip vs {ra["30"]:.0f}% at 30%). There is no dose–response of protection with dip magnitude; there is a threshold beyond which the dip is no longer physiological.')}
{T4}
{fig('f4', 'Categories A–D separate little for the hard outcome over three years; category E (>30%) diverges early for both the kidney composite and AKI admissions. Note that "no dip" (A) fares slightly worse than a mild dip (B): an absent hemodynamic response may itself mark non-response, which is a hypothesis for the predicted-dip analysis, not a conclusion.')}
</section>

<section id="pred">
<p class="layer">Predicted dip · Baseline-defined effect modification</p>
<h2>預期會大幅 dip 的患者，treatment effect 是否比較大？<small>Table 5 · Figure 5</small></h2>
<p class="lead">Predicted dip 只用 baseline 變項（eGFR、UACR、pre-index slope、creatinine CV、loop diuretic、RASi、SBP、HF、AKI history）建立 <code style="font-size:13px">P(dip ≥10% | X₀)</code>，是 pre-treatment characteristic，因此 treatment × predicted-dip 是合法的 causal heterogeneity 問題。模擬預設答案與 2025 empagliflozin IPD meta-analysis 一致：<em>各 tertile 皆有保護，沒有「越預期 dip 越保護」的梯度</em>——dip 是 response marker，不是 protection 的前提。</p>
{T5}
{fig('f5')}
</section>

<section id="tte3">
<p class="layer">TTE-3 · Landmark day 60 · Continue vs discontinue · Clone–censor–weight</p>
<h2>看到 creatinine 上升，應不應該停藥<small>Table 6 · Figure 6 · Tables 7–8 · Figure 7</small></h2>
<p class="lead">符合條件者（SGLT2i、dip ≥10%、landmark 存活且無 KFRT / AKI；n = {R['n3']:,}）在 day 60 各複製為兩個 clone，分別指派 <em>continue</em> 與 <em>discontinue</em>；clone 在偏離指派策略時被 artificially censored，再以含 time-varying eGFR、K、SBP、AKI、住院的 censoring model 計算 IPCW。真實世界的 stop → restart → stop 序列因此不會產生 immortal time 或 discontinuation bias。</p>
{T6}
{fig('f6', f'Overall, continuation lowers the 3-year kidney composite by {abs(rd3):.1f} percentage points ({lo3:.1f} to {hi3:.1f}). The benefit is present in the 10–20% and 20–30% strata. In the >30% stratum the estimate is {rdE:+.1f} ({loE:.1f} to {hiE:.1f}) with n = {R["nE"]}: the study cannot support continuing therapy unconditionally there, and AKI admissions trend against continuation.')}
{T7}
{T8}
{fig('f7', 'The primary estimand is stable across grace-period, dip-window, AKI-exclusion and weight-truncation choices. Two honest signals: dropping the pre-index slope from the models exaggerates the benefit (confounding by trajectory), and untruncated weights widen the interval substantially. The negative control outcome sits on the null.')}
</section>

<section id="atlas">
<p class="layer">Exploratory · Paper 2 preview</p>
<h2>Kidney Drug Dip Atlas：跨藥物的 acute dip 與 chronic preservation<small>Figure 8</small></h2>
<p class="lead">同一套 index / landmark / CCW 骨架可直接套用到 RASi、finerenone、GLP-1RA 與其組合。跨 regimen 的問題不再是「哪個藥比較好」，而是 <em>多藥 GDMT 疊加的 hemodynamic dip 是否仍與 chronic slope preservation 同向</em>，以及組合治療的 dip 是否需要不同的 hold 門檻。</p>
{fig('f8', 'Scenario values only. The atlas asks whether combined regimens move along the same dip–preservation line as single agents (additive hemodynamics) or fall below it (excess hemodynamic cost). Each point would come from its own TTE-1 with the same eligibility and comparator logic.')}
</section>

<section id="bias">
<p class="layer">Methods</p>
<h2>Bias audit<small>Table 9</small></h2>
<p class="lead">這題與一般 propensity-score 研究最大的差別在於 exposure（dip）本身是治療後事件，且 dip 又改變後續用藥行為。以下每一列都對應 protocol 中一個具體的設計決定。</p>
{BIAS}
<div class="dont">
<p>Pre-specified: analyses this protocol does not run</p>
<ul>
<li>Baseline PS-matching on dip vs no-dip followed by a Cox model from day 0.</li>
<li>Follow-up from day 0 with groups defined by week-4 dip (immortal time).</li>
<li>A single baseline creatinine and a single follow-up creatinine to define the dip.</li>
<li>Reading a correlation between larger dip and slower subsequent slope as mediation or as "the dip causes protection".</li>
</ul>
</div>
</section>

<section id="plan">
<p class="layer">Statistical analysis plan</p>
<h2>Pre-specified estimands and models</h2>
<div class="estimand">
<p><strong>TTE-1</strong>　IPTW (stabilised ATE weights, truncated 0.5–99.5th percentile) · piecewise linear mixed model with knot at day 60, random intercept + two random slopes · Aalen–Johansen cumulative incidence with death as competing event · ITT primary, per-protocol with IPCW for adherence secondary · 95% CI by patient-clustered bootstrap (2,000 resamples).</p>
<p><strong>TTE-2</strong>　Landmark day 60 · exposure = acute % change (continuous, RCS 5 knots at 5/27.5/50/72.5/95th percentiles; categories A–E for tables only) · outcomes: chronic slope (linear mixed model), composite / AKI / death (Aalen–Johansen; Fine–Gray for subdistribution HR) · adjusted for the full Table 1 set plus day-60 eGFR, K, SBP · reported as prognostic association.</p>
<p><strong>Predicted dip</strong>　logistic model for P(dip ≥10%) on baseline variables, internally validated (bootstrap optimism) · tertiles · treatment × tertile interaction in the TTE-1 weighted model · same for continuous predicted probability.</p>
<p><strong>TTE-3</strong>　clone–censor–weight · deviation definitions: continue = refill gap ≥30 d; discontinue = any refill · pooled logistic censoring model with time-varying eGFR, K, SBP, AKI, hospitalisation, loop-diuretic change, UACR · weights truncated at 99th percentile · weighted Aalen–Johansen; 3-year risk difference primary · strata 10–20 / 20–30 / &gt;30% · dynamic regime 「continue if dip ≤30%, hold if &gt;30%」 via the same clones.</p>
<p><strong>Sensitivity</strong>　landmark day 90 · dip window 2–12 wk with ≥2 values · exclude AKI code / creatinine ≥1.5× day 0–60 · weight truncation 95th / none · deviation gap 60 d · RASi co-users only · omit pre-index slope (to display its confounding role) · negative control outcome: cataract surgery · E-value for the primary estimand.</p>
<p><strong>Reporting</strong>　target trial specification table (Hernán) · STROBE · ESS and weight distribution for every weighted analysis · all code and the simulation seed archived with the protocol.</p>
</div>
</section>

<section id="notes">
<p class="layer">Simulation notes</p>
<h2>模擬假設與情境值</h2>
<div class="limits">
本頁所有數值由參數化模擬產生（seed {R['seed']}，<code>sim/simulate.py</code>），目的為凍結 manuscript 的 figure / table 規格與 estimand，不代表任何真實結果。模擬假設：
<ul>
<li>Treatment assignment 與 baseline 變項（year、eGFR、age、HF、albuminuria、AKI history、hospitalisation）相關；IPTW 以 logistic regression 於 18 個 baseline 變項估計，Table 1 的 SMD 為模擬資料實算。</li>
<li>eGFR 軌跡 = index eGFR × (1 + dip × exponential approach, τ = 6 d) + chronic slope × t，測量雜訊 SD 2.3，22% 隨機缺測。Dip 的均值取決於 loop diuretic、RASi、SBP、HF、baseline eGFR、NSAID、AKI history；chronic slope 取決於 pre-index slope、albuminuria 與 treatment。</li>
<li>一個 <strong>latent hemodynamic-instability</strong> 變項（盛行率約 7%）同時產生 excessive dip、更陡的 chronic slope 與較高的 AKI / death hazard；它不在 PS model 內，是 Figure 3 下彎與 Table 4 / 7 紅色列的來源。這是刻意植入的誠實負面發現：真實資料中此變項不可觀測，因此 TTE-2 只能是 prognostic。</li>
<li>Physiological dip（≤25%）對 chronic slope 的額外保護設定為每 1% dip +0.025 /yr，即刻意設定為 <em>幾乎沒有 dose–response</em>；Figure 3 的平台段反映此假設。</li>
<li>TTE-3 的 continue / stop 為以 common random numbers 模擬的 potential outcomes；IPCW 的實際權重未模擬，Table 6 為情境值，Figure 6 / Table 7 的信賴區間以 design effect 1.6 的 clustered bootstrap 近似。</li>
<li>Per-protocol RR（Table 3 註）、interaction p（Table 5）、Figure 7 的 sensitivity 偏移量與 Figure 8 全部為情境值，非由模擬資料計算。</li>
<li>真實分析時，以 <code>patient × visit</code> 長表與處方 refill 表取代模擬層；CCW 需以 pooled logistic model 估計 censoring weights，並回報 ESS。</li>
</ul>
</div>
</section>

<footer>Simulated results for study design — 依 PDPA 與醫療法第 60 條，本文件不含任何可識別個人資料；真實資料分析須於 IRB 核准範圍內、ISO 27001 受控環境執行，pseudonymised ID 與對照金鑰分開保存，日期逐病人位移。</footer>
</div>
</body>
</html>"""
out = os.path.join(HERE, "..", "index.html")
open(out, "w").write(html)
print(os.path.abspath(out), len(html) // 1024, "KB")
