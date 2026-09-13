# AI Credit Analyst Copilot

## What this is

A resume-anchor fintech/AI portfolio project, built as a genuine MVP with real engineering
rigor (not a toy demo), targeting an internship at **CreditRiskMonitor (CRM)** via a warm
referral. The owner is a 3rd-year B.S. Economics student at IIT Patna, learning ML/finance/
product concepts *through* building this — explain the "why" behind decisions, not just hand
over working code, except for mechanical/repetitive tasks where moving fast is fine.

GitHub repo: https://github.com/keshavbaviskar33-sudo/AI-Credit-Analyst-Copilot

## Product definition (locked in — do not relitigate without new evidence)

- **User:** a trade-credit/collections manager (e.g. a CreditRiskMonitor subscriber) deciding
  credit exposure to customer companies — NOT a bank lender or bond investor.
- **Core problem:** coverage/attention at scale (monitoring many companies, catching
  deterioration) — secondarily speed and consistency.
- **Output:** risk score/category + ratios + AI explanation + recommendation *category* —
  NEVER a specific dollar credit limit or payment terms (requires business context — margin,
  risk appetite, portfolio exposure — the model structurally cannot have). The human analyst
  always holds final approve/modify/reject authority.
- **MVP unit of work:** one company, one filing snapshot in → structured risk assessment out.
  Portfolio/time-series monitoring is roadmap, not MVP.

## Repo structure (as of this writing)

```
AI credit analyst copilot/          <- repo root, working dir D:\Projects\AI credit analyst copilot
├── app/
│   ├── app.py                      <- Streamlit MVP app (see below)
│   ├── model_v3.pkl                <- trained LogisticRegression (committed)
│   ├── scaler.pkl                  <- fitted StandardScaler, MUST pair with model_v3.pkl
│   ├── shap_background.pkl         <- 100-row sample of scaled training data, SHAP background
│   ├── requirements.txt
│   └── review_log.csv              <- gitignored; analyst review audit trail, generated at runtime
├── notebooks/
│   ├── 01_initial_exploration.ipynb        <- frozen, raw data EDA
│   └── 02_ratio_engineering_and_baseline_model.ipynb  <- ratio engineering, target
│       construction, model comparisons, SHAP work. Authoritative history — app.py's
│       ratio/model logic must match this notebook exactly, not diverge.
├── .streamlit/
│   └── secrets.toml                <- gitignored; holds GEMINI_API_KEY, never commit
├── .gitignore
└── README.md
```

Run the app from the repo root (so `.streamlit/secrets.toml` resolves):
```
python -m streamlit run app/app.py
```

## Dataset & feature engineering (decisions have documented reasoning in notebook 02)

- Dataset: Kaggle `utkarshx27/american-companies-bankruptcy-prediction-dataset` — raw
  financial line items (X1–X18), NOT pre-computed ratios (this project builds its own ratio
  engine). Downloaded via `kagglehub.dataset_download(...)`, requires a Kaggle API token.
- ~78,682 rows → 74,071 after removing "zombie" post-failure rows (status_label is
  persistent once a company fails, not one-time — a real bug found and fixed).
- Target: 1 if the company fails within 1–2 years of that filing (Altman Z-score precedent
  for prediction horizon), else 0. ~0.8% positive class — genuinely extreme imbalance,
  confirmed normal for this domain, not a blocker, but means results need larger test sets/
  cross-validation before being fully trusted.
- Split: the dataset's own published train (1999–2011) / val (2012–2014) / test (2015–2018)
  split, adopted rather than inventing one, after finding 1999 alone holds ~64% of all
  failures (left-censoring artifact).
- 5 features, all winsorized at 1st/99th percentile (critical fix — several had extreme
  finite outliers from near-zero denominators, not caught by NaN/inf checks alone):
  - `current_ratio = X1/X14`
  - `debt_to_equity = X11/(X10-X17)` — **known limitation:** doesn't distinguish healthy low
    leverage from negative-equity distress (confirmed cause of one false negative, company
    C_7958: -7.4 read as "very low leverage" instead of genuine distress).
  - `net_profit_margin = X6/X16`
  - `ebitda_margin = X4/X16` (chosen because EBITDA excludes interest expense, sidestepping
    the dataset's missing interest-expense field)
  - `retained_earnings_to_assets = X15/X10` (an Altman Z-score component)
  - Interest coverage ratio: NOT computable — no interest expense field in source data,
    documented limitation.

## Model selection (4 experiments, documented — do not re-litigate without new evidence)

- **Logistic Regression** (`class_weight='balanced'`, StandardScaler-scaled): TP=3, FN=2,
  FP=3565 on test set (5 total actual failures) — recall=0.60. **Winner, this is `model_v3.pkl`.**
- Random Forest (`class_weight='balanced'`): TP=0 — degenerate, predicts majority class only.
- Random Forest (`class_weight='balanced_subsample'`): TP=0 — same.
- Random Forest + SMOTE (training data only): TP=1, FN=4 — better but still worse than LR.
- XGBoost (`scale_pos_weight=88.8`): TP=0 — worse than Random Forest.
- Conclusion: tree-based methods underperform on this tiny positive class (~580 training
  examples); logistic regression's simpler global structure generalizes better. A genuine,
  documented finding, not an accident.
- SHAP (`LinearExplainer`) for both global feature importance and per-prediction waterfall
  explanations. Global summary plot showed one counterintuitive pattern (net_profit_margin
  direction), investigated and attributed to noise from the tiny positive class, not a bug.

## Current app functionality (`app/app.py`)

Single-page Streamlit app, pipeline: raw financial inputs → 5 ratios → scaled → LR
prediction + probability → SHAP waterfall explanation → **Gemini-generated plain-English
narrative** → human analyst review, logged to CSV.

1. **Inputs:** 9 raw line items (X1, X4, X6, X10, X11, X14, X15, X16, X17) via
   `st.number_input`.
2. **Assess Risk button:** computes ratios, scales via `scaler.pkl`, predicts via
   `model_v3.pkl`, shows prediction + probability.
3. **SHAP waterfall:** built from `explainer = shap.LinearExplainer(model, shap_background)`
   — `shap_background` is a 100-row sample of *scaled training data*, built once at startup,
   **not** a single-row background (that was a real bug: a single-row background collapsed
   every waterfall to near-zero contributions — fixed by regenerating `shap_background.pkl`
   from the real notebook-02 pipeline and reusing the already-deployed `scaler.pkl` to avoid
   train-serve skew). Waterfall displays raw (unscaled) ratio values for analyst readability,
   per an earlier documented decision — only the SHAP contribution values themselves are
   computed on scaled inputs.
4. **AI Analyst Summary (Phase 12, "simplified LLM synthesis" per roadmap):** a single call
   to **Google Gemini** (`google-genai` SDK, model `gemini-3.5-flash`) — NOT the Anthropic
   API. The user explicitly chose Gemini over Claude for this feature; don't re-suggest
   switching back without being asked. Deliberately a single API call, not a multi-tool
   agent — matches the roadmap's explicit scope decision for Phase 12. System prompt
   hard-constrains the model: 3–5 sentence narrative grounded in the given ratios/SHAP
   values, ends with exactly one recommendation category from `{"Standard Monitoring",
   "Increased Monitoring", "Escalate for Analyst Review"}`, never a dollar figure — mirrors
   the locked product definition. Errors from the Gemini call (`google.genai.errors.APIError`)
   are caught and shown as a friendly `st.error`, never crash the app.
5. **Analyst Review:** approve/modify/reject + comment, persisted to `app/review_log.csv`
   (timestamp, ratios, prediction, probability, decision, comment) — gitignored, it's runtime
   data, not source.

## Environment gotchas (specific to this machine — worth knowing before debugging blind)

- **Norton Antivirus does TLS inspection** on this machine, injecting a root certificate that
  Python's `certifi` bundle doesn't trust (though the Windows OS trust store does, which is
  why browsers work fine). This broke HTTPS calls to `api.kaggle.com` and the Gemini API
  specifically, even though e.g. `google.com` worked. Fixed by adding
  `import truststore; truststore.inject_into_ssl()` at the top of `app.py` (and in any
  standalone script hitting these APIs) — makes Python use the OS trust store instead of
  bundled certs. If a new HTTPS integration mysteriously fails with
  `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` on this machine, this
  is almost certainly why.
- **Kaggle auth:** uses the new-style Kaggle API token (`KGAT_...` prefix, generated at
  kaggle.com/settings → API → Create New Token), applied via
  `kagglehub.auth.set_kaggle_api_token(token)` in-process — never write it to a file that
  gets committed. No `kaggle.json` or `~/.kaggle/` credentials are stored on this machine.
- **Gemini auth:** `GEMINI_API_KEY` lives in `.streamlit/secrets.toml` at the repo root
  (Streamlit resolves `.streamlit/secrets.toml` relative to the working directory the app
  was launched from, which is repo root here). `app.py` reads it via `st.secrets` and sets
  it as an env var so the SDK's default credential resolution picks it up. This file is
  gitignored — confirmed via `git check-ignore -v`.
- Both the Kaggle token and the Gemini key were at one point pasted directly into chat by
  the user — flagged for rotation, unclear if actually rotated. Neither was ever written to
  a committed file.
- Python environment here is a global/user install (`pip install` without a venv), Python
  3.14.7, packages under `C:\Users\KESHAV\AppData\Roaming\Python\Python314\site-packages`.
  `model_v3.pkl`/`scaler.pkl` were originally trained under scikit-learn 1.6.1 (likely in
  Colab); this machine runs scikit-learn 1.9.1 — produces a harmless
  `InconsistentVersionWarning` on every load, not a real problem, don't try to "fix" it by
  downgrading sklearn.

## Roadmap status (20-phase plan — see full plan history if resuming long-term work)

| Phase | Status |
|---|---|
| 1. Define product | Done |
| 2. Environment/repo setup | Done |
| 3. Acquire/understand data | Done |
| 4. Document extraction (PDF→structured) | Not started |
| 5. Financial statement parsing | Not started (using clean CSV, not real PDFs, for MVP) |
| 6. Deterministic ratio engine | Done |
| 7. Financial-health analysis | Partial (ratios exist, no narrative synthesis layer beyond Phase 12's single-call LLM) |
| 8. Baseline ML credit-risk model | Done |
| 9. Evaluate/explain ML model | Done (SHAP integrated, background bug fixed) |
| 10. NLP earnings-risk analysis | Not started, explicitly deferred post-MVP |
| 11. Combine outputs | Done (Streamlit app) |
| 12. First agentic workflow (simplified single-call LLM synthesis) | **Done** — Gemini-based, see above. Deliberately not a multi-tool agent. |
| 13. Human analyst review | Done (approve/modify/reject + comment, persisted to CSV) |
| 14. Dashboard | Functionally complete for MVP |
| 15. SQL/database architecture | Not started, deferred post-MVP |
| 16. External API integration | Not started, deferred post-MVP |
| 17. Evaluation/robustness | Ongoing |
| 18. Polish product/UX | Explicitly deferred — do not spend time on styling unless asked |
| 19. Documentation/architecture diagrams | Partial (inline notebook markdown only) |
| 20. Resume/interview prep | Not started |

## Working style preferences

- Explain the WHY behind decisions, not just hand over working code — except for mechanical/
  repetitive implementation tasks, where moving fast with real code is fine and expected.
- When something breaks, treat it as a real debugging/learning opportunity — explain root
  cause, not just the patch.
- Don't silently expand scope beyond what's asked. Flag premature complexity given limited
  available time (exams and other commitments).
- UI polish is explicitly deferred — don't do it unprompted.
- Before pushing to GitHub or making other visible/shared-state changes, confirm unless the
  user has explicitly listed it as a next step already.
