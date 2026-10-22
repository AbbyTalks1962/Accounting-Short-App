import io
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Equity Screening Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1700px}
[data-testid="stMetric"]{background:rgba(120,140,160,.09);border:1px solid rgba(120,140,160,.28);padding:14px 16px;border-radius:12px;min-height:112px}
[data-testid="stMetricValue"]{font-weight:750}
[data-testid="stMetricLabel"]{font-weight:600}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stSidebar"]{border-right:1px solid #e8ebf0}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/JANalytics/us-stock-universe/main/us_listed_companies.csv"
SOURCE_PAGE = "https://github.com/JANalytics/us-stock-universe"

# Stable, distinct colours used for each sector across all charts and highlighted tables.
_SECTOR_PALETTE = (px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1)

def sector_color_map(sectors):
    names = sorted(str(s) for s in sectors if pd.notna(s))
    return {name: _SECTOR_PALETTE[i % len(_SECTOR_PALETTE)] for i, name in enumerate(names)}

def style_sector_cells(frame, color_map):
    """Tint the Sector cells so the table uses the same sector colour key as charts."""
    if frame is None or frame.empty or "Sector" not in frame.columns:
        return frame
    def _sector_style(value):
        color = color_map.get(str(value), "#f3f4f6")
        return f"background-color: {color}; color: #172033; font-weight: 600"
    return frame.style.map(_sector_style, subset=["Sector"])

@st.cache_data(ttl=12*60*60, show_spinner=False)
def fetch_sec_dataset():
    """Fetch a compact, prebuilt SEC-derived market-wide dataset from GitHub; no SEC bulk files or Yahoo calls."""
    response = requests.get(DATA_URL, timeout=60, headers={"User-Agent": "EquityScreeningLab/1.0 educational project"})
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), low_memory=False)


def numcol(df, names, scale=1.0):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce") / scale
    return pd.Series(np.nan, index=df.index, dtype=float)


def txtcol(df, names, default="Unknown"):
    for name in names:
        if name in df.columns:
            s = df[name].fillna("").astype(str).str.strip()
            return s.replace("", default)
    return pd.Series(default, index=df.index, dtype=object)


def build_screen_dataset(raw, count=1000):
    d = raw.copy()
    d["Ticker"] = txtcol(d, ["Ticker", "Symbol"], "").str.upper()
    d["Company"] = txtcol(d, ["Issuer Name", "SEC Company Name", "Company", "Security Name"], "Unknown company")
    d["Sector"] = txtcol(d, ["Sector"], "Unclassified")
    d["Industry"] = txtcol(d, ["Industry (SIC Description)", "Industry", "SIC Description"], "Unclassified")
    # Most dollar amounts in the source's derived columns are already in millions.
    assets_m = numcol(d, ["Total Assets ($M)"])
    if assets_m.isna().all(): assets_m = numcol(d, ["Assets"]) / 1_000_000
    liabilities_m = numcol(d, ["Liabilities"]) / 1_000_000
    equity_m = numcol(d, ["Shareholders Equity ($M)"])
    if equity_m.isna().all(): equity_m = numcol(d, ["Equity"]) / 1_000_000
    current_assets_m = numcol(d, ["AssetsCurrent"]) / 1_000_000
    current_liabilities_m = numcol(d, ["LiabilitiesCurrent"]) / 1_000_000
    if current_assets_m.isna().all(): current_assets_m = numcol(d, ["Current Assets ($M)"])
    if current_liabilities_m.isna().all(): current_liabilities_m = numcol(d, ["Current Liabilities ($M)"])
    cash_m = numcol(d, ["Cash & Investments ($M)"])
    if cash_m.isna().all(): cash_m = numcol(d, ["Cash", "CashAndCashEquivalentsAtCarryingValue"]) / 1_000_000
    debt_m = numcol(d, ["Total Debt ($M)"])
    if debt_m.isna().all():
        debt_m = (numcol(d, ["LongTermDebt", "LongTermDebtNoncurrent"]).fillna(0) + numcol(d, ["CurrentDebt", "LongTermDebtCurrent"]).fillna(0)) / 1_000_000
    revenue_m = numcol(d, ["Revenue ($M)"])
    if revenue_m.isna().all(): revenue_m = numcol(d, ["Revenue"]) / 1_000_000
    net_income_m = numcol(d, ["Net Income ($M)"])
    if net_income_m.isna().all(): net_income_m = numcol(d, ["NetIncome", "NetIncomeLoss"]) / 1_000_000
    gross_profit_m = numcol(d, ["Gross Profit ($M)"])
    if gross_profit_m.isna().all(): gross_profit_m = numcol(d, ["GrossProfit"]) / 1_000_000
    ebit_m = numcol(d, ["Operating Income ($M)"])
    if ebit_m.isna().all(): ebit_m = numcol(d, ["OperatingIncome", "OperatingIncomeLoss"]) / 1_000_000
    ocf_m = numcol(d, ["Operating Cash Flow ($M)"])
    if ocf_m.isna().all(): ocf_m = numcol(d, ["OCF", "NetCashProvidedByUsedInOperatingActivities"]) / 1_000_000
    retained_m = numcol(d, ["Retained Earnings ($M)"])
    if retained_m.isna().all(): retained_m = numcol(d, ["RetainedEarnings", "RetainedEarningsAccumulatedDeficit"]) / 1_000_000
    public_float_m = numcol(d, ["Public Float ($M)"])
    if public_float_m.isna().all(): public_float_m = numcol(d, ["PublicFloat"]) / 1_000_000

    # Transparent one-period accounting-ratio calculations from real SEC-derived values.
    d["STA"] = (net_income_m - ocf_m) / assets_m.replace(0, np.nan)
    # SNOA = net operating assets / total assets; cash and debt are excluded from operating items.
    d["SNOA"] = ((assets_m - cash_m) - (liabilities_m - debt_m)) / assets_m.replace(0, np.nan)
    wc_m = current_assets_m - current_liabilities_m
    # Public float is used as an explicitly labelled market-equity proxy; it is not full market cap.
    z_parts = pd.concat([
        1.2 * wc_m / assets_m.replace(0, np.nan),
        1.4 * retained_m / assets_m.replace(0, np.nan),
        3.3 * ebit_m / assets_m.replace(0, np.nan),
        0.6 * public_float_m / liabilities_m.replace(0, np.nan),
        revenue_m / assets_m.replace(0, np.nan),
    ], axis=1)
    d["Z_Score"] = z_parts.sum(axis=1, min_count=5)
    d["GPA"] = gross_profit_m / assets_m.replace(0, np.nan)
    d["BTM"] = equity_m / public_float_m.replace(0, np.nan)
    d["CFOA"] = ocf_m / assets_m.replace(0, np.nan)
    d["RevenueGrowth"] = numcol(d, ["Revenue Growth 1Y %"]) / 100
    d["RevenueCAGR2Y"] = numcol(d, ["Revenue CAGR 2Y %"]) / 100
    d["GrossMargin"] = numcol(d, ["Gross Margin %"]) / 100
    d["MarketCap"] = public_float_m * 1_000_000
    d["Assets_M"] = assets_m
    d["Revenue_M"] = revenue_m
    d["DataSource"] = "SEC filings via JANalytics US stock universe"
    d["DataDate"] = txtcol(d, ["Revenue Period End", "Assets As Of"], "Latest filed period")
    # Stable diversified sample: rank completeness, then round-robin across sectors.
    d = d[d["Ticker"].str.len().between(1, 12) & (d["Ticker"] != "UNKNOWN")].copy()
    d = d.drop_duplicates("Ticker")
    if "Data Completeness %" in d:
        d["_completeness"] = pd.to_numeric(d["Data Completeness %"], errors="coerce").fillna(0)
    else:
        metric_cols = ["STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin"]
        d["_completeness"] = d[metric_cols].notna().sum(axis=1)
    d = d.sort_values(["_completeness", "Ticker"], ascending=[False, True])
    if count < len(d):
        # Evenly draw from each sector, taking the most complete rows within each sector.
        sectors = sorted(d["Sector"].dropna().unique())
        buckets = {s: d[d["Sector"] == s].copy() for s in sectors}
        chosen = []
        while len(chosen) < count:
            added = False
            for s in sectors:
                if len(chosen) >= count: break
                if len(buckets[s]):
                    chosen.append(buckets[s].iloc[0])
                    buckets[s] = buckets[s].iloc[1:]
                    added = True
            if not added: break
        d = pd.DataFrame(chosen)
    d = d.drop(columns=["_completeness"], errors="ignore").reset_index(drop=True)
    return d


def rank_metric(s, high_good):
    x = pd.to_numeric(s, errors="coerce")
    n = len(x)
    # Missing metrics are ranked last, not treated as zero.
    ranks = x.rank(method="average", ascending=high_good is False, na_option="bottom")
    ranks = ranks.where(x.notna(), n + 1)
    return ranks


def adjusted_values(df, metrics, demean):
    """Create the exact values used for ranking; industry demeaning subtracts each
    industry's mean for that metric while preserving missing values as missing.
    """
    out = df.copy()
    if "Industry" not in out.columns:
        out["Industry"] = "Unclassified"
    industries = out["Industry"].fillna("Unclassified").astype(str)
    for m in metrics:
        values = pd.to_numeric(out[m], errors="coerce")
        if demean:
            industry_means = values.groupby(industries).transform("mean")
            adjusted = values - industry_means
            # If an industry has no usable observations, leave the values missing;
            # never silently replace missing ratios with zero.
            out[m + "_IndustryMean"] = industry_means
            out[m + "_Adj"] = adjusted
        else:
            out[m + "_IndustryMean"] = np.nan
            out[m + "_Adj"] = values
    return out


def add_rank_block(df, metrics, high_good, weights, demean, prefix):
    out = adjusted_values(df, metrics, demean)
    rank_cols = []
    for metric in metrics:
        col = f"R_{metric}"
        out[col] = rank_metric(out[metric + "_Adj"], high_good[metric])
        rank_cols.append(col)
    wsum = sum(max(0, weights.get(m, 1.0)) for m in metrics)
    if wsum == 0: wsum = 1.0
    out[prefix + "_Score"] = sum(out[f"R_{m}"] * max(0, weights.get(m, 1.0)) for m in metrics) / wsum
    out[prefix + "_Rank"] = out[prefix + "_Score"].rank(method="min", ascending=True)
    return out


def rerank_previous(df, weights, demean_flags):
    x = add_rank_block(df, ["STA", "SNOA", "Z_Score"], {"STA": False, "SNOA": False, "Z_Score": True}, weights, demean_flags[1], "Stage1")
    # Stage 2 uses the metrics that are consistently present in the prebuilt source: 1-year revenue growth,
    # 2-year revenue CAGR and gross margin. This is clearly identified as a shorter-horizon proxy, not 7-year MS/MG.
    x = add_rank_block(x, ["RevenueGrowth", "RevenueCAGR2Y", "GrossMargin"], {"RevenueGrowth": True, "RevenueCAGR2Y": True, "GrossMargin": True}, weights, demean_flags[2], "Stage2Metrics")
    x["Cumulative12"] = x["Stage1_Score"] + x["Stage2Metrics_Score"]
    x["Cumulative12_Rank"] = x["Cumulative12"].rank(method="min", ascending=True)
    x = add_rank_block(x, ["GPA", "BTM", "CFOA"], {"GPA": True, "BTM": True, "CFOA": True}, weights, demean_flags[3], "Stage3")
    x["FinalScore"] = x["Stage1_Score"] + x["Stage2Metrics_Score"] + x["Stage3_Score"]
    x["Final_Rank"] = x["FinalScore"].rank(method="min", ascending=True)
    return x


def run_screen(df, cuts, weights, demean_flags):
    current = df.copy()
    stages, audit = {}, []
    for stage, pct in [(1, cuts[1]), (2, cuts[2]), (3, cuts[3])]:
        # Recalculate every stage's rankings for the currently surviving population before each elimination.
        ranked = rerank_previous(current, weights, demean_flags)
        if stage == 1:
            score_col = "Stage1_Score"
        elif stage == 2:
            score_col = "Cumulative12"
        else:
            score_col = "FinalScore"
        ranked = ranked.sort_values([score_col, "Ticker"], ascending=[True, True]).reset_index(drop=True)
        ranked[f"Stage{stage}_CurrentRank"] = np.arange(1, len(ranked) + 1)
        n_remove = int(np.ceil(len(ranked) * pct / 100)) if pct else 0
        # Highest (worst) cumulative scores are eliminated.
        removed = ranked.tail(n_remove)["Ticker"].tolist() if n_remove else []
        removed_set = set(removed)
        ranked["Status"] = np.where(ranked["Ticker"].isin(removed_set), f"Removed at Stage {stage}", "Survived")
        stages[stage] = ranked.copy()
        audit.extend(ranked[ranked["Ticker"].isin(removed_set)][["Ticker", "Company", "Sector", "Industry", "Status"]].to_dict("records"))
        generated_cols = ["Stage1_Score", "Stage1_Rank", "Stage2Metrics_Score", "Stage2Metrics_Rank", "Stage3_Score", "Stage3_Rank", "Cumulative12", "Cumulative12_Rank", "FinalScore", "Final_Rank", "Status", f"Stage{stage}_CurrentRank"]
        current = ranked[~ranked["Ticker"].isin(removed_set)].drop(columns=[c for c in ranked.columns if c.startswith("R_") or c.endswith("_Adj") or c in generated_cols], errors="ignore").copy()
    final = rerank_previous(current, weights, demean_flags).sort_values(["Final_Rank", "Ticker"]).reset_index(drop=True)
    return stages, final, pd.DataFrame(audit)


st.title("Equity Screening Lab")
st.caption("Accounting-based stock screening · staged company analysis · final portfolio · BUY / SELL")

with st.sidebar:
    st.header("Screen settings")
    cuts = {1: st.slider("Stage 1 elimination (%)", 0, 50, 10, 5), 2: st.slider("Stage 2 elimination (%)", 0, 50, 10, 5), 3: st.slider("Stage 3 elimination (%)", 0, 50, 10, 5)}
    st.divider()
    st.subheader("Industry demeaning")
    st.caption("When enabled, each ratio is adjusted by subtracting the average for its industry before companies are ranked. This changes the values used in the stage score; it does not merely change how the table looks.")
    demean_flags = {
        1: st.checkbox("Demean Stage 1 by industry", value=True, key="demean_stage1", help="Adjust STA, SNOA and estimated Z-score relative to the average of each company's industry before ranking."),
        2: st.checkbox("Demean Stage 2 by industry", value=False, key="demean_stage2", help="Adjust revenue growth, two-year revenue CAGR and gross margin relative to each industry's average before ranking."),
        3: st.checkbox("Demean Stage 3 by industry", value=False, key="demean_stage3", help="Adjust GPA, BTM and CFOA relative to each industry's average before ranking."),
    }
    st.divider()
    st.subheader("Ratio weights")
    weights = {}
    for m in ["STA", "SNOA", "Z_Score", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA"]:
        weights[m] = st.slider(f"{m} weight", 0.0, 5.0, 1.0, 0.25, key=f"w_{m}")
    st.divider()
    count = st.select_slider("Companies to screen", options=[100, 250, 500, 750, 1000, 1500, 2000], value=1000)
    if st.button("Refresh dataset", use_container_width=True):
        fetch_sec_dataset.clear()

try:
    with st.spinner("Loading the prebuilt SEC-derived company dataset…"):
        raw = fetch_sec_dataset()
        data = build_screen_dataset(raw, int(count))
except Exception as e:
    st.error("The prebuilt financial dataset could not be reached. This is a single GitHub-hosted CSV download, not a per-company Yahoo/SEC API loop.")
    st.code(str(e))
    st.markdown(f"Dataset source: [{SOURCE_PAGE}]({SOURCE_PAGE})")
    st.stop()

st.caption(f"Source: [JANalytics US stock universe]({SOURCE_PAGE}) · SEC-derived financial records and listing data · {len(raw):,} source rows")

st.subheader("Filter the company universe")
f1, f2, f3, f4 = st.columns([1, 1, 1.25, 1])
sector_values = sorted(data["Sector"].unique().tolist())
selected_sectors = f1.multiselect("Sector", sector_values, default=sector_values)
industry_values = sorted(data.loc[data["Sector"].isin(selected_sectors), "Industry"].unique().tolist()) if selected_sectors else []
selected_industries = f2.multiselect("Industry", industry_values, default=industry_values)
search = f3.text_input("Company / ticker", placeholder="Search a name or ticker")
min_completeness = f4.slider("Minimum data completeness (%)", 0, 100, 0, 10, help="Filters out companies whose source-data completeness score is below this percentage. It measures how much data is available, not how financially strong the company is. Set to 0% to include companies regardless of completeness.")
filtered = data[data["Sector"].isin(selected_sectors) & data["Industry"].isin(selected_industries)].copy()
if search.strip():
    filtered = filtered[filtered["Company"].str.contains(search, case=False, na=False) | filtered["Ticker"].str.contains(search, case=False, na=False)]
if "Data Completeness %" in filtered:
    filtered = filtered[pd.to_numeric(filtered["Data Completeness %"], errors="coerce").fillna(0) >= min_completeness]
SECTOR_COLORS = sector_color_map(data["Sector"].dropna().unique())
if filtered.empty:
    st.warning("No companies match these filters. Widen your filters to continue.")
    st.stop()

stages, final, audit = run_screen(filtered, cuts, weights, demean_flags)
s1, s2, s3 = stages[1], stages[2], stages[3]
removed1 = int((s1.Status == 'Removed at Stage 1').sum())
removed2 = int((s2.Status == 'Removed at Stage 2').sum())
removed3 = int((s3.Status == 'Removed at Stage 3').sum())

# The recommended portfolio is a ranked shortlist of companies that survived all stages.
# It intentionally does not assign target investment weights.
portfolio = final.copy()

# Three clear top-level destinations: overview, full stage analysis, and portfolio actions.
overview_tab, analysis_tab, recommendations_tab = st.tabs(["Overview", "Stage Analysis", "Buy / Sell Recommendations"])

with overview_tab:
    st.header("Screening overview")
    st.caption("A snapshot of the selected company universe and the results of your current screening settings.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Companies screened", f"{len(filtered):,}", help="Companies matching the current sector, industry and completeness filters.", border=True)
    k2.metric("After Stage 1", f"{max(0, len(s1) - removed1):,}", delta=f"−{removed1} removed" if removed1 else "No removals", help="Companies remaining after the safety screen.", border=True)
    k3.metric("After Stage 2", f"{max(0, len(s2) - removed2):,}", delta=f"−{removed2} removed" if removed2 else "No removals", help="Companies remaining after the growth and margin screen.", border=True)
    k4.metric("Recommended portfolio", f"{len(final):,}", delta=f"−{removed3} removed in Stage 3" if removed3 else "Final survivors", help="Companies that survived all three stages.", border=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sectors represented", f"{filtered['Sector'].nunique():,}", border=True)
    m2.metric("Industries represented", f"{filtered['Industry'].nunique():,}", border=True)
    valid_z = pd.to_numeric(final.get("Z_Score", pd.Series(dtype=float)), errors="coerce").dropna()
    m3.metric("Average final Z-score", f"{valid_z.mean():.2f}" if len(valid_z) else "N/A", help="Mean of available estimated Z-scores among final survivors.", border=True)
    left, right = st.columns([1.25, 1])
    with left:
        sector_chart = filtered.groupby("Sector", as_index=False).size().sort_values("size", ascending=False)
        st.plotly_chart(px.bar(sector_chart, x="Sector", y="size", color="Sector", color_discrete_map=SECTOR_COLORS, title="Starting universe by sector", labels={"size": "Companies"}), use_container_width=True)
    with right:
        stage_summary = pd.DataFrame({"Stage": ["Starting universe", "After Stage 1", "After Stage 2", "Final portfolio"], "Companies": [len(filtered), max(0, len(s1)-removed1), max(0, len(s2)-removed2), len(final)]})
        st.plotly_chart(px.line(stage_summary, x="Stage", y="Companies", markers=True, title="Companies remaining after each stage"), use_container_width=True)
    st.subheader("Current top-ranked companies")
    top_cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "FinalScore", "STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA"]
    top_view = final[[c for c in top_cols if c in final]].head(20)
    st.dataframe(style_sector_cells(top_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
    with st.expander("Browse the company database"):
        show_cols = ["Ticker", "Company", "Sector", "Industry", "STA", "SNOA", "Z_Score", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA", "DataDate"]
        st.dataframe(data[[c for c in show_cols if c in data]], use_container_width=True, hide_index=True)
        st.download_button("Download company database CSV", data.to_csv(index=False).encode(), "company_universe.csv", "text/csv", key="overview_download_database")
    st.info("**Data-method note:** the source does not provide eight annual gross-margin observations for every company. Stage 2 therefore uses available one-year revenue growth, two-year revenue CAGR and gross margin as transparent proxies, rather than presenting them as seven-year MS/MG. The Z-score uses public float as a market-equity proxy where available, so it is an estimate rather than the exact standard Altman Z-score. Missing values are not fabricated.")

with analysis_tab:
    st.header("Stage analysis & recommended portfolio")
    st.caption("Each stage ranks the surviving population again. The final portfolio is formed from companies that survive all three stages.")
    st1_tab, st2_tab, st3_tab, portfolio_tab, audit_tab = st.tabs(["Stage 1 · Safety", "Stage 2 · Growth & margins", "Stage 3 · Profitability", "Recommended Portfolio", "Elimination Audit"])
    with st1_tab:
        st.markdown("""
        ### What Stage 1 measures: safety and accounting quality
        - **STA (scaled total accruals)** = (net income − operating cash flow) ÷ total assets. It compares accounting earnings with cash generated by operations. Higher values can mean earnings are less supported by cash, so **lower is preferred** in this screen.
        - **SNOA (scaled net operating assets)** = [(total assets − cash) − (total liabilities − debt)] ÷ total assets. It estimates net operating assets relative to the asset base. Higher values can be associated with more accrual-heavy operating assets, so **lower is preferred** as a screening signal.
        - **Estimated Altman Z-score** = 1.2 × working capital/assets + 1.4 × retained earnings/assets + 3.3 × operating income/assets + 0.6 × public-float proxy/total liabilities + revenue/assets. **Higher is generally better**; a low score can indicate distress risk. Because public float substitutes for full market equity here, this is an estimate, not an exact standard Z-score.

        Each metric is ranked in the preferred direction and combined using the selected weights. If industry demeaning is enabled, the metric is adjusted relative to its industry's average before ranking. The weakest composite ranks are eliminated. These ratios are screening signals, not proof of fraud or a guarantee of safety.
        """)
        st.caption("Industry demeaning is **ON** for Stage 1." if demean_flags[1] else "Industry demeaning is **OFF** for Stage 1; raw ratio values are ranked.")
        if demean_flags[1] and len(s1):
            preview = adjusted_values(s1, ["STA", "SNOA", "Z_Score"], True)
            preview_cols = ["Ticker", "Industry", "STA", "STA_IndustryMean", "STA_Adj", "SNOA", "SNOA_IndustryMean", "SNOA_Adj", "Z_Score", "Z_Score_IndustryMean", "Z_Score_Adj"]
            with st.expander("Check how Stage 1 demeaning changes the values"):
                st.caption("Adjusted value = raw ratio − that industry's average. These adjusted values are what the rank calculations use.")
                st.dataframe(preview[[c for c in preview_cols if c in preview]], use_container_width=True, hide_index=True)
        cols = ["Stage1_CurrentRank", "Ticker", "Company", "Sector", "Industry", "STA", "SNOA", "Z_Score", "Stage1_Score", "Status"]
        st.dataframe(style_sector_cells(s1[[c for c in cols if c in s1]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download Stage 1 results", s1.to_csv(index=False).encode(), "stage1_safety.csv", "text/csv")
    with st2_tab:
        st.markdown("""
        ### What Stage 2 measures: growth and margins
        - **One-year revenue growth** = the source-reported percentage change in revenue over one year. Higher growth is preferred, although unusually high growth is not automatically sustainable.
        - **Two-year revenue CAGR** = the source-reported compound annual revenue growth rate over two years. It summarizes the annualized growth pace across that period.
        - **Gross margin** = gross profit ÷ revenue. It shows how much revenue remains after the direct costs of goods or services, before operating expenses. Higher margin is preferred in this screen, but margins differ by industry.

        The available dataset does not contain consistent seven-year annual margin histories for every company, so this stage uses these clearly labelled shorter-horizon proxies rather than claiming to calculate seven-year MS/MG. These growth/margin metrics are ranked in the preferred direction and combined with the refreshed Stage 1 score; the weakest cumulative ranks are eliminated.
        """)
        st.caption("Industry demeaning is **ON** for Stage 2." if demean_flags[2] else "Industry demeaning is **OFF** for Stage 2; raw ratio values are ranked.")
        if demean_flags[2] and len(s2):
            preview = adjusted_values(s2, ["RevenueGrowth", "RevenueCAGR2Y", "GrossMargin"], True)
            preview_cols = ["Ticker", "Industry", "RevenueGrowth", "RevenueGrowth_IndustryMean", "RevenueGrowth_Adj", "RevenueCAGR2Y", "RevenueCAGR2Y_IndustryMean", "RevenueCAGR2Y_Adj", "GrossMargin", "GrossMargin_IndustryMean", "GrossMargin_Adj"]
            with st.expander("Check how Stage 2 demeaning changes the values"):
                st.caption("Adjusted value = raw ratio − that industry's average. These adjusted values are what the rank calculations use.")
                st.dataframe(preview[[c for c in preview_cols if c in preview]], use_container_width=True, hide_index=True)
        cols = ["Ticker", "Company", "Sector", "Industry", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "Stage1_Score", "Stage2Metrics_Score", "Cumulative12", "Status"]
        st.dataframe(style_sector_cells(s2[[c for c in cols if c in s2]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download Stage 2 results", s2.to_csv(index=False).encode(), "stage2_growth_margins.csv", "text/csv")
    with st3_tab:
        st.markdown("""
        ### What Stage 3 measures: profitability, value and cash generation
        - **GPA (gross profitability to assets)** = gross profit ÷ total assets. It measures gross profit generated per dollar of assets. Higher is preferred.
        - **BTM (book-to-market)** = book shareholders' equity ÷ the available public-float market-value proxy. A higher value can indicate that book equity is high relative to the market-value proxy, which this screen treats as a potential value signal. The proxy is imperfect, so interpret it cautiously and consistently with your course's methodology.
        - **CFOA (cash flow from operations to assets)** = operating cash flow ÷ total assets. It measures operating cash generated per dollar of assets. Higher is preferred.

        The three ratios are ranked separately in the preferred direction, then combined using the selected weights. Missing observations are ranked last rather than replaced with zero. The final portfolio consists of the companies that survive all three stages.
        """)
        st.caption("Industry demeaning is **ON** for Stage 3." if demean_flags[3] else "Industry demeaning is **OFF** for Stage 3; raw ratio values are ranked.")
        if demean_flags[3] and len(final):
            preview = adjusted_values(final, ["GPA", "BTM", "CFOA"], True)
            preview_cols = ["Ticker", "Industry", "GPA", "GPA_IndustryMean", "GPA_Adj", "BTM", "BTM_IndustryMean", "BTM_Adj", "CFOA", "CFOA_IndustryMean", "CFOA_Adj"]
            with st.expander("Check how Stage 3 demeaning changes the values"):
                st.caption("Adjusted value = raw ratio − that industry's average. These adjusted values are what the rank calculations use.")
                st.dataframe(preview[[c for c in preview_cols if c in preview]], use_container_width=True, hide_index=True)
        cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "GPA", "BTM", "CFOA", "Stage1_Score", "Stage2Metrics_Score", "Stage3_Score", "FinalScore", "Status"]
        st.dataframe(style_sector_cells(final[[c for c in cols if c in final]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download final survivors", final.to_csv(index=False).encode(), "stage3_survivors.csv", "text/csv")
    with portfolio_tab:
        st.subheader("Recommended portfolio")
        st.caption("This is a ranked shortlist of companies that survived all three screening stages. It does not assign investment amounts or target portfolio weights.")
        if len(portfolio):
            st.metric("Companies that passed all three stages", f"{len(portfolio):,}")
            portfolio_cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "FinalScore", "STA", "SNOA", "Z_Score", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA"]
            portfolio_view = portfolio[[c for c in portfolio_cols if c in portfolio]].sort_values("Final_Rank")
            st.dataframe(style_sector_cells(portfolio_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
            st.download_button("Download recommended portfolio", portfolio.to_csv(index=False).encode(), "recommended_portfolio.csv", "text/csv")
        else:
            st.warning("No companies survived the current screening settings.")


    with audit_tab:
        if audit.empty:
            st.write("No companies were eliminated under the current settings.")
        else:
            st.dataframe(audit, use_container_width=True, hide_index=True)
            st.download_button("Download elimination audit", audit.to_csv(index=False).encode(), "elimination_audit.csv", "text/csv")
with recommendations_tab:
    st.header("Buy / Sell recommendations")
    st.caption("SELL decisions focus on accrual-quality red flags. BUY decisions focus on profitability and value. Z-score is a separate financial-distress context flag, not a flipped BUY/SELL rule.")

    with st.expander("Methodology: what makes a BUY or SELL?", expanded=True):
        st.markdown("""
        ### SELL: accrual-quality red flags
        **STA** is calculated here as `(net income − operating cash flow) ÷ total assets`. A higher value means accounting earnings exceed operating cash flow by more relative to assets, which can be a warning sign about earnings quality. **SNOA** is calculated as `[(total assets − cash) − (total liabilities − debt)] ÷ total assets`; higher values indicate more net operating assets relative to the asset base. These are screening indicators, not proof of manipulation.

        A company is flagged for an **accrual red flag** when either STA or SNOA is in the worst 20% of the current filtered portfolio, or when both measures are above their respective medians. The cut-off can be adjusted below. The Z-score does **not** by itself trigger SELL.

        ### BUY: profitability and value ranks
        **GPA** = gross profit ÷ total assets; higher is preferred because the company generates more gross profit per dollar of assets. **CFOA** = operating cash flow ÷ total assets; higher is preferred because operations generate more cash relative to assets. **BTM** = book equity ÷ available public-float market-value proxy in this dataset; higher is treated as a value signal in this screen, but it is an imperfect proxy and should be interpreted carefully.

        BUY decisions use **only GPA, BTM and CFOA**. Each metric is ranked across the current final portfolio, with missing values placed last rather than replaced with zero. The profitability/value score is the average rank of these three metrics (rank 1 is best). A company receives BUY if its combined rank falls within the selected top percentile. STA and SNOA do not affect this BUY calculation.

        ### SELL: accrual-quality red flags
        **STA** is `(net income − operating cash flow) ÷ total assets`; a higher value can indicate that accounting earnings are less supported by operating cash. **SNOA** is `[(total assets − cash) − (total liabilities − debt)] ÷ total assets`; higher values are treated as an accrual-quality warning signal. These measures are screening indicators, not proof of manipulation. A company receives a SELL if it falls outside the selected BUY profitability/value threshold; the SELL explanation separately identifies when an STA/SNOA accrual flag is present. STA and SNOA do not change a company's BUY eligibility.
        

        ### Z-score: a separate risk check, not the recommendation switch
        The app shows the estimated Altman Z-score and a separate review flag. A low Z-score may suggest financial distress and deserves investigation, but it does not automatically determine BUY or SELL. The available source uses public float as a proxy for market equity, so this is an estimated Z-score, not a definitive standard-model result.

        ### Other screening metrics and data quality
        The overall survivor portfolio still reflects all three stages: STA, SNOA and estimated Z-score; Stage 2 revenue growth, two-year revenue CAGR and gross margin; and Stage 3 GPA, BTM and CFOA. The data source does not provide consistent seven-year margin histories for all companies, so Stage 2 uses explicitly labelled shorter-horizon proxies instead of claiming to calculate the requested seven-year MS/MG. Missing financial inputs remain missing; metric ranks put missing observations last.
        """)

    if len(portfolio):
        # Signals are intentionally NOT a simple inversion of Z-score or the overall composite.
        signal_controls_1, signal_controls_2 = st.columns(2)
        buy_pct = signal_controls_1.slider("BUY: top percentile of profitability/value ranks", 5, 80, 40, 5, key="buy_threshold")
        accrual_cut = signal_controls_2.slider("Accrual red-flag threshold (worst percentile)", 10, 40, 20, 5, key="accrual_flag_threshold")

        recs = portfolio.copy().sort_values(["Final_Rank", "Ticker"]).reset_index(drop=True)
        # Build directional metric ranks independently of the full screening score.
        for metric in ["STA", "SNOA"]:
            vals = pd.to_numeric(recs[metric], errors="coerce")
            # For accrual flags, larger values are worse; percentile is 0..100.
            recs[f"{metric}_BadPct"] = vals.rank(method="average", pct=True) * 100
        sta_bad = recs["STA_BadPct"] >= (100 - accrual_cut)
        snoa_bad = recs["SNOA_BadPct"] >= (100 - accrual_cut)
        both_above_median = (pd.to_numeric(recs["STA"], errors="coerce") > pd.to_numeric(recs["STA"], errors="coerce").median()) & (pd.to_numeric(recs["SNOA"], errors="coerce") > pd.to_numeric(recs["SNOA"], errors="coerce").median())
        recs["Accrual_Red_Flag"] = (sta_bad | snoa_bad | both_above_median).fillna(False)

        # Higher GPA, BTM and CFOA are preferred for the BUY-side profitability/value score.
        value_profit_rank_cols = []
        for metric in ["GPA", "BTM", "CFOA"]:
            vals = pd.to_numeric(recs[metric], errors="coerce")
            rank_col = f"{metric}_GoodRank"
            # Missing values are explicitly ranked last.
            recs[rank_col] = vals.rank(method="average", ascending=False, na_option="bottom")
            recs.loc[vals.isna(), rank_col] = len(recs) + 1
            value_profit_rank_cols.append(rank_col)
        recs["Profitability_Value_Score"] = recs[value_profit_rank_cols].mean(axis=1)
        recs["Profitability_Value_Rank"] = recs["Profitability_Value_Score"].rank(method="min", ascending=True)
        recs["Strong_Profitability_Value"] = recs["Profitability_Value_Rank"] <= max(1, int(np.ceil(len(recs) * buy_pct / 100)))

        # BUY eligibility is determined ONLY by GPA, BTM and CFOA. STA/SNOA are
        # used to explain SELL-side accrual warnings, never to override a BUY.
        recs["Signal"] = np.where(recs["Strong_Profitability_Value"], "BUY", "SELL")
        recs["Recommendation_Reason"] = np.select(
            [recs["Strong_Profitability_Value"], recs["Accrual_Red_Flag"]],
            ["BUY: qualifies on combined GPA, BTM and CFOA ranks; STA/SNOA do not affect BUY eligibility.", "SELL: outside the selected BUY profitability/value threshold; also has an STA/SNOA accrual-quality flag."],
            default="SELL: does not meet the selected GPA, BTM and CFOA BUY threshold."
        )
        recs["Z_Score_Review_Flag"] = np.where(pd.to_numeric(recs["Z_Score"], errors="coerce").lt(1.81), "Review: low estimated Z-score", "")

        buy_count = int((recs["Signal"] == "BUY").sum())
        sell_count = int((recs["Signal"] == "SELL").sum())
        red_flag_count = int(recs["Accrual_Red_Flag"].sum())
        a, b, c, d = st.columns(4)
        a.metric("BUY recommendations", f"{buy_count:,}", help="Qualifies based only on the combined GPA, BTM and CFOA rank. STA/SNOA do not affect BUY eligibility.", border=True)
        b.metric("SELL recommendations", f"{sell_count:,}", help="Companies outside the selected GPA/BTM/CFOA BUY threshold; accrual flags are shown separately as context.", border=True)
        c.metric("Accrual red flags", f"{red_flag_count:,}", help="Flagged from STA and/or SNOA relative to the current final portfolio.", border=True)
        d.metric("Low Z-score review flags", f"{int(recs['Z_Score_Review_Flag'].ne('').sum()):,}", help="Separate review flag only; Z-score does not directly decide BUY/SELL.", border=True)

        signal_chart = recs["Signal"].value_counts().reindex(["BUY", "SELL"], fill_value=0).rename_axis("Signal").reset_index(name="Companies")
        st.plotly_chart(px.bar(signal_chart, x="Signal", y="Companies", color="Signal", color_discrete_map={"BUY": "#2E8B57", "SELL": "#C94C4C"}, title="BUY / SELL recommendations"), use_container_width=True)

        def build_case(row, signal):
            pieces = []
            def fmt(metric, pct=False):
                v = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                if pd.isna(v): return f"{metric}: unavailable"
                return f"{metric}: {v:.1%}" if pct else f"{metric}: {v:.3f}"
            if signal == "BUY":
                pieces.append(f"Profitability/value rank {int(row['Profitability_Value_Rank'])} of {len(recs)}")
                pieces.append(fmt("GPA")); pieces.append(fmt("BTM")); pieces.append(fmt("CFOA"))
            else:
                if bool(row["Accrual_Red_Flag"]):
                    flagged = []
                    if bool(sta_bad.loc[row.name]): flagged.append("STA is in the high-accrual tail")
                    if bool(snoa_bad.loc[row.name]): flagged.append("SNOA is in the high-accrual tail")
                    if bool(both_above_median.loc[row.name]): flagged.append("both STA and SNOA exceed their medians")
                    pieces.extend(flagged or ["STA/SNOA accrual-quality flag"])
                else:
                    pieces.append(f"Profitability/value rank {int(row['Profitability_Value_Rank'])} of {len(recs)} does not meet the BUY threshold")
                    pieces.append(fmt("GPA")); pieces.append(fmt("BTM")); pieces.append(fmt("CFOA"))
            z = pd.to_numeric(pd.Series([row.get("Z_Score")]), errors="coerce").iloc[0]
            if pd.notna(z): pieces.append(f"estimated Z-score {z:.2f}" + (" (separate distress review flag)" if z < 1.81 else " (no low-Z review flag)"))
            if pd.isna(row.get("STA")) or pd.isna(row.get("SNOA")) or pd.isna(row.get("GPA")) or pd.isna(row.get("BTM")) or pd.isna(row.get("CFOA")):
                pieces.append("some key inputs are missing; interpret cautiously")
            return "; ".join(pieces) + "."

        st.subheader("Written investment cases")
        buy_cases = recs[recs["Signal"] == "BUY"].sort_values("Profitability_Value_Rank").head(5)
        sell_cases = recs[recs["Signal"] == "SELL"].copy()
        sell_cases["_priority"] = sell_cases["Accrual_Red_Flag"].astype(int)
        sell_cases = sell_cases.sort_values(["_priority", "STA_BadPct", "SNOA_BadPct", "Profitability_Value_Rank"], ascending=[False, False, False, False]).head(5)
        case_col1, case_col2 = st.columns(2)
        with case_col1:
            st.markdown("#### Top BUY cases")
            if buy_cases.empty:
                st.info("No companies currently fall within the selected BUY percentile. Increase the BUY threshold or review the filters.")
            for _, row in buy_cases.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Ticker']} — {row['Company']}**")
                    st.caption(f"{row['Sector']} · {row['Industry']}")
                    st.write(build_case(row, "BUY"))
        with case_col2:
            st.markdown("#### Top SELL cases")
            if sell_cases.empty:
                st.info("No SELL cases in the current filtered portfolio.")
            for _, row in sell_cases.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Ticker']} — {row['Company']}**")
                    st.caption(f"{row['Sector']} · {row['Industry']}")
                    st.write(build_case(row, "SELL"))

        st.subheader("Company-level recommendations")
        recommendation_cols = ["Signal", "Recommendation_Reason", "Final_Rank", "Profitability_Value_Rank", "Ticker", "Company", "Sector", "Industry", "FinalScore", "STA", "SNOA", "Accrual_Red_Flag", "Z_Score", "Z_Score_Review_Flag", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA"]
        recommendation_view = recs[[c for c in recommendation_cols if c in recs]]
        st.dataframe(style_sector_cells(recommendation_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download BUY / SELL recommendations", recs.to_csv(index=False).encode(), "buy_sell_recommendations.csv", "text/csv")
        st.caption("Filters above apply before screening and recommendations, so changing sector, industry, search or completeness changes the companies being ranked and flagged. BUY/SELL labels are educational screening outputs, not predictions of future returns.")
    else:
        st.warning("No companies survived the screening process, so there are no portfolio recommendations to display.")

st.caption("Source: JANalytics US stock universe. Financial fields may be blank when filings are unavailable or not comparable. This educational tool does not provide personalized investment advice.")
