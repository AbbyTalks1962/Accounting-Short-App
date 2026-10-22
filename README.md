# Equity Screening Lab

A Streamlit educational stock-screening dashboard with three main tabs:

1. **Overview** — universe summary, populated KPI cards, sector distribution and top-ranked companies.
2. **Stage Analysis** — safety, growth/margin and profitability screens, elimination audit, and recommended portfolio construction.
3. **Buy / Sell Recommendations** — BUY/SELL signals for the recommended portfolio.

## Data

The app downloads a prebuilt public-company dataset from the JANalytics US stock universe GitHub repository at runtime and caches it for 12 hours. No large SEC file or manually uploaded Excel file is required. An internet connection is needed for the initial download.

Source: https://github.com/JANalytics/us-stock-universe

## Styling updates

- KPI cards now show the screened population, surviving populations, final portfolio size, sectors and industries represented, and average final Z-score.
- Sector charts use a consistent distinct colour for each sector.
- Sector cells in company tables are tinted with the matching sector colour.
- The requested explanatory sentence about ranking direction and data loading has been removed from the page header.

## Important methodology notes

- The source does not provide eight annual gross-margin observations for every company. Stage 2 therefore uses available one-year revenue growth, two-year revenue CAGR and gross margin as proxies rather than presenting them as seven-year MS/MG.
- The Z-score uses public float as a market-equity proxy where available, so it is an estimate rather than the exact standard Altman Z-score.
- Missing values are not fabricated. Some metrics may be unavailable for some companies.
- BUY/SELL signals are not based on flipping the overall rank or on Z-score alone. SELL flags focus on high STA/SNOA accrual signals; BUY eligibility depends only on strong combined GPA, BTM and CFOA ranks; STA/SNOA do not affect BUY eligibility. Z-score is displayed as a separate distress-review flag. Filters recalculate the universe, ranks and recommendation flags. The app generates short written cases for the top BUY and SELL names. Signals are educational screening outputs, not price forecasts or personalized investment advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload `app.py`, `requirements.txt`, and `README.md` to the root of a GitHub repository, then deploy `app.py` through Streamlit Community Cloud.


## BUY / SELL methodology

The recommendations tab uses an explicit two-part decision rule. BUY is based only on the combined GPA, BTM and CFOA rank. STA/SNOA are used to describe accrual-quality concerns in SELL explanations and never override BUY eligibility. Companies outside the selected BUY threshold receive SELL to maintain only two labels, with a distinct reason and separate accrual-flag context. Altman Z-score is a separate distress-review flag and does not itself trigger SELL or BUY. Filters recalculate ranks and flags, and the app provides written cases for the top BUY and SELL names. These are educational screening outputs, not personalized investment advice.


**Portfolio note:** The Recommended Portfolio tab lists companies that survive all three screening stages in rank order. It does not assign target weights or recommend how much money to allocate to any company.
