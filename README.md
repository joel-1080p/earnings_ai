# Quarterly Earnings AI

## HOW IT WORKS
Earnings AI takes historical quarterly earning ratios and uses them as inputs for sklearn.

## WEB SCRAPING
When scraping and organizing the data, I tried to stick to companies that operated similarly.
In this instance, I used the Information Technology sector.
I would scrape the ratios the day after the earnings report along with the underlining stock price.
I then compaired that price to the day before the next earnings report.
If the underlining price had gone up more than 10%, the target value would be set 1. If not, then it would be set to 0.

### The Dataset
`dataset.csv` consists of the following quarterly earnigns ratios dating back to 2010.
- EP, Price Book, ROE, ROA,
- Debt to Equity, Gross Margin, Operating Margin,
- Current Ratio, Quick Ratio,
- Price FCF, EPS, Book Value per Share,
- Intrest Coverage, Asset Turnover, Debt Asset

The final value (the target value) would be a 1 or 0 if the underlining stock price had gone up 10% or more.

## Requirements

-   [Python](https://www.python.org)
-   [Pandas](https://github.com/pydata/pandas)
-   [sklearn](https://scikit-learn.org/stable/)
-   [matplotlib](https://matplotlib.org/)

### P.S.

Please drop me a note with any feedback you have.

**Joel**
