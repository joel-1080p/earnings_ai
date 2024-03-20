# Quarterly Earnings AI

### HOW IT WORKS
Earnings AI takes historical quarterly earning ratios and uses them as input for the sklearn nural network.

### WEB SCRAPING
When scraping and organizing the data, I tried to stick to companies that operated similarly.
In this instance, I used the Information Technology sector.
I would scrape the ratios  day after the earnings report and compaire the underlining stock price to the day before the next earnings report.
If the underlining stock price had gone up more than 10%, the target value would be set 1. If not, then it would be set to 0.

### The Dataset
`dataset.csv` consists of the following quarterly earnigns ratios 
EP, Price Book, ROE, ROA,
Debt to Equity, Gross Margin, Operating Margin,
Current Ratio, Quick Ratio,
Price FCF, EPS, Book Value per Share,
Intrest Coverage, Asset Turnover, Debt Asset

The final value (1 or 0) would be if the underlining stock price had gone up 10% or more.

### P.S.

Please drop me a note with any feedback you have.

**Joel**
