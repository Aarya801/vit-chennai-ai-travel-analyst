# Interview Cheat Sheet

**Why CatBoost?** Mixed categorical + numeric data, nonlinear interactions, native categorical handling.

**Why log1p(price)?** Price is right-skewed and has extreme values; log scale makes training less dominated by outliers.

**Why MAE?** It is interpretable in rupees.

**Why R²?** It measures explained variance, but it is reported alongside absolute-error metrics.

**Why held-out test data?** To measure performance on a later departure-date partition. Here this is only a calendar-time proxy: without fare quote timestamps it does not establish generalization to unseen booking times.

**Hardest part?** Data cleaning: inconsistent duration, stops and passenger-count representations plus missing values.

**Why not guarantee cheapest booking?** Historical data cannot guarantee future fares.

**What would you improve?** Normalize airport aliases, add richer time features, time-aware validation, uncertainty intervals and route-level forecasting.
