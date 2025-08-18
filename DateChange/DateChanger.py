# Code to generate random dates for transactions
# This script generates random dates for a set of transactions
# and saves the updated DataFrame to a new CSV file.

import pandas as pd
import numpy as np

df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Define how many days you want
days = 1100  # ~3 years of transactions

# Randomly assign each transaction to one of those days
df["Date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
    np.random.randint(0, days, size=len(df)), unit="D"
)

df.to_csv("personal_finance_with_dates_changed.csv", index=False)
