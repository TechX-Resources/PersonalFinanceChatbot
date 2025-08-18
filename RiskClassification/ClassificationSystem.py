# Risk Classification System
# This script classifies financial transactions based on risk factors derived from spending patterns.

import pandas as pd
import numpy as np


# Risk Metric Functions

def spending_to_income_ratio(df):
    """
    Calculates the ratio of total expenses to total income.
    
    - Uses the 'Type' column to distinguish:
        Credit = inflow (income)
        Debit = outflow (expenses)
    - A high ratio (close to or above 1) means the user is spending
      as much or more than they earn, which increases financial risk.
    """
    income = df[df['Type'] == 'Credit']['Amount'].sum()
    expenses = df[df['Type'] == 'Debit']['Amount'].sum()
    if income == 0:
        return np.inf  # Infinite risk if no income but there are expenses
    return expenses / income


def balance_volatility(df):
    """
    Measures how much the account balance fluctuates over time.
    
    - Uses the 'Balance After Transaction' column.
    - Computes the standard deviation of balances, divided by the mean.
    - A higher value means more volatility (unstable finances).
    - Example: Someone whose balance swings wildly is at higher risk.
    """
    balances = df['Balance After Transaction']
    if balances.mean() == 0:
        return 0
    return balances.std() / balances.mean()


def negative_balance_ratio(df):
    """
    Calculates the fraction of transactions that result in a negative balance.
    
    - Checks the 'Balance After Transaction' column.
    - If this ratio is high, the user frequently goes negative, which indicates poor financial health.
    """
    negatives = (df['Balance After Transaction'] < 0).sum()
    return negatives / len(df)


def large_purchase_ratio(df, threshold=0.3):
    """
    Identifies the proportion of 'large' purchases compared to average spending.
    
    - Defines a 'large purchase' as a debit transaction that is greater
      than (threshold * average debit amount).
    - Example: If average spending is $100, with threshold=0.3,
      then anything > $130 is a large purchase.
    - A high ratio suggests risky spending habits.
    """
    debits = df[df['Type'] == 'Debit']['Amount']
    if len(debits) == 0:
        return 0
    avg_spend = debits.mean()
    large_purchases = (debits > avg_spend * (1 + threshold)).sum()
    return large_purchases / len(debits)



# Risk Scoring Function


def compute_risk_score(df):
    """
    Combines multiple financial health metrics into a single risk score (0–100).
    
    Metrics included:
    - Spending-to-Income Ratio
    - Balance Volatility
    - Negative Balance Ratio
    - Large Purchase Ratio
    
    The score is a weighted average of normalized metrics.
    A higher score = higher financial risk.
    """
    ratio = spending_to_income_ratio(df)
    volatility = balance_volatility(df)
    negatives = negative_balance_ratio(df)
    large_purchases = large_purchase_ratio(df)
    
    # Normalize metrics to 0–1 (simple scaling)
    normalized_ratio = min(ratio, 2) / 2        # Cap at 2 (200% of income)
    normalized_volatility = min(volatility, 1)  # Cap at 1 (very high volatility)
    normalized_negatives = negatives            # Already a ratio 0–1
    normalized_large = large_purchases          # Already a ratio 0–1

    # Weighted average 
    # Adjusted weights based on percieved importance of each metric
    score = (
        0.35 * normalized_ratio +
        0.2 * normalized_volatility +
        0.35 * normalized_negatives +
        0.1 * normalized_large
    ) * 100
    
    return score


def assign_risk_bucket(score):
    """
    Converts a numeric risk score into a risk category.
    
    - Low Risk:    score < 33
    - Medium Risk: 33 <= score < 66
    - High Risk:   score >= 66
    
    This provides an interpretable label that can be shown to users.
    """
    if score < 33:
        return "Low Risk"
    elif score < 66:
        return "Medium Risk"
    else:
        return "High Risk"



# Example Usage:


# Load your dataset
df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Calculate risk score + bucket
score = compute_risk_score(df)
bucket = assign_risk_bucket(score)

print(f"Risk Score: {score:.2f}")
print(f"Risk Category: {bucket}")
