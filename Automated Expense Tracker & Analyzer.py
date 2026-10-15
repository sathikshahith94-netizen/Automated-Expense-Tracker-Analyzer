import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =====================================================================
# 1. Define standard transaction schema & rule-based categorization
# =====================================================================

DEFAULT_CATEGORIES = {
    "groceries": ["walmart", "kroger", "target", "safeway", "whole foods"],
    "utilities": ["electric", "water", "gas", "internet", "comcast", "verizon"],
    "dining": ["starbucks", "mcdonalds", "subway", "restaurant", "cafe", "pizza"],
    "transportation": ["uber", "lyft", "gas station", "shell", "chevron", "subway transit"],
    "entertainment": ["netflix", "spotify", "hulu", "cinema", "disney"],
    "income": ["salary", "paycheck", "direct deposit", "venmo cashout"]
}

def categorize_transaction(description, category_rules=DEFAULT_CATEGORIES):
    """Rule-based categorization based on keywords in the description."""
    if not isinstance(description, str):
        return "unassigned"
    desc_lower = description.lower()
    for category, keywords in category_rules.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category
    return "other"

# =====================================================================
# 2. Data Loading & Normalization
# =====================================================================

def load_and_normalize_expenses(csv_path, category_rules=DEFAULT_CATEGORIES):
    """Loads, cleans, and normalizes bank CSV transactions to standard schema."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Normalize columns to lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Map dynamic input columns to standard schema:
    # Required standard schema: date, description, amount, category, account
    rename_map = {}
    for col in df.columns:
        if "date" in col:
            rename_map[col] = "date"
        elif "desc" in col or "merchant" in col or "details" in col:
            rename_map[col] = "description"
        elif "amt" in col or "amount" in col or "price" in col:
            rename_map[col] = "amount"
        elif "acc" in col or "card" in col:
            rename_map[col] = "account"

    df = df.rename(columns=rename_map)

    # Fill missing standard columns if they don't exist
    if "account" not in df.columns:
        df["account"] = "default_account"

    # Ensure critical columns exist
    for col in ["date", "description", "amount"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column mapping for: '{col}' in CSV.")

    # Convert and clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()

    # Remove incomplete entries
    df = df.dropna(subset=["date", "amount"]).copy()

    # Categorize automatically
    df["category"] = df["description"].apply(lambda d: categorize_transaction(d, category_rules))

    return df

# =====================================================================
# 3. Monthly Spending Insights & Report Generation
# =====================================================================

def generate_expense_insights(df, monthly_budget=2000):
    """Calculates monthly totals, category shares, recurring expenses, and budget variance."""
    df["year_month"] = df["date"].dt.to_period("M")

    # Separate spending (negative/positive values representing outflow vs income)
    # Assuming negative numbers or explicitly categorized expenses are spending
    expenses_df = df[df["category"] != "income"].copy()
    expenses_df["amount_abs"] = expenses_df["amount"].abs()

    print("=== Spend Summary by Month ===")
    monthly_totals = expenses_df.groupby("year_month")["amount_abs"].sum().reset_index()
    monthly_totals["budget"] = monthly_budget
    monthly_totals["variance"] = monthly_totals["budget"] - monthly_totals["amount_abs"]
    print(monthly_totals.to_string(index=False))
    print("\n")

    print("=== Category Distribution ===")
    category_share = expenses_df.groupby("category")["amount_abs"].sum().reset_index()
    total_spend = category_share["amount_abs"].sum()
    category_share["percentage_share"] = (category_share["amount_abs"] / total_spend * 100).round(2)
    print(category_share.to_string(index=False))
    print("\n")

    # Simple heuristic to identify potential recurring transactions
    print("=== Potential Recurring Expenses ===")
    recurring = expenses_df.groupby(["description", "amount_abs"]).size().reset_index(name="frequency")
    recurring = recurring[recurring["frequency"] > 1]
    print(recurring.to_string(index=False))
    print("\n")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Convert year_month period to string format for plotting
    monthly_totals["year_month_str"] = monthly_totals["year_month"].astype(str)

    # Plot 1: Monthly Trend against Budget
    sns.barplot(ax=axes[0], data=monthly_totals, x="year_month_str", y="amount_abs", hue="year_month_str", palette="Blues_d", legend=False)
    axes[0].axhline(monthly_budget, color="red", linestyle="--", label=f"Budget Limit (${monthly_budget})")
    axes[0].set_title("Monthly Expenses vs Budget")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Total Spend ($)")
    axes[0].legend()

    # Plot 2: Category Breakdown
    axes[1].pie(category_share["amount_abs"], labels=category_share["category"], autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
    axes[1].set_title("Spending Share by Category")

    plt.tight_layout()
    plt.show()

# =====================================================================
# 4. Generate Demo CSV and Run pipeline
# =====================================================================

demo_csv_content = """Transaction Date,Description,Amount,Account
2026-01-02,Walmart grocery store,-152.50,Credit Card
2026-01-05,Netflix subscription,-15.99,Credit Card
2026-01-10,Salary Paycheck,2500.00,Checking
2026-01-15,Starbucks Coffee,-6.50,Credit Card
2026-01-20,City Electric Utility,-85.00,Checking Account
2026-01-22,Shell Gas Station,-45.00,Credit Card
2026-02-02,Walmart grocery store,-160.00,Credit Card
2026-02-05,Netflix subscription,-15.99,Credit Card
2026-02-12,Salary Paycheck,2500.00,Checking
2026-02-18,McDonalds,-12.40,Credit Card
2026-02-25,Comcast Internet,-79.99,Checking Account
"""

with open("sample_expenses.csv", "w") as f:
    f.write(demo_csv_content)

print("Generated sample_expenses.csv successfully! Starting processing pipeline...\n")
df_normalized = load_and_normalize_expenses("sample_expenses.csv")
if df_normalized is not None:
    generate_expense_insights(df_normalized, monthly_budget=500)