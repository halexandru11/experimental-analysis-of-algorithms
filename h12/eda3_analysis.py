import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns


DATA_FILE = Path("EDA3.csv")
OUTPUT_DIR = Path("plots")
TARGET_VAR = "Strength"
X_VAR = "Age"
Y_VAR = "Strength"


def configure_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def print_general_overview(df: pd.DataFrame) -> None:
    print("1. GENERAL OVERVIEW")
    print("\nHead of dataset:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nInfo:")
    print(df.info())

    print("\nDescribe (numeric columns):")
    print(df.describe())

    print("\nCount of each category in Sex:")
    print("Non-null count:", df["Sex"].count())
    print(df["Sex"].value_counts())

    print("\nCount of each category in Party:")
    print("Non-null count:", df["Party"].count())
    print(df["Party"].value_counts())

    print(f"\nDrop NA example for {TARGET_VAR}:")
    print(df[TARGET_VAR].dropna())

    print("\nStatistics per group:")
    print("Average strength by party:")
    print(df.groupby("Party")[TARGET_VAR].mean())
    print("\nAverage strength by sex and party:")
    print(df.groupby(["Sex", "Party"])[TARGET_VAR].mean())


def save_strength_distribution_plots(series: pd.Series) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=10, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title("Histogram of Strength")
    plt.xlabel("Strength")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strength_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot(series)
    plt.title("Boxplot of Strength")
    plt.ylabel("Strength")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strength_boxplot.png", dpi=150)
    plt.close()


def analyze_strength_distribution(df: pd.DataFrame) -> None:
    print("\n2. DISTRIBUTION ANALYSIS FOR Strength")
    series = df[TARGET_VAR].dropna()

    print("Mean:", series.mean())
    print("Median:", series.median())
    print("Mode:", series.mode().tolist())
    print("Variance:", series.var())
    print("Standard deviation:", series.std())
    print("Range:", series.max() - series.min())
    print("Interquartile range:", series.quantile(0.75) - series.quantile(0.25))
    print("Skewness:", series.skew())
    print("Kurtosis:", series.kurt())

    save_strength_distribution_plots(series)


def save_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_strength_heatmap.png", dpi=150)
    plt.close()


def save_age_strength_scatter_plot(pair_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(pair_df[X_VAR], pair_df[Y_VAR], color="darkred", alpha=0.8)
    plt.title("Scatter Plot of Age vs Strength")
    plt.xlabel("Age")
    plt.ylabel("Strength")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_strength_scatter.png", dpi=150)
    plt.close()


def analyze_age_strength_relationship(df: pd.DataFrame) -> None:
    print("\n3. COVARIANCE AND CORRELATION BETWEEN Age AND Strength")
    pair_df = df[[X_VAR, Y_VAR]].dropna()

    print("\nCovariance matrix:")
    print(pair_df.cov())

    print("\nCorrelation matrix:")
    corr_matrix = pair_df.corr()
    print(corr_matrix)

    corr_coef, p_value = pearsonr(pair_df[X_VAR], pair_df[Y_VAR])
    print(f"\nPearson correlation coefficient between {X_VAR} and {Y_VAR}: {corr_coef}")
    print("P-value:", p_value)

    save_correlation_heatmap(corr_matrix)
    save_age_strength_scatter_plot(pair_df)


def main() -> None:
    configure_environment()
    df = load_data()
    print_general_overview(df)
    analyze_strength_distribution(df)
    analyze_age_strength_relationship(df)

    print("\nPlots saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
