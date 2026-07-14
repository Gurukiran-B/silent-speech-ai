import pandas as pd

# ==========================================
# Load Dataset
# ==========================================

try:
    # Try reading with header
    df = pd.read_csv("data.csv")

    # If expected columns are not present,
    # reload assuming no header.
    if list(df.columns) != ["height", "width", "label"]:
        df = pd.read_csv(
            "data.csv",
            header=None,
            names=["height", "width", "label"]
        )

except Exception:
    df = pd.read_csv(
        "data.csv",
        header=None,
        names=["height", "width", "label"]
    )

# ==========================================
# Convert Data Types
# ==========================================

df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["width"] = pd.to_numeric(df["width"], errors="coerce")

# ==========================================
# Dataset Overview
# ==========================================

print("\n" + "=" * 60)
print("          SILENT SPEECH DATASET ANALYSIS")
print("=" * 60)

print(f"\nTotal Samples : {len(df)}")
print(f"Features      : {list(df.columns)}")
print(f"Unique Words  : {df['label'].nunique()}")

print("\nWords Present:")
print(df["label"].unique())

print("\n" + "=" * 60)

# ==========================================
# Samples Per Word
# ==========================================

print("\nSamples Per Word\n")

print(df["label"].value_counts())

print("\n" + "=" * 60)

# ==========================================
# Missing Values
# ==========================================

print("\nMissing Values\n")

print(df.isnull().sum())

print("\n" + "=" * 60)

# ==========================================
# Duplicate Samples
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate Samples :", duplicates)

print("\n" + "=" * 60)

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Information\n")

df.info()

print("\n" + "=" * 60)

# ==========================================
# Overall Statistics
# ==========================================

print("\nOverall Statistics\n")

print(df[["height", "width"]].describe())

print("\n" + "=" * 60)

# ==========================================
# Statistics Per Word
# ==========================================

print("\nStatistics Per Word")

for word in sorted(df["label"].unique()):

    print("\n" + "-" * 50)
    print(f"WORD : {word.upper()}")
    print("-" * 50)

    subset = df[df["label"] == word]

    print(subset[["height", "width"]].describe())

print("\n" + "=" * 60)

# ==========================================
# Height Range
# ==========================================

print("\nHeight Range")

print("Minimum Height :", df["height"].min())
print("Maximum Height :", df["height"].max())

print("\nWidth Range")

print("Minimum Width :", df["width"].min())
print("Maximum Width :", df["width"].max())

print("\n" + "=" * 60)

# ==========================================
# Outlier Check
# ==========================================

print("\nSamples with Height <= 1")

small_height = df[df["height"] <= 1]

print(len(small_height))

print("\nSamples with Width <= 10")

small_width = df[df["width"] <= 10]

print(len(small_width))

print("\n" + "=" * 60)

print("\nDataset Analysis Completed Successfully")