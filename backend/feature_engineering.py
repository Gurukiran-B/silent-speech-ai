import pandas as pd

print("=" * 60)
print("      SILENT SPEECH FEATURE ENGINEERING")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

try:
    df = pd.read_csv("data.csv")

    if list(df.columns) != ["height", "width", "label"]:
        df = pd.read_csv(
            "data.csv",
            header=None,
            names=["height", "width", "label"]
        )

except:
    df = pd.read_csv(
        "data.csv",
        header=None,
        names=["height", "width", "label"]
    )

print(f"\nLoaded {len(df)} samples")

# ==========================================
# Remove Duplicates
# ==========================================

duplicates = df.duplicated().sum()

print("Duplicate Samples :", duplicates)

df = df.drop_duplicates()

print("Remaining Samples :", len(df))

# ==========================================
# Feature Engineering
# ==========================================

print("\nGenerating New Features...")

# Avoid division by zero
EPSILON = 1e-6

# Mouth Aspect Ratio
df["mar"] = df["height"] / (df["width"] + EPSILON)

# Width / Height Ratio
df["ratio"] = df["width"] / (df["height"] + EPSILON)

# Mouth Area
df["area"] = df["height"] * df["width"]

# Normalized Height
df["height_norm"] = df["height"] / df["height"].max()

# Normalized Width
df["width_norm"] = df["width"] / df["width"].max()

# ==========================================
# Display Sample
# ==========================================

print("\nFirst 10 Samples\n")

print(df.head(10))

# ==========================================
# Statistics
# ==========================================

print("\nDataset Statistics\n")

print(df.describe())

# ==========================================
# Save Dataset
# ==========================================

OUTPUT = "enhanced_data.csv"

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("Enhanced Dataset Saved Successfully")
print("=" * 60)

print(f"\nSaved As : {OUTPUT}")

print("\nNew Features Added")

print("✓ Height")
print("✓ Width")
print("✓ MAR")
print("✓ Width/Height Ratio")
print("✓ Mouth Area")
print("✓ Normalized Height")
print("✓ Normalized Width")

print("\nTotal Features :", len(df.columns)-1)