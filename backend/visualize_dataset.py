import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Dataset
# -------------------------------

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

sns.set_style("whitegrid")

print("Generating Visualizations...")

# -----------------------------------
# 1. Scatter Plot
# -----------------------------------

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="width",
    y="height",
    hue="label",
    s=70
)

plt.title("Height vs Width")
plt.xlabel("Width")
plt.ylabel("Height")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# -----------------------------------
# 2. Height Distribution
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["height"],
    bins=30,
    kde=True
)

plt.title("Height Distribution")
plt.tight_layout()
plt.savefig("height_distribution.png")
plt.show()

# -----------------------------------
# 3. Width Distribution
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["width"],
    bins=30,
    kde=True
)

plt.title("Width Distribution")
plt.tight_layout()
plt.savefig("width_distribution.png")
plt.show()

# -----------------------------------
# 4. Box Plot
# -----------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="label",
    y="height"
)

plt.title("Height by Word")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("height_boxplot.png")
plt.show()

# -----------------------------------
# 5. Correlation Matrix
# -----------------------------------

plt.figure(figsize=(5, 4))

corr = df[["height", "width"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

print("\nVisualization Complete")