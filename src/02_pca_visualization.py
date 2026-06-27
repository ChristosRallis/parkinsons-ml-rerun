import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# Paths
# =========================
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "parkinsons.data"
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# Separate features & labels
# =========================
X = df.drop(columns=["name", "status"])
y = df["status"]

# =========================
# Standardization
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# PCA (2 components)
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# Explained variance
# =========================
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# =========================
# Visualization
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    alpha=0.7,
    label="Healthy"
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    alpha=0.7,
    label="Parkinson"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Parkinson's Voice Dataset")
plt.legend()
plt.grid(True)

# =========================
# Save figure (for thesis)
# PCA was used solely for exploratory visualization and not for model training.
# =========================
plt.savefig(
    RESULTS_PATH / "pca_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
