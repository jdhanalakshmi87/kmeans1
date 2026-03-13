# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Load dataset
data = pd.read_csv("/Users/dhana/Documents/BDA/Even 25-26 T1/Lab/customers_large_dataset.csv")

# Step 3: Select two columns for plotting
X = data[["AnnualIncome", "SpendingScore"]]

# Step 4: Apply K-Means
k = 3   # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
# Step 5: Get cluster labels
labels = kmeans.labels_
# Step 6: Plot the clusters
plt.figure()
plt.scatter(X["AnnualIncome"], X["SpendingScore"], c=labels)
# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()
