# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# STEP 2: LOAD DATASET
data = pd.read_csv("C:\\Users\\Priyadharshini R\\Desktop\\archive-mall customers\\Mall_Customers.csv")

print("Mall Customer Segmentation using Unsupervised Machine Learning")
print("Dataset Loaded Successfully!\n")
print(data.head())

# STEP 3: SELECT FEATURES
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# STEP 4: FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# STEP 5: ELBOW METHOD
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss,color='lightgreen',marker='o',markerfacecolor='darkblue',markeredgecolor='darkblue',linestyle='-',linewidth=2)
plt.title("Elbow Method for Optimal K", fontsize=14)
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("WCSS (Inertia)", fontsize=12)
plt.grid(True)
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()


# STEP 6: APPLY K-MEANS
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans

# STEP 7: SILHOUETTE SCORE
score = silhouette_score(X_scaled, y_kmeans)
print(f"\nSilhouette Score: {score:.3f}")


# STEP 8: CLUSTER VISUALIZATION
plt.figure()
plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=y_kmeans,
    cmap='viridis'
)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()



# STEP 9: CLUSTER ANALYSIS
print("\n Cluster Summary (Mean Values):\n")
cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)


print("\n Number of Customers in Each Cluster:\n")
print(data['Cluster'].value_counts())



# STEP 10: SMART SEGMENT LABELING
def label_cluster(row):
    income = row['Annual Income (k$)']
    score = row['Spending Score (1-100)']

    if income > 70 and score > 60:
        return "High Value Customers"
    elif income > 70 and score <= 40:
        return "Rich but Low Spenders"
    elif income <= 40 and score > 60:
        return "Low Income High Spenders"
    elif income <= 40 and score <= 40:
        return "Low Value Customers"
    else:
        return "Average Customers"


data['Segment'] = data.apply(label_cluster, axis=1)


print("\n Sample Segmented Data:\n")
print(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Segment']].head())




# STEP 11: SEGMENT DISTRIBUTION
plt.figure(figsize=(8,5))

data['Segment'] = data['Segment'].replace({
    "High Value Customers": "High Value",
    "Rich but Low Spenders": "Rich Low Spend",
    "Low Income High Spenders": "Low Inc High Spend",
    "Low Value Customers": "Low Value",
    "Average Customers": "Average"
})


segment_counts = data['Segment'].value_counts()

import numpy as np
colors = ['#A8DADC', '#FFD6A5', '#CAFFBF', '#FFC6FF', '#BDB2FF']

bars = plt.bar(segment_counts.index, segment_counts.values, color=colors)

plt.title("Customer Segment Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Segment")
plt.ylabel("Count")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,height,f'{int(height)}',ha='center',va='bottom' )
plt.tight_layout()
plt.show()



# STEP 12: CENTROIDS VISUALIZATION

centroids = kmeans.cluster_centers_
plt.figure()

plt.scatter(
    X_scaled[:, 1],
    X_scaled[:, 2],
    c=y_kmeans,
    cmap='viridis'
)

plt.scatter(
    centroids[:, 1],
    centroids[:, 2],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)

plt.legend()
plt.title("Clusters with Centroids (Scaled)")
plt.xlabel("Scaled Income")
plt.ylabel("Scaled Spending Score")
plt.show()



# STEP 13: NEW CUSTOMER PREDICTION

def predict_customer(age, income, score):
    new_data = scaler.transform([[age, income, score]])
    cluster = kmeans.predict(new_data)[0]

    # Apply same logic for labeling
    if income > 70 and score > 60:
        segment = "High Value Customers"
    elif income > 70 and score <= 40:
        segment = "Rich but Low Spenders"
    elif income <= 40 and score > 60:
        segment = "Low Income High Spenders"
    elif income <= 40 and score <= 40:
        segment = "Low Value Customers"
    else:
        segment = "Average Customers"

    print("\nNew Customer Details:")
    print(f"Age: {age}, Income: {income}, Score: {score}")
    print(f" Predicted Cluster: {cluster}")
    print(f" Segment: {segment}")




# STEP 14: TESTING
predict_customer(30, 60, 70)
predict_customer(50, 30, 20)
predict_customer(25, 90, 85)