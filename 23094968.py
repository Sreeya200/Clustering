import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
df = pd.read_csv('/WineQT.csv')

# Preview the dataset
print("Dataset Head:")
print(df.head())

# Check for missing values
df = df.dropna()

# Select features for clustering
X = df[['fixed acidity', 'volatile acidity', 'citric acid']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Plot for Wine Quality Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means clustering with optimal clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Scatter plot for Clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['fixed acidity'], df['volatile acidity'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Wine Quality Clustering')
plt.xlabel('Fixed Acidity')
plt.ylabel('Volatile Acidity')
plt.colorbar(label='Cluster')
plt.show()

# Linear regression to predict quality based on alcohol
X_reg = df[['alcohol']]
y = df['quality']

# Linear regression model
regressor = LinearRegression()
regressor.fit(X_reg, y)

# Add predictions to the dataset
df['FittedLine'] = regressor.predict(X_reg)

# Plot actual vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(df['alcohol'], df['quality'], color='blue', alpha=0.5, label='Actual')
plt.plot(df['alcohol'], df['FittedLine'], color='red', label='Fitted Line')
plt.title('Linear Fitting: Quality vs Alcohol')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.legend()
plt.show()

# Correlation heatmap
correlation_matrix = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'quality']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Wine Quality')
plt.show()
