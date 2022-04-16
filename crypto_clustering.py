#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[1]:


# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ### Deliverable 1: Preprocessing the Data for PCA

# In[2]:


# Load the crypto_data.csv dataset.
crypto_df = pd.read_csv('crypto_data.csv')
crypto_df.set_index('Unnamed: 0', inplace=True)
crypto_df.head(10)


# In[3]:


# Keep all the cryptocurrencies that are being traded.
crypto_df = crypto_df[crypto_df.IsTrading == True]
crypto_df.head(10)


# In[4]:


# Keep all the cryptocurrencies that have a working algorithm.
crypto_df.dropna(subset=['Algorithm'])
crypto_df.head(10)


# In[5]:


# Remove the "IsTrading" column. 
crypto_df = crypto_df.drop(columns=['IsTrading'])
crypto_df.head(10)


# In[6]:


# Remove rows that have at least 1 null value.
crypto_df = crypto_df.dropna()


# In[7]:


crypto_df.head()


# In[8]:


# Keep the rows where coins are mined.
crypto_df = crypto_df[crypto_df.TotalCoinsMined > 0]
crypto_df.head(10)


# In[10]:


# Create a new DataFrame that holds only the cryptocurrencies names.
clean_crypto = crypto_df[['CoinName']].copy()
clean_crypto


# In[11]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_df = crypto_df.drop(columns=['CoinName'])


# In[12]:


crypto_df.head(10)


# In[13]:


# Use get_dummies() to create variables for text features.
X = crypto_df.copy()
X = pd.get_dummies(X)
X.head(10)


# In[14]:


# Standardize the data with StandardScaler().
scaled_crypto = StandardScaler().fit_transform(X)
print(scaled_crypto[0:5])


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[15]:


# Using PCA to reduce dimension to three principal components.
pca = PCA(n_components=3)


# In[16]:


# Create a DataFrame with the three principal components.
crypto_pca = pca.fit_transform(scaled_crypto)


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[ ]:


# Create an elbow curve to find the best value for K.
# YOUR CODE HERE


# Running K-Means with `k=4`

# In[ ]:


# Initialize the K-Means model.
# YOUR CODE HERE

# Fit the model
# YOUR CODE HERE

# Predict clusters
# YOUR CODE HERE


# In[ ]:


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
# YOUR CODE HERE

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
# YOUR CODE HERE

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
# YOUR CODE HERE

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[ ]:


# Creating a 3D-Scatter with the PCA data and the clusters
# YOUR CODE HERE


# In[ ]:


# Create a table with tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Print the total number of tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
# YOUR CODE HERE

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
# YOUR CODE HERE

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
# YOUR CODE HERE

plot_df.head(10)


# In[ ]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
# YOUR CODE HERE


# In[ ]:




