import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = pd.read_csv('COVID19_line_list_data.csv')

# Select the columns you want to plot (assuming 'column1' and 'column2' are the names of the columns)
x = dataset['column1']
y = dataset['column2']

# Plot the dataset using scatter plot
plt.scatter(x, y)
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Dataset Scatter Plot')
plt.show()

# Select the columns for PCA (assuming you want to use 'column1' and 'column2' for PCA)
pca_data = dataset[['column1', 'column2']]

# Perform PCA
pca = PCA(n_components=2)  # Specify the number of components you want to keep (e.g., 2 for 2D projection)
pca_result = pca.fit_transform(pca_data)

# Plot the PCA result
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.show()


