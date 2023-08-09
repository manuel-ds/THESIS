import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd

csv_file = 'dow_emb_price.csv'

df = pd.read_csv(csv_file, sep=';')

# Display the principal components DataFrame
index = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'holiday', 'bridge day']

df = df.reset_index(drop=True)  # Resetting index to numeric indices
df.index = index  # Assigning the desired index

from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=3)  # Specify the desired number of components
principal_components = pca.fit_transform(df)

# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

principal_df = principal_df.reset_index(drop=True)  # Resetting index to numeric indices
principal_df.index = index  # Assigning the desired index


import numpy as np
import matplotlib.pyplot as plt

# Reference point coordinates
reference_point = np.array([0, 0, 0])

# Calculate Euclidean distances
distances = np.linalg.norm(principal_components - reference_point, axis=1)

# Normalize distances to [0, 1] range
normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

# Create colormap
cmap = plt.cm.get_cmap('viridis')

import seaborn as sns
#sns.set_theme(style='darkgrid')
# Create a 3D plot
sns.set(rc={'figure.facecolor':(0,0,0,0), 'axes.facecolor':(0,0,0,0)})

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
#define seaborn background colors

# Increase point size
point_size = 300

# Plot the data points with colored markers and increased point size
sc = ax.scatter3D(principal_df['PC1'], principal_df['PC2'], principal_df['PC3'], c=normalized_distances, cmap=cmap, s=point_size, 
                  )
# Add colorbar
cbar = fig.colorbar(sc)
cbar.set_label('Normalized Distance')

# Add labels for each point
for index, row in principal_df.iterrows():
    ax.text(row['PC1'], row['PC2'], row['PC3'], index, size=15, zorder=10, color='k')

# Set labels for each axis
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Adjust the padding around the plot
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.tight_layout()
def rotate(angle):
    ax.view_init(azim=angle)

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
rot_animation.save('rotation.gif', dpi=80, writer='imagemagick') 
plt.show()
