import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../data/sample_protein_sequences.csv")

features = df[['Sequence_Length']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

similarity = cosine_similarity(scaled_features)

plt.imshow(similarity, cmap='viridis')
plt.colorbar()
plt.title("Protein Sequence Similarity Matrix")
plt.xticks(range(len(df)), df['Protein_ID'])
plt.yticks(range(len(df)), df['Protein_ID'])
plt.tight_layout()
plt.savefig("../visuals/protein_similarity_heatmap.png", dpi=300)
plt.show()
