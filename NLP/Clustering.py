
# # NLP Clustering Algorithm

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

# #### Preprocessing function
# * Reducing inflection in words to their root form
# * Removing punctuation
# * Everything in lowercase
# * Separating the sentence into a list of words

# %%


def text_process(text):
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc])
    nopunc = [word.lower() for word in nopunc.split()
              if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]


# #### Vectorizing embedded words

# %%
def vectorizer(sent, m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
                vec = np.add(vec, m[w])
            numw += 1
        except:
            pass

    return np.asarray(vec) / numw


# ## Preprocessing Data

# #### Reading list of raised issues - inspired by petition raised on government platform

# %%
sentences = pd.read_csv("data.csv")

# %%
sentences

# %%
splitSent = []
for i in range(len(sentences)):
    splitSent.append(text_process(sentences["strings"][i]))
print(splitSent)


# #### Word embedding function

# %%
m = Word2Vec(splitSent, size=50, min_count=1, sg=1)

l = []
for i in splitSent:
    l.append(vectorizer(i, m))

X = np.array(l)


# ## Clustering Sentences

# ### Elbow method to determine number of clusters

# %%

wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #### Clustering similar sentences

# %%
n_cluster = 4
clf = KMeans(n_clusters=n_cluster, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)
print(labels)
for index, sentence in enumerate(splitSent):
    print(str(labels[index]) + " : " + str(sentence))


# #### Using Principal component analysis to reduce dimensinality to plot clusters

# %%
pca = PCA(n_components=n_cluster).fit(X)
coords = pca.transform(X)
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E",
                "#CCCCCC", "#D2CA0D", "#522A64", "#A3DV05", "#FC6514"]
colors = [label_colors[i] for i in labels]
plt.scatter(coords[:, 0], coords[:, 1], c=colors)
centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1],
            marker='X', s=200, linewidths=2, c="#444d61")
plt.show()
