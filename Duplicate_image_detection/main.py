
from sklearn.cluster import AgglomerativeClustering
from keras.applications.vgg16 import VGG16
from keras.models import Model
import cv2
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

path = 'images'

images = []
with os.scandir(path) as files:  #os.scandir creates an iterable of files in directory
    for file in files:
        if not file.name.startswith('.'):
            images.append(os.path.join(path,file.name))


model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def get_features(filename,model):
    print(filename)
    image = cv2.imread(filename)
    image = cv2.resize(image,(224,224))

    features = model.predict(image.reshape(1,224,224,3))

    return features

feature_data = {}
for image in images:
    feature_data[image] = get_features(image,model)

names = np.array(list(feature_data.keys()))
features = np.array(list(feature_data.values()))
features = features.reshape(-1,4096)
'''
pca = PCA(n_components=10, random_state=23)
pca.fit(features)
reduced_features = pca.transform(features)
'''
clusters = AgglomerativeClustering(n_clusters=None,linkage='ward',distance_threshold=70,affinity='euclidean').fit(features)

result = {}
for cluster,name in zip(clusters.labels_,names):
    if cluster in result:
        result[cluster].append(name)
    else:
        result[cluster] = [name]

for key, value in result.items():
    print(key, ' : ', value)
