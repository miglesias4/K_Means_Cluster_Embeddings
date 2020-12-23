#Matthew Iglesias

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
def read_embeddings(n=1000):
    # Reads n embeddings from file
    # Returns a dictionary where embedding[w] is the embeding of string w
    embedding = {}
    count = 0
    with open('glove.6B.50d.txt', encoding="utf8") as f: 
        for line in f: 
            ls = line.split(" ")
            emb = [np.float32(x) for x in ls[1:]]
            embedding[ls[0]]=np.array(emb) 
            count+=1 
            if count>= n:
                break
    return embedding

def find_similarity(a,b): #Created a find similarity function to asist in computing the similarity of words
    dist = 0.0
    dot = np.dot(a,b)
    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    sim = dot/(norm_a)/norm_b
    return sim
    
if __name__ == "__main__":  

    vocabulary_size = 30000        
    embedding = read_embeddings(vocabulary_size)
    
    # Create array X of size (vocabulary_size,50) containing embeddings
    X = []
    for val in embedding:
        X.append(embedding[val])
    
    X = np.array(X) #Shape is (30000,50)
    
    # Cluster the data 
    #model = KMeans(... 
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    pred = kmeans.predict(X)

    # Retrieve cluster centers, an array of size  (10,50)
    #centers = model.cluster_centers_ 
    centers = kmeans.cluster_centers_
    centers = np.array(centers) #convert to numpy array

    # For each c in centers, find the word whose embedding is most similar to c and print it
    #for c in centers: #for evert cluster c
    max_sim = -999
    count = 0
    for c in centers: #Iterates though each centroid, 10 in toal
        count = count +1
        for w in embedding: #Iterate through each word in embedding dict to compute embedding and most similar
            sim = find_similarity(c, embedding[w]) #Compute similarity between words in both pair and options
            if sim > max_sim:
                max_sim = sim
                closest_word = w #Award closest_word with the best similar word in cluster
                
        print('Cluster {}, the word embedding closest is {} '.format(count,closest_word))