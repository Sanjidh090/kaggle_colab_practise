import numpy as np
from gensim.models import Word2Vec

# Sample sentence
sentence = ['machine', 'learning', 'is', 'fascinating']

# Create the Word2Vec model
sentences = [['machine', 'learning', 'is', 'fascinating']]  # A list of sentences for training
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Get word vectors for each word in the sentence
word_vectors = [model.wv[word] for word in sentence]

# Calculate the sentence embedding by averaging word vectors
sentence_embedding = np.mean(word_vectors, axis=0)

# Print the resulting sentence embedding
print(sentence_embedding)
