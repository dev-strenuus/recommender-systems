from evaluation_function import evaluate_algorithm
from Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sparse
import numpy as np
class ItemCBFKNNRecommender(object):
    
    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM
    
    
    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
        topK=topK, normalize=normalize,
        similarity = similarity)

        self.W_sparse = similarity_object.compute_similarity()
        
        #print (self.W_sparse)
        sparse.save_npz('cbsim.npz', self.W_sparse, compressed=True)
        sparse_matrix = sparse.load_npz('cbsim.npz')
        #print (sparse_matrix)
        return self.W_sparse
    
    
    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        
        # rank items
        ranking = scores.argsort()[::-1]
        
        return ranking[:at]
    
    
    def filter_seen(self, user_id, scores):
        
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]
        
        user_profile = self.URM.indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf
        
        return scores
