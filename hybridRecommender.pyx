import random
import numpy as np
from MatrixFactorization.MatrixFactorization_RMSE import IALS_numpy

cdef class Recommender(object):

    cdef double a, b, c , d, e, maxTopPop
    cdef int cont
    cdef int[:] userbasedSim_indices, userbasedSim_indptr, contentSim_indices, contentSim_indptr, graphSim_indices, graphSim_indptr, collaborativeSim_indices, collaborativeSim_indptr, slimSim_indices, slimSim_indptr
    cdef double[:] userbasedSim_data, contentSim_data, graphSim_data, slimSim_data, collaborativeSim_data, topPOP
    cdef int[:] URM_transpose_indices, URM_transpose_indptr
    cdef double[:] durations
    cdef object builder, UCF_similarity, URM_train, MF_recommender
    cdef str alg
    def __init__(self, contentSimilarity, collaborativeSimilarity, userbasedSimilarity, slimSimilarity, graphSimilarity, a, b, c, d, e, builder, alg):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.collaborativeSim_indices = np.array(collaborativeSimilarity.indices, dtype=np.int32)
        self.collaborativeSim_indptr = np.array(collaborativeSimilarity.indptr, dtype=np.int32)
        self.collaborativeSim_data = np.array(collaborativeSimilarity.data, dtype=np.double)
        self.userbasedSim_indices = np.array(userbasedSimilarity.indices, dtype=np.int32)
        self.userbasedSim_indptr = np.array(userbasedSimilarity.indptr, dtype=np.int32)
        self.userbasedSim_data = np.array(userbasedSimilarity.data, dtype=np.double)
        self.contentSim_indices = np.array(contentSimilarity.indices, dtype=np.int32)
        self.contentSim_indptr = np.array(contentSimilarity.indptr, dtype=np.int32)
        self.contentSim_data = np.array(contentSimilarity.data, dtype=np.double)
        self.graphSim_indices = np.array(graphSimilarity.indices, dtype=np.int32)
        self.graphSim_indptr = np.array(graphSimilarity.indptr, dtype=np.int32)
        self.graphSim_data = np.array(graphSimilarity.data, dtype=np.double)
        self.slimSim_indices = np.array(slimSimilarity.indices, dtype=np.int32)
        self.slimSim_indptr = np.array(slimSimilarity.indptr, dtype=np.int32)
        self.slimSim_data = np.array(slimSimilarity.data, dtype=np.double)
        self.URM_transpose_indices = np.array(builder.URM_train_transpose.indices, dtype=np.int32)
        self.URM_transpose_indptr = np.array(builder.URM_train_transpose.indptr, dtype=np.int32)
        self.cont = -1
        self.builder = builder
        self.alg = alg
        self.UCF_similarity = userbasedSimilarity
        self.URM_train = self.builder.get_URM_train()
        self.calculate_TopPop()
        self.durations = self.builder.get_durations().astype(np.double)
        self.MF_recommender = IALS_numpy(num_factors=350,
                                 reg=0.001,
                                 iters=20,
                                 scaling='log',
                                 alpha=40,
                                 epsilon=1.0,
                                 init_mean=0.0,
                                 init_std=0.1,
                                 rnd_seed=42)
        self.MF_recommender.fit(self.URM_train)

    cdef calculate_TopPop(self):
        self.topPOP = np.zeros(self.URM_transpose_indptr.size, dtype=np.double)
        cdef int k
        self.maxTopPop = 0
        for k in range(self.URM_transpose_indptr.size-1):
            self.topPOP[k] = self.URM_transpose_indptr[k+1]-self.URM_transpose_indptr[k]
            self.maxTopPop = max(self.maxTopPop, self.topPOP[k])
        for k in range(self.URM_transpose_indptr.size-1):
            self.topPOP[k] = self.topPOP[k]/self.maxTopPop

    #For the algorithms that have a similarity matrix as an output of the model (everyone except MF and User-Based, for which are consider directly all the scores given in output from their model), the recommendation phase is managed passing its similarity matrix and its weight
    cdef calculate_rankings(self, int[:] matrix_indices, int[:] matrix_indptr, double[:] matrix_data, int[:] tracks, double weight):
        cdef set tracksSet = set(tracks)
        cdef dict best = {}
        cdef double temp = 1
        cdef double minimum = 0
        cdef double avg = 0
        cdef double q = 1/(len(tracks)+1)**1.05#1.1 #1.15
        cdef int row_start
        cdef int row_end
        cdef int[:] similarTracks
        cdef double[:] similarityValues
        cdef int track
        cdef int lenSimTracks
        cdef int i

       #The for loop is done in reverse order to give more relevance to the last songs contained in the playlist. The decrement for the first 5000 ordered playlists is managed through the use of the temp variable.
        for track in tracks[::-1]: #Loop on each track in the playlist
            row_start = matrix_indptr[track]
            row_end = matrix_indptr[track+1]
            similarTracks = matrix_indices[row_start:row_end]
            similarityValues = matrix_data[row_start:row_end]
            lenSimTracks = len(similarTracks)
            for i in range(0, lenSimTracks):#Loop on the top-k most similar tracks of track 
                if not similarTracks[i] in tracksSet:
                    if similarTracks[i] in best: 
                        best[similarTracks[i]]=best[similarTracks[i]]-similarityValues[i]*(temp/(1+self.topPOP[track])**1.2)*(1/(1+abs(self.durations[track]-self.durations[similarTracks[i]])))**0.015
                    else:
                        best[similarTracks[i]]=-1*similarityValues[i]*(temp/(1+self.topPOP[track])**1.2)*(1/(1+abs(self.durations[track]-self.durations[similarTracks[i]])))**0.015
                    avg = avg - similarityValues[i]*(temp/(1+self.topPOP[track])**1.2)*(1/(1+abs(self.durations[track]-self.durations[similarTracks[i]])))**0.015
                    minimum = min(minimum, best[similarTracks[i]])
            if self.cont < 5000:
                temp -= q
        if len(best) > 0:
            avg = avg/len(best)
        #Normalization 
        for i in best:
            best[i] = weight*best[i]/minimum*(-1)+weight*avg*0.9
        return best
    
    cdef filter_seen(self, user_id, scores):
        
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id+1]
        user_profile = self.URM_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    cdef userbased_calculate_ratings(self, int playlist, dict tracks, int[:] matrix_indices, int[:] matrix_indptr, double[:] matrix_data, double weight):
        similar_playlists = self.UCF_similarity[playlist]
        scores = similar_playlists.dot(self.URM_train).toarray().ravel()
        scores = self.filter_seen(playlist, scores)
        #ranking = scores.argsort()[::-1]
        cdef int k
        cdef double minimum = 0, avg = 0
        cdef dict best = {}
        for k in tracks:
            minimum = min(minimum, -1*scores[k])
            avg = avg - scores[k]
        avg = avg/len(tracks)
        for k in tracks:
            if minimum != 0:
                best[k] = weight*(-1)*(scores[k]/minimum*(-1))+avg*weight*1.2#1.3
        return best

    def final_score(self, dict best):
        preSorted = [[v, k] for k,v in best.items()]
        best1 = np.empty((max(11,len(preSorted)), 2), dtype=object)
        cdef int i
        for i in range(len(preSorted)):
            best1[i] = preSorted[i]
        if len(preSorted) < 11:
            for i in range(len(preSorted), 11):
                best1[i] = [0, random.randint(0, 20000)]
        best1 = best1[best1[:,0].argpartition(10)][0:10]
        best1 = best1[best1[:,0].argsort()][:,1]
        return best1

    def recommend_CBF(self, playlist):

        cdef int[:] tracks = self.builder.get_tracks_inside_playlist_train(playlist)
        cdef dict content_ratings = self.calculate_rankings(self.contentSim_indices, self.contentSim_indptr, self.contentSim_data, tracks, self.a)
        return self.final_score(content_ratings)
        
    def recommend_GRAPH(self, playlist):

        cdef int[:] tracks = self.builder.get_tracks_inside_playlist_train(playlist)
        cdef dict graph_ratings = self.calculate_rankings(self.graphSim_indices, self.graphSim_indptr, self.graphSim_data, tracks, self.b)
        return self.final_score(graph_ratings)
       
    def recommend_UCF(self, playlist):
        pass

    def recommend_SLIM(self, playlist):

        cdef int[:] tracks = self.builder.get_tracks_inside_playlist_train(playlist)
        cdef dict slim_ratings = self.calculate_rankings(self.slimSim_indices, self.slimSim_indptr, self.slimSim_data, tracks, self.d)
        return self.final_score(slim_ratings)

    def recommend_MF(self, playlist):
        self.recommend_MF.recommend(playlist, 10)

    #The return value of each calculate_rankings is a dictionary that contains the scores of the first top-k recommendations for each algorithm (top-k different for each algorithm)
    def recommend_HYBRID(self, playlist):
        
        cdef int[:] tracks = self.builder.get_tracks_inside_playlist_train(playlist)
        cdef dict content_ratings = self.calculate_rankings(self.contentSim_indices, self.contentSim_indptr, self.contentSim_data, tracks, self.a)
        cdef dict graph_ratings = self.calculate_rankings(self.graphSim_indices, self.graphSim_indptr, self.graphSim_data, tracks, self.e)
        cdef dict slim_ratings = self.calculate_rankings(self.slimSim_indices, self.slimSim_indptr, self.slimSim_data, tracks, self.d)
        cdef dict collaborative_ratings = self.calculate_rankings(self.collaborativeSim_indices, self.collaborativeSim_indptr, self.collaborativeSim_data, tracks, self.b)
        cdef int k

        #At the beginning graph_ratings contains only the scores computed from the graph based algorithm, then those computed from the slim algorithm are added in case of already existings recommendations, otherwise they are simply inserted from scratch
        for k in slim_ratings:
            if k in graph_ratings:
                graph_ratings[k] = graph_ratings[k] + slim_ratings[k]
            else: #
                graph_ratings[k] = slim_ratings[k] #

        #From here the scores computed from other algorithms are added only in case of already existings recommendations
        for k in content_ratings:
            if k in graph_ratings:
                graph_ratings[k] = graph_ratings[k] + content_ratings[k]

        for k in collaborative_ratings:
            if k in graph_ratings:
                graph_ratings[k] = graph_ratings[k] + collaborative_ratings[k]
        
        cdef dict best = graph_ratings
        cdef dict userbased_ratings = self.userbased_calculate_ratings(playlist, best, self.userbasedSim_indices, self.userbasedSim_indptr, self.userbasedSim_data, self.c)
        for k in userbased_ratings:
            if k in best:
                best[k] = best[k] + userbased_ratings[k]
            #else:
            #    best[k] = userbased_ratings[k]

        cdef double[:] mf_scores = self.MF_recommender.getScores(playlist)
        cdef double minimum = 0, avg = 0
        for k in range(mf_scores.size):
            minimum = min(minimum, -1*mf_scores[k])
            avg = avg - mf_scores[k]
        avg = avg/mf_scores.size
 
        for k in range(mf_scores.size):
            if k in best and abs(mf_scores[k]) > abs(avg):
                best[k] = best[k] + 0.28*(-1)*(mf_scores[k]/minimum*(-1))+avg*2.5#*1.3
                

        return self.final_score(best)


    def recommend(self, playlist):

        self.cont = self.cont + 1
        if self.cont % 1000 == 0:
            print(str(self.cont))
        
        if self.alg == "HYBRID":
            return self.recommend_HYBRID(playlist)

        elif self.alg == "CBF":
            return self.recommend_CBF(playlist)

        elif self.alg == "ICF":
            return self.recommend_ICF(playlist)

        elif self.alg == "UCF":
            return self.recommend_UCF(playlist)

        elif self.alg == "SLIM":
            return self.recommend_SLIM(playlist)

        elif self.alg == "GRAPH":
            return self.recommend_GRAPH(playlist)

        elif self.alg == "MF":
            return self.recommend_MF(playlist)

        

