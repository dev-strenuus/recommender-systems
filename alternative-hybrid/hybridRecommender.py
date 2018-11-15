import random
import numpy as np
class HybridRecommender(object):

    def __init__(self, contentSimilarity, collaborativeSimilarity, a, b):
        self.bestSimilarTracks = a*contentSimilarity + b*collaborativeSimilarity
        self.cont = -1
    
    def recommend(self, playlist, builder):
        self.cont = self.cont + 1
        tracks = builder.get_tracks_inside_playlist_train(playlist)
        tracksSet = set(tracks)
        best = {}
        temp = 1
        for track in tracks:
            row_start = self.bestSimilarTracks.indptr[track]
            row_end = self.bestSimilarTracks.indptr[track+1]
            similarTracks = self.bestSimilarTracks.indices[row_start:row_end]
            similarityValues = self.bestSimilarTracks.data[row_start:row_end]
            for i in range(0, len(similarTracks)):
                if not similarTracks[i] in tracksSet:
                    if similarTracks[i] in best:
                        best[similarTracks[i]]=best[similarTracks[i]]-similarityValues[i]*temp
                    else:
                        best[similarTracks[i]]=-1*similarityValues[i]*temp
            if self.cont < 5000:
                temp -= 1/(len(tracks)+1)
        preSorted = [[v, k] for k,v in best.items()]
        best = np.empty((max(11,len(preSorted)), 2), dtype=object)
        for i in range(len(preSorted)):
            best[i] = preSorted[i]
        if len(preSorted) < 11:
            for i in range(len(preSorted), 11):
                best[i] = [0, random.randint(0, 20000)]
            print(best)
        best = best[best[:,0].argpartition(10)][0:10]
        best = best[best[:,0].argsort()][:,1]
        return best

    def recommend2(self, playlist, builder):
        # compute the scores using the dot product
        playlist_profile = builder.get_URM()[playlist]
        scores = playlist_profile.dot(self.bestSimilarTracks).toarray().ravel()
        
        scores = self.filter_seen(playlist, scores, builder)
        
        # rank items
        ranking = scores.argsort()[::-1]
        
        return ranking[:10]

    def filter_seen(self, playlist, scores, builder):
        
        start_pos = builder.get_URM()[playlist].indptr[playlist]
        end_pos = builder.get_URM()[playlist].indptr[playlist+1]
        
        user_profile = builder.get_URM()[playlist].indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf
        
        return scores
