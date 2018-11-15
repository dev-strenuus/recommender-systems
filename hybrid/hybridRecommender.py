import random
import numpy as np
from collections import Counter
class HybridRecommender(object):

    def __init__(self, contentSimilarity, collaborativeSimilarity, a, b):
        self.contentSimilarity = contentSimilarity
        self.collaborativeSimilarity = collaborativeSimilarity
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

    def calculate_rankings(self, matrix, tracks,weight):
        tracksSet = set(tracks)
        best = {}
        temp = 1
        minimum = 0
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
                    minimum = min(minimum, best[similarTracks[i]])
            if self.cont < 5000:
                temp -= 1/(len(tracks)+1)**3
        for k in best:
            best[k] = weight*best[k]/minimum*(-1)
        return best

        


    def recommend1(self, playlist, builder):
        self.cont = self.cont + 1
        tracks = builder.get_tracks_inside_playlist_train(playlist)
        content_ratings = self.calculate_rankings(self.contentSimilarity, tracks, 0.10)
        collaborative_ratings = self.calculate_rankings(self.contentSimilarity, tracks, 1)
        content_ratings = Counter(content_ratings)
        for k in content_ratings:
            if k in collaborative_ratings:
                collaborative_ratings[k] = collaborative_ratings[k] + content_ratings[k]
        best = collaborative_ratings
        #print("best")
        #print(best)
        preSorted = [[v, k] for k,v in best.items()]
        best = np.empty((max(11,len(preSorted)), 2), dtype=object)
        for i in range(len(preSorted)):
            best[i] = preSorted[i]
        if len(preSorted) < 11:
            for i in range(len(preSorted), 11):
                best[i] = [0, random.randint(0, 20000)]
            print(best)
        best = best[best[:,0].argpartition(10)][0:10]
        if playlist == 7:
            print(best)
        best = best[best[:,0].argsort()][:,1]
        return best

