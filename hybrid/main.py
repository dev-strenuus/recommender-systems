from data_splitter import train_test_holdout
URM_train, URM_test = train_test_holdout(Builder().get_URM(), train_perc = 0.8)
recommender = ItemCBFKNNRecommender(URM_train, Builder().get_ICM())
contentSimilarity = recommender.fit(shrink=0.0, topK=500)

recommender = ItemCBFKNNRecommender(URM_train, Builder().get_URM_transpose())
collaborativeSimilarity = recommender.fit(shrink=5.0, topK=500)

file=open("collaborative-submission.csv",'a')
file.write("playlist_id,track_ids"+"\n")
for user_id in Builder().get_target_playlists():
    s = str(recommender.recommend(user_id, at=10))
    s = s[1:len(s)-1]
    file.write(str(user_id)+","+  s   +"\n")

def merge(a, b):
    hybridMatrix = a*contentSimilarity + b*collaborativeSimilarity
    return hybridMatrix
bestSimilarTracks = merge(0.18, 1.0)
df = pd.read_csv('train.csv')
df1 = pd.read_csv('target_playlists.csv')
playlists = np.array(df1['playlist_id'])
import random
submission = np.empty((len(playlists),2), dtype=object)
cont = -1
for playlist in playlists:
    cont = cont + 1
    tracks = np.array(df[df['playlist_id']==playlist]['track_id'])
    submission[cont][0] = playlist
    tracksSet = set(tracks)
    best = {}
    temp = 1
    for track in tracks:
        if cont < 5000:
            temp -= 1/(len(tracks)+1)
        row_start = bestSimilarTracks.indptr[track]
        row_end = bestSimilarTracks.indptr[track+1]
        similarTracks = bestSimilarTracks.indices[row_start:row_end]
        similarityValues = bestSimilarTracks.data[row_start:row_end]
        for i in range(0, len(similarTracks)):
            if not similarTracks[i] in tracksSet:
                if similarTracks[i] in best:
                    best[similarTracks[i]]=best[similarTracks[i]]-similarityValues[i]*temp
                else:
                    best[similarTracks[i]]=-1*similarityValues[i]*temp
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
    best = str(best)
    submission[cont][1] = best[1:len(best)-1]
df2 = pd.DataFrame(submission, columns=['playlist_id','track_ids'])
df2.to_csv('hybrid_submission.csv', index=False)

evaluate_algorithm(URM_test, recommender,at=10)