import pandas as pd
import numpy as np
df = pd.read_csv('input/train.csv')
size = df.loc[df['track_id'].idxmax()]['track_id']
playlistsForTrack = np.empty([size], dtype=set)
for i in range(size):
    playlistsForTrack[i] = set(df[df['track_id']==i]['playlist_id'])
bestSimilarTracks = np.empty((size), dtype=object)
for i in range(size):
    bestSimilarTracks[i] = np.empty((size, 2), dtype=object)
    bestSimilarTracks[i][i] = [0,i]
    for j in range(i):
        den = ((len(playlistsForTrack[i]))*(len(playlistsForTrack[j])))**0.5+10
        similarity = len(playlistsForTrack[i] & playlistsForTrack[j])/den
        bestSimilarTracks[i][j] = [-1*similarity, j]
    for j in range(i+1, size):
        den = ((len(playlistsForTrack[i]))*(len(playlistsForTrack[j])))**0.5+10
        similarity = len(playlistsForTrack[i] & playlistsForTrack[j])/den
        bestSimilarTracks[i][j] = [-1*similarity, j]
    bestSimilarTracks[i] = bestSimilarTracks[i][bestSimilarTracks[i][:,0].argpartition(10)][0:10]
    bestSimilarTracks[i] = bestSimilarTracks[i][bestSimilarTracks[i][:,0] !=0]#[:,1]
    bestSimilarTracks[i][:,0] = -1*bestSimilarTracks[i][:,0]
df1 = pd.read_csv('input/target_playlists.csv')
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
        for similar in bestSimilarTracks[track]:
            if not similar[1] in tracksSet:
                if similar[1] in best:
                    best[similar[1]]=best[similar[1]]-similar[0]*temp
                else:
                    best[similar[1]]=-1*similar[0]*temp
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
df2.to_csv('solution.csv', index=False)
