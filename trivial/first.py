df = pd.read_csv('input/train.csv')
size = df.loc[df['track_id'].idxmax()]['track_id']
print(size)
playlistsForTrack = np.empty([size], dtype=set)
for i in range(size):
    playlistsForTrack[i] = set(df[df['track_id']==i]['playlist_id'])
bestSimilarTracks = np.empty((size), dtype=object)
print(size)
print(bestSimilarTracks)
for i in range(size):
    print(i)
    bestSimilarTracks[i] = np.empty((size, 2), dtype=object)
    bestSimilarTracks[i][i] = [0,i]
    for j in range(i):
        similarity = len(playlistsForTrack[i] & playlistsForTrack[j])
        bestSimilarTracks[i][j] = [-1*similarity, j]
    for j in range(i+1, size):
        similarity = len(playlistsForTrack[i] & playlistsForTrack[j])
        bestSimilarTracks[i][j] = [-1*similarity, j]
    bestSimilarTracks[i] = np.partition(bestSimilarTracks[i], 10, axis=0)[0:10]
    bestSimilarTracks[i] = bestSimilarTracks[i][bestSimilarTracks[i][:,0] !=0][:,1]

