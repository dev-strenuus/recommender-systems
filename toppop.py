import numpy as np
import pandas as pd

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('target_playlists.csv')
playlists_id=df2["playlist_id"]
df1 = np.array(df1)
unique, counts = np.unique(df1[:,[1]], return_counts=True)
trackid_nstreams=np.asarray((unique, counts)).T
trackid_nstreams_sorted=trackid_nstreams[trackid_nstreams[:, 1].argsort()]
rows,cols = np.shape(trackid_nstreams_sorted)
top10 = trackid_nstreams_sorted[rows-10:rows,:]
top10_id = top10[:,:1]
top10_ordered = top10_id[::-1]
top = np.reshape(top10_ordered, (1,10))
a = ' '.join(map(str, top[0]))
file=open("submission2.csv",'a')
file.write("playlist_id,track_ids"+"\n")
i=0
for u in playlists_id:
    file.write(str(playlists_id[i])+","+str(a)+"\n")
    i=i+1