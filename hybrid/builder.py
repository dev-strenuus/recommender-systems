import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn import feature_extraction


class Builder(object):
    
    def __init__(self):
        self.train= pd.read_csv('train.csv')
        self.target_playlists = pd.read_csv('target_playlists.csv')
        self.tracks = pd.read_csv('tracks.csv')
        self.playlists = self.get_playlists()
    
    def get_tracks(self):
        tracks = self.tracks['track_id'].unique()
        return np.sort(tracks)
    
    def get_playlists(self):
        playlists = self.train['playlist_id'].unique()
        return np.sort(playlists)
    
    def get_target_playlists(self):
        target_playlists = self.target_playlists['playlist_id'].unique()
        return np.sort(target_playlists)
    
    def get_artists(self):
        artists = self.tracks['artist_id'].unique()
        return np.sort(artists)
    
    def get_albums(self):
        albums = self.tracks['album_id'].unique()
        return np.sort(albums)
    
    def get_durations(self):
        durations = self.tracks['duration_sec'].unique()
        return np.sort(durations)
    
    def get_target_playlist_index(self, target_playlist):
        return np.where(self.playlists == target_playlist)[0][0] #DA RIVEDERE
    
    def get_URM(self):
        grouped = self.train.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
        self.URM = MultiLabelBinarizer(classes=self.get_tracks(), sparse_output=True).fit_transform(grouped)
        return self.URM
    
    def get_URM_transpose(self):
        grouped = self.train.groupby('track_id', as_index=True).apply(lambda x: list(x['playlist_id']))
        self.URM = MultiLabelBinarizer(classes=self.get_playlists(), sparse_output=True).fit_transform(grouped)
        return self.URM
    
    def get_ICM(self):
        artists = self.tracks.reindex(columns=['track_id', 'artist_id'])
        artists.sort_values(by='track_id', inplace=True)
        artists_list = [[a] for a in artists['artist_id']]
        icm_artists = MultiLabelBinarizer(classes=self.get_artists(), sparse_output=True).fit_transform(artists_list)
        icm_artists_csr = icm_artists.tocsr()
        #return icm_artists_csr
        
        albums = self.tracks.reindex(columns=['track_id', 'album_id'])
        albums.sort_values(by='track_id', inplace=True)
        albums_list = [[a] for a in albums['album_id']]
        icm_albums = MultiLabelBinarizer(classes=self.get_albums(), sparse_output=True).fit_transform(albums_list)
        icm_albums_csr = icm_albums.tocsr()
        #return icm_albums_csr
        
        #durations= self.tracks.reindex(columns=['track_id', 'duration_sec'])
        #durations.sort_values(by='track_id', inplace=True)
        #durations_list = [[d] for d in durations['duration_sec']]
        #icm_durations = MultiLabelBinarizer(classes=self.get_durations(), sparse_output=True).fit_transform(durations_list)
        #icm_durations_csr= icm_durations.tocsr()
        
        return sparse.hstack((icm_artists_csr,icm_albums_csr))

