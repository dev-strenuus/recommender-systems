import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn import feature_extraction


class Builder(object):
    
    def __init__(self, local):
        if local == True:
            self.train= pd.read_csv('input/train_holdout.csv')
            self.test= pd.read_csv('input/test_holdout.csv')
        else:
            self.train = pd.read_csv('input/new_train.csv')
        self.target_playlists = pd.read_csv('input/target_playlists.csv')
        self.tracks = pd.read_csv('input/tracks.csv')
        #self.ordered_train=pd.read_csv('input/train_sequential.csv')
        #for playlist in np.array(self.target_playlists['playlist_id'])[0:5000]:
        #    self.train = self.train[self.train.playlist_id!=playlist]
        #self.train = self.train.append(self.ordered_train)
        #print (self.train[self.train['playlist_id']==7]['track_id'])
        self.playlists = self.get_playlists()
        #self.tracks_inside_playlists_train = np.empty((len(self.playlists)), dtype=object)

    def get_train_pd(self):
        return self.train

    def get_target_playlists_pd(self):
        return self.target_playlists

    def get_tracks_pd(self):
        return self.tracks

    def get_ordered_target_playlists(self):
        return np.array(self.target_playlists['playlist_id'])[0:5000]

    def get_unordered_target_playlists(self):
        return np.array(self.target_playlists['playlist_id'])[5000:]

    def get_tracks_inside_playlist_train(self, playlist):
        return np.array(self.train[self.train['playlist_id']==playlist]['track_id'])
    
    
    """def train_test_holdout(self, train_perc):
        playlistsSize = len(self.get_playlists())
        tracksSize = len(self.get_tracks())
        target_playlists = self.get_target_playlists()
        cont = 0
        URM_test_row = np.empty(0)
        URM_test_col = np.empty(0)
        URM_test_values = np.empty(0)
        URM_train_row = np.empty(0)
        URM_train_col = np.empty(0)
        URM_train_values = np.empty(0)
        file_train =open("train_holdout.csv",'a')
        file_train.write("playlist_id,track_id"+"\n")
        file_test =open("test_holdout.csv",'a')
        file_test.write("playlist_id,track_id"+"\n")
        np.random.seed(10)
        for playlist in range(0,playlistsSize):
            tracks = np.array(self.train[self.train['playlist_id']==playlist]['track_id'])
            train = []
            test = []
            if cont < len(target_playlists) and playlist == target_playlists[cont]:
                if cont >= 5000:
                    np.random.shuffle(tracks)
                cont = cont + 1
                train = tracks[0:int(len(tracks)*train_perc)]
                test = tracks[int(len(tracks)*train_perc):]
            else:
                train = tracks
            self.tracks_inside_playlists_train[playlist] = train
            for track in train:
                file_train.write(str(playlist)+","+str(track)+"\n")
            for track in test:
                file_test.write(str(playlist)+","+str(track)+"\n")
        
            #else:
            #    URM_train_row = np.append(URM_train_row, [playlist]*len(tracks))
            #    URM_train_col = np.append(URM_train_col, tracks)
            #    URM_train_values = np.append(URM_train_values, [1]*len(tracks))
        self.URM_train = sps.csr_matrix( (URM_train_values,(URM_train_row, URM_train_col)), shape=(playlistsSize, tracksSize))
        self.URM_test = sps.csr_matrix( (URM_test_values,(URM_test_row, URM_test_col)), shape=(playlistsSize, tracksSize))"""
        #return self.URM_train, self.URM_test"""
    
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
    
    def get_URM_test(self):
        playlistsSize = len(self.get_playlists())
        tracksSize = len(self.get_tracks())
        URM_test_row = np.zeros(self.test.shape[0])
        URM_test_col = np.zeros(self.test.shape[0])
        URM_test_values = np.zeros(self.test.shape[0])
        cont = 0
        for playlist in range(playlistsSize):
            tracks = np.array(self.test[self.test['playlist_id']==playlist]['track_id'])
            length = len(tracks)
            if length > 0:
                URM_test_row[cont:cont+length] = [playlist]*length
                URM_test_col[cont:cont+length] = tracks
                URM_test_values[cont:cont+length] = [1]*length
                cont = cont + length
        self.URM_test = sps.csr_matrix( (URM_test_values,(URM_test_row, URM_test_col)), shape=(playlistsSize, tracksSize))
        return self.URM_test

    def get_URM_train(self):
        playlistsSize = len(self.get_playlists())
        tracksSize = len(self.get_tracks())
        URM_train_row = np.zeros(self.train.shape[0])
        URM_train_col = np.zeros(self.train.shape[0])
        URM_train_values = np.zeros(self.train.shape[0])
        cont = 0
        for playlist in range(playlistsSize):
            tracks = np.array(self.train[self.train['playlist_id']==playlist]['track_id'])
            length = len(tracks)
            if length > 0:
                URM_train_row[cont:cont+length] = [playlist]*length
                URM_train_col[cont:cont+length] = tracks
                URM_train_values[cont:cont+length] = [1]*length
                cont = cont + length
        self.URM_train = sps.csr_matrix( (URM_train_values,(URM_train_row, URM_train_col)), shape=(playlistsSize, tracksSize))
        return self.URM_train
    
    def get_URM_transpose_train(self):
        playlistsSize = len(self.get_playlists())
        tracksSize = len(self.get_tracks())
        URM_train_row = np.zeros(self.train.shape[0])
        URM_train_col = np.zeros(self.train.shape[0])
        URM_train_values = np.zeros(self.train.shape[0])
        cont = 0
        for track in range(tracksSize):
            playlists = np.array(self.train[self.train['track_id']==track]['playlist_id'])
            length = len(playlists)
            if length > 0:
                URM_train_row[cont:cont+length] = [track]*length
                URM_train_col[cont:cont+length] = playlists
                URM_train_values[cont:cont+length] = [1]*length
                cont = cont + length
        self.URM_train_transpose = sps.csr_matrix( (URM_train_values,(URM_train_row, URM_train_col)), shape=(tracksSize, playlistsSize))
        return self.URM_train_transpose
    
    def get_ICM(self, a):
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
        
        return sps.hstack((a*icm_artists_csr,icm_albums_csr))

