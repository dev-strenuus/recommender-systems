import builder
import ItemCBFKNNRecommender as calc_sim
from evaluation_function import evaluate_algorithm
from hybridRecommender import hybridRecommender

class Main(object):
    
    def __init__(self):
        self.b = builder.Builder()
        self.URM_train, self.URM_test = self.b.train_test_holdout(0.8)
    
    def compute_similarity(self):
        recommender = calc_sim.ItemCBFKNNRecommender(self.URM_train, self.b.get_ICM(0.65))
        self.contentSimilarity = recommender.fit(shrink=0.0, topK=500)
        
        recommender = calc_sim.ItemCBFKNNRecommender(self.URM_train, self.b.get_URM_transpose())
        self.collaborativeSimilarity = recommender.fit(shrink=5.0, topK=500)

    def run_local(self):
        self.compute_similarity()
        evaluate_algorithm(self.URM_test, hybridRecommender(self.contentSimilarity, self.collaborativeSimilarity, 0.2),self.b)

    def run_online(self):
        self.compute_similarity()
        self.print_to_csv()
    
    def print_to_csv(self):
        file=open("hybrid-submission.csv",'a')
        file.write("playlist_id,track_ids"+"\n")
        recommender = hybridRecommender(self.contentSimilarity, self.collaborativeSimilarity, 0.2)
        for playlist in self.b.get_ordered_target_playlists():
            s = str(recommender.recommend(playlist,self.b))
            s = s[1:len(s)-1]
            file.write(str(playlist)+","+s+"\n")
        for playlist in self.b.get_unordered_target_playlists():
            s = str(recommender.recommend(playlist,self.b))
            s = s[1:len(s)-1]
            file.write(str(playlist)+","+s+"\n")
