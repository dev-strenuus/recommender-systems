import builder
import ItemCBFKNNRecommender as calc_sim
from evaluation_function import evaluate_algorithm
from hybridRecommender import HybridRecommender
b = builder.Builder()
URM_train, URM_test = b.train_test_holdout(0.8)

def compute_similarity():
    recommender = calc_sim.ItemCBFKNNRecommender(URM_train, b.get_ICM(0.8))
    contentSimilarity = recommender.fit(shrink=0.0, topK=500)
    recommender = calc_sim.ItemCBFKNNRecommender(URM_train, b.get_URM_transpose())
    collaborativeSimilarity = recommender.fit(shrink=5.0, topK=500)
    return contentSimilarity, collaborativeSimilarity

def run_local():
    contentSimilarity, collaborativeSimilarity = compute_similarity()
    return evaluate_algorithm(URM_test, HybridRecommender(contentSimilarity, collaborativeSimilarity, 0, 0),b)

def run_online():
    contentSimilarity, collaborativeSimilarity = compute_similarity()
    print_to_csv(contentSimilarity, collaborativeSimilarity)

def print_to_csv(contentSimilarity, collaborativeSimilarity):
    file=open("hybrid-submission.csv",'a')
    file.write("playlist_id,track_ids"+"\n")
    recommender = HybridRecommender(contentSimilarity, collaborativeSimilarity, 0.2, 1)
    for playlist in b.get_ordered_target_playlists():
        s = str(recommender.recommend(playlist,b))
        s = s[1:len(s)-1]
        file.write(str(playlist)+","+s+"\n")
    for playlist in b.get_unordered_target_playlists():
        s = str(recommender.recommend(playlist,b))
        s = s[1:len(s)-1]
        file.write(str(playlist)+","+s+"\n")