import builder
import ItemCBFKNNRecommender as calc_sim
from evaluation_function import evaluate_algorithm 
from hybridRecommender import hybridRecommender


def init(self):
    self.b = builder.Builder()
    self.URM_train, self.URM_test = self.b.train_test_holdout(0.8)


def run(self):
    recommender = calc_sim.ItemCBFKNNRecommender(self.URM_train, self.b.get_ICM(0.65))
    contentSimilarity = recommender.fit(shrink=0.0, topK=500)

    recommender = calc_sim.ItemCBFKNNRecommender(self.URM_train, self.b.get_URM_transpose())
    collaborativeSimilarity = recommender.fit(shrink=5.0, topK=500)

    evaluate_algorithm(self.URM_test, hybridRecommender(contentSimilarity, collaborativeSimilarity),10,self.b)

def print_to_csv():
    pass