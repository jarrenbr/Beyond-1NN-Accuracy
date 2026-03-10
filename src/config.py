def lens_to_csv(lens_name: str) -> str:
    return lens_name.replace(' ', '_').replace('/', '_').replace(':', '_') + "_results.csv"

# LENS_FILE_NAMES = {
#     "Locality": "locality_results.csv",
#     "Clustering": "cluster_results.csv",
#     "Density": "density_results.csv",
#     "SVM": "svm_results.csv",
#     "AggregateClustering": "agg_cluster_results.csv",
#     "KNN3" : "knn3_results.csv",
#     "KNN5" : "knn5_results.csv",
#     "KNN7" : "knn7_results.csv",
#     "LocalityAccuracy": "locality_accuracy_results.csv"
# }

class LensCol:
    p_value = "p_value"
    score = "score"
    ranking = "ranking"
    all_cols = [p_value, score, ranking]

class ExpCol:
    domain = "domain"
    dataset = "dataset"
    measure = "measure"
    window_size = "window_size"
    lens = "lens"
    p_value = LensCol.p_value
    score = LensCol.score
    ranking = LensCol.ranking
    all_cols = [dataset, measure, window_size, p_value, score, ranking]

exp_file = "all_scores.csv"