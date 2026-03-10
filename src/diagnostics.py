"""
REFERENCE ONLY FILE
Our intention in sharing this file is to share more low-level details such as the 30 count threshold for doing the exact
Mann-Whitney; error handling for MCC: if undefined, then: all incorrect -> -1, all correct -> 1, otherwise 0; and
spectral clustering using the "cluster_qr" label assignment method.
For HMP, we use max(p-value, 1e-12) for each p-value to ensure numerical stability. We reason that it's not realistic
to be more confident than "1 in a trillion" with our datasets.
This file contains some future work such as how well SVC performs when trained on all data, including the test.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn import metrics as skm
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, matthews_corrcoef
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar

import common as cmn
import metrics_library
from common import STAT_DTYPE
from config import TriLensConfig, LensCol
from dataset_library import TabularDataset

_WHITNEY_EXACT_TEST_CUTOFF = 30 # minority test


def mcnemar_exact_test(correct_in_method:np.ndarray, correct_baseline:np.ndarray)-> float:
    contingency_table = np.array([
        [np.sum(correct_in_method & correct_baseline), np.sum(correct_in_method & ~correct_baseline)],
        [np.sum(~correct_in_method & correct_baseline), np.sum(~correct_in_method & ~correct_baseline)]
    ], dtype=np.int64)
    res = mcnemar(contingency_table, exact=True)
    return res.pvalue

def auc_whitney(scores, mask0, mask1, use_exact_test:bool):
    # Perform Mann-Whitney U test for P-value
    u_stat, p_val = mannwhitneyu(
        scores[mask1], scores[mask0],
        alternative="two-sided",
        method='exact' if use_exact_test else 'asymptotic',
    )

    n1 = mask1.sum()
    n2 = mask0.sum()
    auc_score = u_stat / (n1 * n2)
    auc_score = max(auc_score, 1 - auc_score)
    return auc_score, p_val


def calc_mcc(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    if not np.isfinite(mcc):
        ncorrect = (labels == preds).sum()
        # If two holes and not all (in)correct
        if 0 < ncorrect < preds.shape[0]:
            mcc = 0.
        elif ncorrect == 0:
            mcc = -1.
        else:  # All correct
            mcc = 1.
    return mcc

def get_mcc_acc_mcnemar(predictions, labels, baseline):
    mcc = calc_mcc(labels, predictions)
    hits = predictions == labels
    acc = hits.mean()
    mcnemar_p_val = mcnemar_exact_test(
        correct_in_method= hits,
        correct_baseline=baseline,
    )
    return mcc, acc, mcnemar_p_val


def rbf_kernel(
    distances:np.ndarray,
):
    median = max(np.median(distances), 1e-10)
    gamma = 1/ (2* median**2)
    return np.exp(-gamma * distances **2 )

def ari_score_pvalue(
        labels: np.ndarray,
        cluster_labels: np.ndarray,
) -> tuple[float, float]:
    ari_obs = adjusted_rand_score(labels, cluster_labels)
    table = skm.confusion_matrix(labels, cluster_labels, labels=[0,1])
    assert table.shape == (2, 2), "Contingency table must be 2x2 for p-value calculation"
    _, p_value = fisher_exact(table, alternative='two-sided')
    return ari_obs, p_value

def _oddize(x: int) -> int:
    # Keep as-is if odd; else go to the nearest odd below (so we don't overshoot n-1)
    return x if (x % 2 == 1) else max(1, x - 1)
class TriLens:
    def __init__(self, config: TriLensConfig, dataset_name: str, dataset_size: int):
        n = int(dataset_size)

        # Build the k-grid once (labels -> integer k).
        # For majority vote we'll tie-break using dataset majority, so even k is OK.
        k_map: dict[str, int] = {
            "1": 1,
            "3": 3,
            "5": 5,
            "7": 7,
        }

        k_map["√n"] = int(np.floor(np.sqrt(n)))

        # Percent-of-(n-1), rounded up
        for pct in (2, 5):
            k_pct = int(np.ceil(pct / 100.0 * (n - 1)))
            k_map[f"{pct}%"] = max(1, k_pct)

        k_map["n-1"] = n-1
        k_map["n"]   = n

        self._k_nn_test_values = k_map
        self.config = config
        self.dataset_name = dataset_name

        self._results = {
            name: pd.DataFrame(columns=LensCol.all_cols, index=list(config.measures.keys()))
            for name in cmn.Lens.all_names
        }

    @property
    def results(self) -> dict[str, pd.DataFrame]:
        return self._results

    def _rank_results(
            self,
            results_df: pd.DataFrame,
            ascending: bool,
    ) -> pd.DataFrame:
        ranking_col = LensCol.ranking
        score_col = LensCol.score
        significant_mask = (results_df[score_col] != np.nan)
        results_df[ranking_col] = np.nan
        ranks = results_df.loc[significant_mask, score_col].rank(ascending=ascending, method='average')
        results_df.loc[significant_mask, ranking_col] = ranks
        significant_count = significant_mask.sum()
        measures_count = results_df.shape[0]
        if significant_count < measures_count:
            non_sig_rank = (measures_count + significant_count + 1) / 2
            results_df.loc[~significant_mask, ranking_col] = non_sig_rank
        return results_df

    def knn_scores(
            self,
            distances: np.ndarray,
            measure_name: str,
            labels: np.ndarray,
            use_exact_test: bool,
    ) -> pd.DataFrame:

        y = labels.astype(np.int8)
        n = y.size
        majority_label = int(y.mean() >= 0.5)
        correct_baseline = (y == majority_label)

        score_col = LensCol.score
        p_value_col = LensCol.p_value
        name = measure_name

        standard_k_values = [k for k in self._k_nn_test_values.keys() if k not in ("n", "n-1")]

        # Precompute neighbor lists up to the largest k that actually uses neighbors
        max_k_neighbors = max(val for key, val in self._k_nn_test_values.items() if key in standard_k_values)

        # Indices of the smallest distances (include self; we’ll remove it below)
        idxs_knn = np.argpartition(distances, max_k_neighbors, axis=-1)[:, :max_k_neighbors + 1]

        # Storage for per-k predictions/scores
        maj_preds = defaultdict(list)  # majority-vote predictions per k
        w_preds = defaultdict(list)  # weighted avg-distance predictions per k
        w_scores = defaultdict(list)  # avg_dif (continuous) per k for AUC

        # Build neighbor order and compute per-k items
        for i, nn_idx in enumerate(idxs_knn):
            # drop self if present
            og_shape = nn_idx.shape[0]
            nn_idx = nn_idx[nn_idx != i]
            # sort the retained neighbors by distance
            order = np.argsort(distances[i, nn_idx])
            nn_idx = nn_idx[order]
            if og_shape == nn_idx.shape[0]:
                nn_idx = nn_idx[:-1]

            nn_dists = distances[i, nn_idx]
            nn_y = y[nn_idx]

            for k_label, k in self._k_nn_test_values.items():
                if k_label not in standard_k_values:
                    # handled after the loop (true all-points path)
                    continue

                k_eff = min(k, nn_idx.size)
                if k_eff <= 0:
                    # degenerate, fallback to dataset majority
                    maj_preds[k_label].append(majority_label)
                    w_preds[k_label].append(majority_label)
                    w_scores[k_label].append(np.inf)  # won't be used
                    continue

                loc_y = nn_y[:k_eff]
                loc_d = nn_dists[:k_eff]

                # Majority vote with dataset-level tie-break
                s = int(loc_y.sum())
                if s * 2 > k_eff:
                    maj = 1
                elif s * 2 < k_eff:
                    maj = 0
                else:
                    maj = majority_label
                maj_preds[k_label].append(maj)

                # Weighted (average-distance within class among the k neighbors)
                m0 = (loc_y == 0)
                m1 = ~m0
                c0 = int(m0.sum())
                c1 = int(m1.sum())

                # means; if a class is absent among the k neighbors, set its mean to +inf
                avg0 = (loc_d[m0].sum() / c0) if c0 > 0 else np.inf
                avg1 = (loc_d[m1].sum() / c1) if c1 > 0 else np.inf

                # tie-break toward local majority (maj)
                w_pred = 1 if (avg1 < avg0) else (0 if (avg1 > avg0) else maj)
                w_preds[k_label].append(w_pred)

                # For AUC
                w_scores[k_label].append(avg1 - avg0)

        # Convert lists to arrays and write majority + weighted metrics for all neighbor-based ks
        for k_label in standard_k_values:
            # majority k-NN
            assert len(maj_preds[k_label]) == n
            preds = np.asarray(maj_preds[k_label], dtype=np.int8)
            mcc, acc, p = get_mcc_acc_mcnemar(preds, y, correct_baseline)
            self._results[cmn.get_knn_mcc_name(k_label)].loc[name, score_col] = mcc
            self._results[cmn.get_knn_mcc_name(k_label)].loc[name, p_value_col] = p
            self._results[cmn.get_knn_acc_name(k_label)].loc[name, score_col] = acc
            self._results[cmn.get_knn_acc_name(k_label)].loc[name, p_value_col] = p

            # weighted k-NN (MCC/Acc + AUC)
            assert len(w_preds[k_label]) == n
            wpred = np.asarray(w_preds[k_label], dtype=np.int8)
            mcc, acc, p = get_mcc_acc_mcnemar(wpred, y, correct_baseline)
            wbase = cmn.get_weighted_knn_name(k_label)
            self._results[cmn.get_knn_mcc_name_from_base(wbase)].loc[name, score_col] = mcc
            self._results[cmn.get_knn_mcc_name_from_base(wbase)].loc[name, p_value_col] = p
            self._results[cmn.get_knn_acc_name_from_base(wbase)].loc[name, score_col] = acc
            self._results[cmn.get_knn_acc_name_from_base(wbase)].loc[name, p_value_col] = p

            # AUC
            diffs = np.asarray(w_scores[k_label], dtype=np.float64)
            auc, p_auc = auc_whitney(
                scores=diffs,
                mask0=(y == 0),
                mask1=(y == 1),
                use_exact_test=use_exact_test,
            )
            self._results[cmn.get_knn_auc_name_from_base(wbase)].loc[name, score_col] = auc
            self._results[cmn.get_knn_auc_name_from_base(wbase)].loc[name, p_value_col] = p_auc

        c0 = (y == 0)
        c1 = ~c0
        m0 = int(c0.sum())
        m1 = int(c1.sum())

        # global class means per row
        sum0 = distances[:, c0].sum(axis=1)
        sum1 = distances[:, c1].sum(axis=1)
        avg0 = sum0 / max(1, m0)
        avg1 = sum1 / max(1, m1)

        # predictions with dataset-majority tie-break
        preds_w_all = np.where(
            avg1 < avg0, 1,
            np.where(avg1 > avg0, 0, majority_label)
        ).astype(np.int8)

        mcc, acc, p = get_mcc_acc_mcnemar(preds_w_all, y, correct_baseline)
        wbase_all = cmn.get_weighted_knn_name("n")
        self._results[cmn.get_knn_mcc_name_from_base(wbase_all)].loc[name, score_col] = mcc
        self._results[cmn.get_knn_mcc_name_from_base(wbase_all)].loc[name, p_value_col] = p
        self._results[cmn.get_knn_acc_name_from_base(wbase_all)].loc[name, score_col] = acc
        self._results[cmn.get_knn_acc_name_from_base(wbase_all)].loc[name, p_value_col] = p

        diffs_all = (avg1 - avg0).astype(np.float64)
        auc_all, p_auc_all = auc_whitney(
            scores=diffs_all,
            mask0=(y == 0),
            mask1=(y == 1),
            use_exact_test=use_exact_test,
        )
        self._results[cmn.get_knn_auc_name_from_base(wbase_all)].loc[name, score_col] = auc_all
        self._results[cmn.get_knn_auc_name_from_base(wbase_all)].loc[name, p_value_col] = p_auc_all

        # ---------- derive k = "n-1" from the "n" means above ----------
        # Identity: for i in class c with size m_c,
        #   avg_c^{(n-1)}(i) = (m_c * avg_c^{(n)}(i) - d(i,i)) / (m_c - 1)
        # The opposite-class mean is unchanged.
        diag = np.diag(distances)

        avg0_n1 = avg0.copy()
        avg1_n1 = avg1.copy()

        if m0 > 1:
            avg0_n1[c0] = (m0 * avg0[c0] - diag[c0]) / (m0 - 1)
        else:
            avg0_n1[c0] = np.inf  # only self in class 0; should never happen

        if m1 > 1:
            avg1_n1[c1] = (m1 * avg1[c1] - diag[c1]) / (m1 - 1)
        else:
            avg1_n1[c1] = np.inf  # only self in class 1; also should never happen

        preds_w_n1 = np.where(
            avg1_n1 < avg0_n1, 1,
            np.where(avg1_n1 > avg0_n1, 0, majority_label)
        ).astype(np.int8)

        mcc_n1, acc_n1, p_n1 = get_mcc_acc_mcnemar(preds_w_n1, y, correct_baseline)
        wbase_n1 = cmn.get_weighted_knn_name("n-1")
        self._results[cmn.get_knn_mcc_name_from_base(wbase_n1)].loc[name, score_col] = mcc_n1
        self._results[cmn.get_knn_mcc_name_from_base(wbase_n1)].loc[name, p_value_col] = p_n1
        self._results[cmn.get_knn_acc_name_from_base(wbase_n1)].loc[name, score_col] = acc_n1
        self._results[cmn.get_knn_acc_name_from_base(wbase_n1)].loc[name, p_value_col] = p_n1

        diffs_n1 = (avg1_n1 - avg0_n1).astype(np.float64)
        auc_n1, p_auc_n1 = auc_whitney(
            scores=diffs_n1,
            mask0=(y == 0),
            mask1=(y == 1),
            use_exact_test=use_exact_test,
        )
        self._results[cmn.get_knn_auc_name_from_base(wbase_n1)].loc[name, score_col] = auc_n1
        self._results[cmn.get_knn_auc_name_from_base(wbase_n1)].loc[name, p_value_col] = p_auc_n1

        return

    def cluster_and_density_scores(
            self,
            distances : np.ndarray,
            measure_name : str,
            labels: np.ndarray,
            use_exact_test: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        y = labels
        n = len(y)
        score_col = LensCol.score
        p_value_col = LensCol.p_value
        q = self.config.q
        eps = np.finfo(np.float32).eps
        densities = np.empty(n, dtype=STAT_DTYPE)
        D = distances
        name = measure_name

        clusterer = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")
        preds = clusterer.fit_predict(D)
        ari, p_val = ari_score_pvalue(y, preds)
        self._results[cmn.Lens.agg_cluster].loc[name, score_col] = ari
        self._results[cmn.Lens.agg_cluster].loc[name, p_value_col] = p_val

        clusterer = KMedoids(n_clusters=2, metric="precomputed", method="fasterpam")
        preds = clusterer.fit_predict(D)
        ari, p_val = ari_score_pvalue(y, preds)
        self._results[cmn.Lens.k_medoids].loc[name, score_col] = ari
        self._results[cmn.Lens.k_medoids].loc[name, p_value_col] = p_val

        # Density: RNLDD-AUC
        k = max(int(q * (n - 1)), 1) # nearest neighbors
        k_include_self = k + 1  # include self in the neighborhood
        for i in range(n):
            # partial sort to find k nearest neighbors
            neighbors = np.argpartition(D[i], k)[:k_include_self]
            # exclude self, avoid division by zero
            total_dist = D[i,neighbors].sum() - D[i,i] + eps
            densities[i] = k / total_dist
        # Rank-normalize: highest density -> rank 1
        ranks = pd.Series(densities).rank(ascending=False, method='average')
        s = (ranks - 1) / (n - 1)  # in [0,1]

        one_mask = (y == 1)
        zero_mask = (y == 0)

        # Perform Mann-Whitney U test for P-value
        u_stat, p_val = mannwhitneyu(
            s[one_mask], s[zero_mask],
            alternative="two-sided",
            method='exact' if use_exact_test else 'asymptotic',
        )
        self._results[cmn.Lens.rndldd_auc].loc[name, p_value_col] = p_val

        # Score = RNLDD-AUC
        n1 = one_mask.sum()
        n2 = zero_mask.sum()
        auc_score = u_stat / (n1 * n2)
        self._results[cmn.Lens.rndldd_auc].loc[name, score_col] = max(auc_score, 1 - auc_score)

        ### SVC

        majority_label = (labels.mean() >= 0.5).astype(np.int8)
        correct_baseline = y == majority_label
        S = rbf_kernel(D)
        svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
        svm.fit(X=S, y = y,)
        preds = svm.predict(S)
        mcc = calc_mcc(y, preds)
        mcnemar_p_val = mcnemar_exact_test(
            correct_in_method=preds == labels,
            correct_baseline=correct_baseline,
        )
        self._results[cmn.Lens.svm].loc[name, score_col] = mcc
        self._results[cmn.Lens.svm].loc[name, p_value_col] = mcnemar_p_val

        # SVC-LOO
        preds = []
        max_svc_samples = cmn.MAX_SVC_SAMPLES
        idxs = (
            np.random.choice(n, size=max_svc_samples, replace=False)
            if n > max_svc_samples
            else np.arange(n)
        )

        for i in idxs:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_train = S[mask][:, mask]
            y_train = y[mask]
            x_test = S[i, mask][None, :]

            svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
            svm.fit(X=x_train, y = y_train,)
            preds.append(svm.predict(x_test))

        preds = np.array(preds)
        y_subset = y[idxs]
        mcc = calc_mcc(y_subset, preds)
        mcnemar_p_val = mcnemar_exact_test(
            correct_in_method = preds == y_subset,
            correct_baseline = majority_label == y_subset,
        )
        self._results[cmn.Lens.svm_loo].loc[name, score_col] = mcc
        self._results[cmn.Lens.svm_loo].loc[name, p_value_col] = mcnemar_p_val

        # import warnings
        # warnings.filterwarnings("error")
        clusterer = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            assign_labels="cluster_qr",
            # n_neighbors=min(30, S.shape[0]-1),
            random_state=42
        )
        preds = clusterer.fit_predict(S)
        ari, p_val = ari_score_pvalue(y, preds)
        self._results[cmn.Lens.spectral].loc[name, score_col] = ari
        self._results[cmn.Lens.spectral].loc[name, p_value_col] = p_val
        return


    def run(self, tabData: TabularDataset) -> None:
        n = len(tabData)
        use_exact_test = tabData.minority_class_count < _WHITNEY_EXACT_TEST_CUTOFF
        fmax = np.finfo(np.float32).max

        print(f"Beginning TriLens for {tabData.name} with {n} samples.")
        labels = tabData.y
        a = tabData.x[:, np.newaxis, :]  # Shape: (n, 1, features)
        b = tabData.x[np.newaxis, :, :]  # Shape: (1, n, features)
        for (memo_key, memo_func), measure_keys in metrics_library.memo_map.items():
            kwargs = {"a": a, "b": b}
            if memo_func is not None:
                kwargs[memo_key] = memo_func(kwargs["a"], kwargs["b"])

            for measure_key in measure_keys:
                # Compute pairwise distances
                with np.errstate(all="ignore"):
                    D = self.config.measures[measure_key](
                        **kwargs,
                    ).astype(np.float64)

                # We ensure that all undefined values from measures are identity cases
                np.nan_to_num(D, nan=0, copy=False)
                np.clip(D, a_min=0, a_max=fmax, out=D)

                self.knn_scores(D, measure_key, labels, use_exact_test)
                self.cluster_and_density_scores(D, measure_key, labels, use_exact_test)
                del D

        for key in self._results.keys():
            self._results[key] = self._rank_results(self._results[key], ascending=False)

