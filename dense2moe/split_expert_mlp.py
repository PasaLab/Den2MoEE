#!/usr/bin/env python3
import os
import json
import numpy as np

def normalize_rows(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def cosine_distance_matrix(X, C):
    sim = X.dot(C.T)
    return np.maximum(1.0 - sim, 0.0)

def kmeans_pp_init(X, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = X.shape[0]
    idx = rng.integers(0, n)
    centers = [X[idx].copy()]
    if k == 1:
        return np.vstack(centers)
    dist2 = cosine_distance_matrix(X, np.vstack(centers)).squeeze()
    for _ in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers.append(X[idx].copy())
        dist_new = cosine_distance_matrix(X, np.vstack(centers)).min(axis=1)
        dist2 = dist_new
    return np.vstack(centers)


def balanced_assignment_greedy(dist_mat, cap):
    n, k = dist_mat.shape
    if n != k * cap:
        raise ValueError(f"balanced_assignment_greedy requires n == k*cap, got n={n}, k={k}, cap={cap}")
    # Create tuples
    idx_i, idx_j = np.indices((n, k))
    flat = np.stack([dist_mat.ravel(), idx_i.ravel(), idx_j.ravel()], axis=1)
    order = np.argsort(flat[:, 0], kind='stable')
    assigned = -np.ones(n, dtype=int)
    rem = np.full(k, cap, dtype=int)
    for pos in order:
        d, i, j = flat[pos]
        i = int(i); j = int(j)
        if assigned[i] != -1:
            continue
        if rem[j] <= 0:
            continue
        assigned[i] = j
        rem[j] -= 1
        # early exit when all assigned
        if (assigned != -1).all():
            break
    if (assigned == -1).any():
        raise RuntimeError("Assignment failed to assign all points")
    return assigned

def constrained_equal_size_kmeans(X, k, max_iter=50, tol=1e-6, rng=None):
    n, d = X.shape
    cap = n // k
    if rng is None:
        rng = np.random.default_rng()

    Xn = normalize_rows(X)
    centers = kmeans_pp_init(Xn, k, rng=rng)  # (k, d)
    centers = normalize_rows(centers)

    labels = np.full(n, -1, dtype=int)
    prev_centers = centers.copy()
    for it in range(max_iter):
        dist = cosine_distance_matrix(Xn, centers)  # (n, k)
        labels = balanced_assignment_greedy(dist, cap)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            idxs = np.where(labels == j)[0]
            if len(idxs) == 0:
                # fallback: re-init with random point
                new_centers[j] = Xn[rng.integers(0, n)]
            else:
                new_centers[j] = Xn[idxs].mean(axis=0)
        new_centers = normalize_rows(new_centers)
        shift = np.linalg.norm(prev_centers - new_centers)
        if shift <= tol:
            centers = new_centers
            break
        centers = new_centers
        prev_centers = centers.copy()
    return labels, centers


def calculate_coverage_matrix(cluster_mean_matrix, top_p=50):
    cluster_mean_matrix = np.asarray(cluster_mean_matrix)
    thresholds = np.percentile(cluster_mean_matrix, top_p, axis=0)  # (D,)
    coverage_matrix = (cluster_mean_matrix > thresholds).astype(int)
    return coverage_matrix



def process_layer_importance(
    importance_json_path,
    k,
    output_path,
    coverage_thr,
    seed=12345,
    max_iter=50,
    coverage_top_p=75
):
    """
    importance_json_path: path to neuron_importance_layer{L}.json
    k: target splits per layer MLP
    output_path: where to save result json for this layer
    """
    rng = np.random.default_rng(seed)

    with open(importance_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # group only by layer_id (ignore expert_id, since MLP is shared)
    layer_id = None
    neurons = []
    for elem in data:
        if layer_id is None:
            layer_id = elem.get("layer_id", None)
        nid = int(elem["neuron_id"])
        vec = np.array(elem["domain_vector"], dtype=float)
        neurons.append((nid, vec))

    neurons = sorted(neurons, key=lambda x: x[0])  # sort by neuron_id
    neuron_ids = [n for n, v in neurons]
    vectors = np.stack([v for n, v in neurons], axis=0)  # (num_neurons, dim)
    num_neurons = vectors.shape[0]

    if num_neurons == 0:
        raise ValueError(f"Layer {layer_id} has 0 neurons")
    if num_neurons % k != 0:
        raise ValueError(
            f"Layer {layer_id} has {num_neurons} neurons, which is not divisible by k={k}."
        )

    labels, centers = constrained_equal_size_kmeans(vectors, k, max_iter=max_iter, rng=rng)

    clusters = {}
    for j in range(k):
        idxs = np.where(labels == j)[0]
        cluster_neuron_ids = [int(neuron_ids[i]) for i in idxs]
        assert len(cluster_neuron_ids) == num_neurons // k, (
            f"Cluster size mismatch for layer {layer_id}: "
            f"got {len(cluster_neuron_ids)}, expected {num_neurons // k}"
        )
        cluster_neuron_ids.sort()
        cluster_mean_scores = vectors[idxs].mean(axis=0)

        clusters[j] = {
            "layer_id": layer_id,
            "micro_expert_id": j,
            "cluster_neuron_ids": cluster_neuron_ids,
            "cluster_mean_scores": cluster_mean_scores.tolist()
        }

    # coverage matrix
    coverage_matrix = calculate_coverage_matrix(
        [clusters[j]["cluster_mean_scores"] for j in range(k)],
        top_p=coverage_top_p
    )  # (k, D)
    for j in range(k):
        coverage_vec = coverage_matrix[j].tolist()
        coverage_score = sum(coverage_vec)
        clusters[j]["coverage_vector"] = coverage_vec
        clusters[j]["coverage_score"] = coverage_score

    expert_list = []
    for j in range(k):
        cov_score = clusters[j]["coverage_score"]
        norm_val = float(np.linalg.norm(clusters[j]["cluster_mean_scores"]))
        expert_list.append((cov_score, norm_val, j))

    expert_list.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_n = max(1, int(k * coverage_thr))
    top_experts = set(j for (_, _, j) in expert_list[:top_n])

    for j in range(k):
        if j in top_experts:
            clusters[j]["expert_type"] = "shared"
        else:
            clusters[j]["expert_type"] = "routed"

    # save
    layer_result = clusters

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(layer_result, f, indent=2, ensure_ascii=False)

    print(f"Saved splits for layer {layer_id} to {output_path}")
    return output_path



DATA_DIR = ""
TIME_STAMP = ""

SEED = 12345
MAX_ITER = 100
K = 8
COVERAGE_TOP_P=50
COVERAGE_THR=0.125
IMPORTANCE_DIR = f"{DATA_DIR}/importances"
OUTPUT_DIR = f"{DATA_DIR}/expert_splits"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find files
    files = []
    for fn in os.listdir(IMPORTANCE_DIR):
        if fn.startswith("neuron_importance_layer") and fn.endswith(f"_{TIME_STAMP}.json"):
            files.append(os.path.join(IMPORTANCE_DIR, fn))
    files = sorted(files, key=lambda x: int(os.path.basename(x).split("layer")[1].split("_")[0]))

    if len(files) == 0:
        raise FileNotFoundError(f"No importance files found in {IMPORTANCE_DIR} with timestamp {TIME_STAMP}")

    for fpath in files:
        layer_str = os.path.basename(fpath)
        layer_id = int(layer_str.split("layer")[1].split("_")[0])
        out_fname = os.path.join(OUTPUT_DIR, f"neuron_splits_layer{layer_id}_k{K}_{TIME_STAMP}.json")
        process_layer_importance(
            fpath,
            k=K,
            output_path=out_fname,
            coverage_thr=COVERAGE_THR,
            seed=SEED,
            max_iter=MAX_ITER,
            coverage_top_p=COVERAGE_TOP_P
        )   

if __name__ == "__main__":
    main()
