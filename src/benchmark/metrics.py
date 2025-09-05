import numpy as np
from typing import List, Optional

def recall_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    Calculate recall@k for vector retrieval.

    Recall@k measures the intersection between true top-k vectors and
    returned top-k vectors, divided by the number of true top-k vectors.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Number of results to consider

    Returns:
        Average recall@k across all queries
    """
    if k > predicted.shape[1]:
        k = predicted.shape[1]

    # For each query, calculate how many of the ground truth items are in the top-k predictions
    n_queries = ground_truth.shape[0]
    recalls = np.zeros(n_queries)

    for i in range(n_queries):
        # Get the set of top-k ground truth indices for this query
        gt_set = set(ground_truth[i, :k]) if ground_truth.shape[1] >= k else set(ground_truth[i])
        # Get the set of predicted indices up to k
        pred_set = set(predicted[i, :k])
        # Calculate recall: |intersection| / |ground_truth|
        recalls[i] = len(gt_set.intersection(pred_set)) / len(gt_set) if len(gt_set) > 0 else 0.0

    return np.mean(recalls)

def precision_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    Calculate precision@k for vector retrieval.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Number of results to consider

    Returns:
        Average precision@k across all queries
    """
    if k > predicted.shape[1]:
        k = predicted.shape[1]

    # For each query, calculate how many of the top-k predictions are in the ground truth
    n_queries = ground_truth.shape[0]
    precisions = np.zeros(n_queries)

    for i in range(n_queries):
        # Get the set of ground truth indices for this query
        gt_set = set(ground_truth[i])
        # Get the set of predicted indices up to k
        pred_set = set(predicted[i, :k])
        # Calculate precision: |intersection| / |predicted|
        precisions[i] = len(gt_set.intersection(pred_set)) / k

    return np.mean(precisions)

def mean_average_precision(ground_truth: np.ndarray, predicted: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Mean Average Precision (MAP) for vector retrieval.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Optional limit on number of predictions to consider

    Returns:
        MAP score across all queries
    """
    n_queries = ground_truth.shape[0]
    if k is None:
        k = predicted.shape[1]
    else:
        k = min(k, predicted.shape[1])

    # Calculate AP for each query
    aps = np.zeros(n_queries)

    for i in range(n_queries):
        # Get the set of ground truth indices for this query
        gt_set = set(ground_truth[i])

        # Calculate precision at each position where a relevant item is found
        relevant_positions = []
        num_relevant = 0

        for j in range(k):
            if predicted[i, j] in gt_set:
                num_relevant += 1
                relevant_positions.append(num_relevant / (j + 1))

        # Calculate AP
        if len(relevant_positions) > 0:
            aps[i] = sum(relevant_positions) / len(gt_set)

    return np.mean(aps)


def ndcg_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k for vector retrieval.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Number of results to consider

    Returns:
        Average NDCG@k across all queries
    """
    if k > predicted.shape[1]:
        k = predicted.shape[1]

    n_queries = ground_truth.shape[0]
    ndcg_scores = np.zeros(n_queries)

    for i in range(n_queries):
        gt_set = set(ground_truth[i])
        dcg = 0.0
        idcg = 0.0

        # Calculate DCG
        for j in range(k):
            if predicted[i, j] in gt_set:
                # Using binary relevance (1 for relevant, 0 for irrelevant)
                # Formula: rel_i / log2(i+2)
                dcg += 1.0 / np.log2(j + 2)

        # Calculate ideal DCG (IDCG)
        # IDCG is the DCG for the perfect ranking (all relevant items at the top)
        for j in range(min(len(gt_set), k)):
            idcg += 1.0 / np.log2(j + 2)

        # Calculate NDCG
        if idcg > 0:
            ndcg_scores[i] = dcg / idcg

    return np.mean(ndcg_scores)


def hit_rate_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    Calculate hit rate at k for vector retrieval.
    Hit rate@k measures the proportion of queries where at least one relevant item
    is retrieved in the top-k results.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Number of results to consider

    Returns:
        Hit rate@k across all queries
    """
    if k > predicted.shape[1]:
        k = predicted.shape[1]

    n_queries = ground_truth.shape[0]
    hits = np.zeros(n_queries, dtype=np.bool_)

    for i in range(n_queries):
        gt_set = set(ground_truth[i])
        pred_set = set(predicted[i, :k])

        # If there's at least one hit in the top-k, count it as a hit
        if len(gt_set.intersection(pred_set)) > 0:
            hits[i] = True

    return np.mean(hits)


def mean_reciprocal_rank(ground_truth: np.ndarray, predicted: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for vector retrieval.
    MRR is the average of reciprocal ranks of the first relevant item for each query.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k: Optional limit on number of predictions to consider

    Returns:
        MRR score across all queries
    """
    n_queries = ground_truth.shape[0]
    if k is None:
        k = predicted.shape[1]
    else:
        k = min(k, predicted.shape[1])

    reciprocal_ranks = np.zeros(n_queries)

    for i in range(n_queries):
        gt_set = set(ground_truth[i])

        # Find the rank of the first relevant item
        for j in range(k):
            if predicted[i, j] in gt_set:
                reciprocal_ranks[i] = 1.0 / (j + 1)
                break

    return np.mean(reciprocal_ranks)


def compute_cost_latency(timing_data: List[float]) -> dict:
    """
    Calculate compute cost metrics based on query latency measurements.

    Args:
        timing_data: List of latency measurements in seconds for each query

    Returns:
        Dictionary containing various latency statistics:
        - mean: Average latency across all queries
        - median: Median latency
        - p95: 95th percentile latency
        - p99: 99th percentile latency
        - min: Minimum latency
        - max: Maximum latency
    """
    timing_array = np.array(timing_data)

    return {
        "mean": np.mean(timing_array),
        "median": np.median(timing_array),
        "p95": np.percentile(timing_array, 95),
        "p99": np.percentile(timing_array, 99),
        "min": np.min(timing_array),
        "max": np.max(timing_array)
    }


def vector_similarity_count(dataset_size: int, query_count: int, algorithm_type: str = "exhaustive") -> int:
    """
    Calculate the number of vector similarity computations performed during retrieval.

    Args:
        dataset_size: Number of vectors in the dataset/index
        query_count: Number of queries performed
        algorithm_type: Type of algorithm used for retrieval. Options:
            - "exhaustive": Compares each query against all vectors in the dataset
            - "approximate": Uses an estimate based on algorithm complexity

    Returns:
        Total number of vector similarity computations performed
    """
    if algorithm_type == "exhaustive":
        # Each query is compared against every vector in the dataset
        return query_count * dataset_size
    elif algorithm_type == "approximate":
        # For approximate algorithms, the number depends on the specific implementation
        # This is a simplified estimate that assumes log(N) complexity
        return query_count * int(np.ceil(np.log2(dataset_size)))
    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
