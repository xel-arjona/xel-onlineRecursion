from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.p2 import P2QuantileState


# =============================================================================
# CONFIGURATION
# =============================================================================

PROBABILITIES = (
    0.01,
    0.05,
    0.50,
    0.95,
    0.99,
)

SAMPLE_SIZES = (
    5,
    10,
    20,
    50,
    100,
    250,
    500,
    1000,
)

TRIALS = 1000
SEED = 1729

RESULTS_PATH = Path(
    "results/normal_extreme_tail.csv"
)

STANDARD_NORMAL = NormalDist(
    mu=0.0,
    sigma=1.0,
)


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class Result:
    probability: float
    sample_size: int

    p2_type7_bias: float
    p2_type7_mae: float
    p2_type7_rmse: float
    p2_type7_p95_abs: float
    p2_type7_max_abs: float

    p2_true_bias: float
    p2_true_mae: float

    type7_true_bias: float
    type7_true_mae: float

    p2_type7_mae_ratio: float


# =============================================================================
# ESTIMATORS
# =============================================================================

def p2_quantile(
    sample: np.ndarray,
    probability: float,
) -> float:

    state = P2QuantileState()

    for observation in sample:
        state.update(
            float(observation),
            probability,
        )

    return state.value


def type7_quantile(
    sample: np.ndarray,
    probability: float,
) -> float:

    return float(
        np.quantile(
            sample,
            probability,
            method="linear",
        )
    )


def true_normal_quantile(
    probability: float,
) -> float:

    return STANDARD_NORMAL.inv_cdf(
        probability
    )


# =============================================================================
# ERROR HELPERS
# =============================================================================

def mae(
    errors: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(errors)
        )
    )


def rmse(
    errors: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                errors * errors
            )
        )
    )


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment() -> list[Result]:

    rng = np.random.default_rng(
        SEED
    )

    results: list[Result] = []

    # Generate the same trial population for all probabilities at a
    # given N. This makes lower/upper-tail comparisons better paired.
    for sample_size in SAMPLE_SIZES:

        p2_type7_errors = {
            p: np.empty(TRIALS)
            for p in PROBABILITIES
        }

        p2_true_errors = {
            p: np.empty(TRIALS)
            for p in PROBABILITIES
        }

        type7_true_errors = {
            p: np.empty(TRIALS)
            for p in PROBABILITIES
        }

        for trial in range(TRIALS):

            sample = rng.normal(
                loc=0.0,
                scale=1.0,
                size=sample_size,
            )

            for probability in PROBABILITIES:

                p2_estimate = p2_quantile(
                    sample,
                    probability,
                )

                type7_estimate = type7_quantile(
                    sample,
                    probability,
                )

                true_quantile = true_normal_quantile(
                    probability
                )

                p2_type7_errors[probability][trial] = (
                    p2_estimate
                    - type7_estimate
                )

                p2_true_errors[probability][trial] = (
                    p2_estimate
                    - true_quantile
                )

                type7_true_errors[probability][trial] = (
                    type7_estimate
                    - true_quantile
                )

        for probability in PROBABILITIES:

            compression_errors = (
                p2_type7_errors[probability]
            )

            p2_population_errors = (
                p2_true_errors[probability]
            )

            type7_population_errors = (
                type7_true_errors[probability]
            )

            compression_abs = np.abs(
                compression_errors
            )

            p2_true_mae = mae(
                p2_population_errors
            )

            type7_true_mae = mae(
                type7_population_errors
            )

            ratio = (
                p2_true_mae / type7_true_mae
                if type7_true_mae > 0.0
                else float("nan")
            )

            results.append(
                Result(
                    probability=probability,
                    sample_size=sample_size,

                    p2_type7_bias=float(
                        np.mean(
                            compression_errors
                        )
                    ),

                    p2_type7_mae=mae(
                        compression_errors
                    ),

                    p2_type7_rmse=rmse(
                        compression_errors
                    ),

                    p2_type7_p95_abs=float(
                        np.quantile(
                            compression_abs,
                            0.95,
                            method="linear",
                        )
                    ),

                    p2_type7_max_abs=float(
                        np.max(
                            compression_abs
                        )
                    ),

                    p2_true_bias=float(
                        np.mean(
                            p2_population_errors
                        )
                    ),

                    p2_true_mae=p2_true_mae,

                    type7_true_bias=float(
                        np.mean(
                            type7_population_errors
                        )
                    ),

                    type7_true_mae=type7_true_mae,

                    p2_type7_mae_ratio=ratio,
                )
            )

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(
    results: list[Result],
) -> None:

    print(
        "\n"
        "P² Gaussian Extreme-Tail Validation\n"
        "Distribution: Normal(0,1)\n"
        f"Trials per cell: {TRIALS}\n"
        f"Seed: {SEED}\n"
    )

    header = (
        f"{'p':>6}"
        f"{'N':>8}"
        f"{'P2-T7 MAE':>14}"
        f"{'P95 |err|':>14}"
        f"{'P2 True MAE':>14}"
        f"{'T7 True MAE':>14}"
        f"{'P2/T7':>10}"
        f"{'P2 Bias':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        print(
            f"{result.probability:6.2f}"
            f"{result.sample_size:8d}"
            f"{result.p2_type7_mae:14.6f}"
            f"{result.p2_type7_p95_abs:14.6f}"
            f"{result.p2_true_mae:14.6f}"
            f"{result.type7_true_mae:14.6f}"
            f"{result.p2_type7_mae_ratio:10.4f}"
            f"{result.p2_true_bias:12.6f}"
        )


def save_csv(
    results: list[Result],
) -> None:

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "probability",
                "sample_size",
                "p2_type7_bias",
                "p2_type7_mae",
                "p2_type7_rmse",
                "p2_type7_p95_abs",
                "p2_type7_max_abs",
                "p2_true_bias",
                "p2_true_mae",
                "type7_true_bias",
                "type7_true_mae",
                "p2_type7_mae_ratio",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.probability,
                    result.sample_size,
                    result.p2_type7_bias,
                    result.p2_type7_mae,
                    result.p2_type7_rmse,
                    result.p2_type7_p95_abs,
                    result.p2_type7_max_abs,
                    result.p2_true_bias,
                    result.p2_true_mae,
                    result.type7_true_bias,
                    result.type7_true_mae,
                    result.p2_type7_mae_ratio,
                ]
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    results = run_experiment()

    print_results(
        results
    )

    save_csv(
        results
    )

    print(
        "\n"
        f"Saved results to: {RESULTS_PATH}"
        "\n"
    )


if __name__ == "__main__":
    main()
