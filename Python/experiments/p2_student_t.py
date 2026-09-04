from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from src.p2 import P2QuantileState


# =============================================================================
# CONFIGURATION
# =============================================================================

DEGREES_OF_FREEDOM = (
    3,
    5,
)

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
    "results/student_t_validation.csv"
)


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class Result:
    degrees_of_freedom: int
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


# =============================================================================
# UNIT-VARIANCE STUDENT-t
# =============================================================================

def standard_deviation(
    degrees_of_freedom: int,
) -> float:
    """
    Standard deviation of a conventional Student-t(df)
    with scale parameter 1.

        Var(T) = df / (df - 2)

    Both df=3 and df=5 therefore have finite variance.
    """

    return float(
        np.sqrt(
            degrees_of_freedom
            / (degrees_of_freedom - 2.0)
        )
    )


def generate_sample(
    rng: np.random.Generator,
    degrees_of_freedom: int,
    sample_size: int,
) -> np.ndarray:
    """
    Generate Student-t observations standardized to unit variance.
    """

    raw = rng.standard_t(
        df=degrees_of_freedom,
        size=sample_size,
    )

    return (
        raw
        / standard_deviation(
            degrees_of_freedom
        )
    )


def true_quantile(
    degrees_of_freedom: int,
    probability: float,
) -> float:
    """
    Exact quantile of the unit-variance Student-t population.
    """

    raw_quantile = student_t.ppf(
        probability,
        df=degrees_of_freedom,
    )

    return float(
        raw_quantile
        / standard_deviation(
            degrees_of_freedom
        )
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

    results: list[Result] = []

    for degrees_of_freedom in DEGREES_OF_FREEDOM:

        # Independent but reproducible RNG stream for each distribution.
        rng = np.random.default_rng(
            SEED + degrees_of_freedom
        )

        for sample_size in SAMPLE_SIZES:

            p2_type7_errors = {
                p: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for p in PROBABILITIES
            }

            p2_true_errors = {
                p: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for p in PROBABILITIES
            }

            type7_true_errors = {
                p: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for p in PROBABILITIES
            }

            for trial in range(TRIALS):

                sample = generate_sample(
                    rng,
                    degrees_of_freedom,
                    sample_size,
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

                    population_quantile = true_quantile(
                        degrees_of_freedom,
                        probability,
                    )

                    p2_type7_errors[
                        probability
                    ][trial] = (
                        p2_estimate
                        - type7_estimate
                    )

                    p2_true_errors[
                        probability
                    ][trial] = (
                        p2_estimate
                        - population_quantile
                    )

                    type7_true_errors[
                        probability
                    ][trial] = (
                        type7_estimate
                        - population_quantile
                    )

            for probability in PROBABILITIES:

                compression_errors = (
                    p2_type7_errors[
                        probability
                    ]
                )

                p2_population_errors = (
                    p2_true_errors[
                        probability
                    ]
                )

                type7_population_errors = (
                    type7_true_errors[
                        probability
                    ]
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
                    p2_true_mae
                    / type7_true_mae
                    if type7_true_mae > 0.0
                    else float("nan")
                )

                results.append(
                    Result(
                        degrees_of_freedom=(
                            degrees_of_freedom
                        ),

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

                        type7_true_mae=(
                            type7_true_mae
                        ),

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

    for degrees_of_freedom in DEGREES_OF_FREEDOM:

        print(
            "\n"
            "P² Heavy-Tail Validation\n"
            f"Distribution: unit-variance Student-t(df={degrees_of_freedom})\n"
            f"Trials per cell: {TRIALS}\n"
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

            if (
                result.degrees_of_freedom
                != degrees_of_freedom
            ):
                continue

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
                "degrees_of_freedom",
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
                    result.degrees_of_freedom,
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
