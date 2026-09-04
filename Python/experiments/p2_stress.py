from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

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
    "results/p2_stress_validation.csv"
)


# =============================================================================
# DISCRETE POPULATION
# =============================================================================

DISCRETE_VALUES = np.array(
    [
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
    ]
)

DISCRETE_PROBABILITIES = np.array(
    [
        0.02,
        0.18,
        0.60,
        0.18,
        0.02,
    ]
)

DISCRETE_CDF = np.cumsum(
    DISCRETE_PROBABILITIES
)


# =============================================================================
# CONTAMINATED GAUSSIAN
# =============================================================================

CONTAMINATION_PROBABILITY = 0.01
OUTLIER_SIGMA = 10.0

CONTAMINATED_SIGMA = np.sqrt(
    (
        1.0
        - CONTAMINATION_PROBABILITY
    )
    + (
        CONTAMINATION_PROBABILITY
        * OUTLIER_SIGMA**2
    )
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass(frozen=True)
class Family:
    name: str
    seed_offset: int
    generator: Callable[
        [np.random.Generator, int],
        np.ndarray,
    ]
    true_quantile: Callable[
        [float],
        float,
    ]


@dataclass
class Result:
    family: str
    probability: float
    sample_size: int

    true_quantile: float

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
# P² AND TYPE-7 ESTIMATORS
# =============================================================================

def p2_quantiles(
    sample: np.ndarray,
) -> dict[float, float]:
    """
    Evaluate all configured P² quantiles in one traversal
    through the population.
    """

    states = {
        probability: P2QuantileState()
        for probability in PROBABILITIES
    }

    for observation in sample:

        value = float(
            observation
        )

        for probability, state in states.items():

            state.update(
                value,
                probability,
            )

    return {
        probability: state.value
        for probability, state in states.items()
    }


def type7_quantiles(
    sample: np.ndarray,
) -> dict[float, float]:
    """
    Full-memory Hyndman-Fan Type-7 empirical quantiles.
    """

    estimates = np.quantile(
        sample,
        PROBABILITIES,
        method="linear",
    )

    return {
        probability: float(
            estimates[index]
        )
        for index, probability
        in enumerate(PROBABILITIES)
    }


# =============================================================================
# FAMILY 1
#
# SHIFTED EXPONENTIAL
#
# X ~ Exponential(1) - 1
#
# Mean     = 0
# Variance = 1
#
# Strongly right-skewed continuous population.
# =============================================================================

def generate_exponential(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return (
        rng.exponential(
            scale=1.0,
            size=sample_size,
        )
        - 1.0
    )


def exponential_true_quantile(
    probability: float,
) -> float:

    return float(
        -np.log1p(
            -probability
        )
        - 1.0
    )


# =============================================================================
# FAMILY 2
#
# DISCRETE / REPEATED POPULATION
#
# This deliberately produces many ties.
#
# The population quantile uses:
#
#     Q(p) = inf{x : F(x) >= p}
#
# Type-7 and P² may interpolate between support values. That is expected
# and is part of what this experiment is designed to expose.
# =============================================================================

def generate_discrete(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return rng.choice(
        DISCRETE_VALUES,
        size=sample_size,
        p=DISCRETE_PROBABILITIES,
    )


def discrete_true_quantile(
    probability: float,
) -> float:

    index = int(
        np.searchsorted(
            DISCRETE_CDF,
            probability,
            side="left",
        )
    )

    return float(
        DISCRETE_VALUES[index]
    )


# =============================================================================
# FAMILY 3
#
# CONTAMINATED GAUSSIAN
#
# 99% N(0,1)
#  1% N(0,10)
#
# Then standardized back to variance 1.
#
# Symmetric distribution with rare extreme observations.
# =============================================================================

def generate_contaminated_gaussian(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    sample = rng.normal(
        loc=0.0,
        scale=1.0,
        size=sample_size,
    )

    contaminated = (
        rng.random(
            sample_size
        )
        < CONTAMINATION_PROBABILITY
    )

    contamination_count = int(
        np.sum(
            contaminated
        )
    )

    if contamination_count > 0:

        sample[contaminated] = rng.normal(
            loc=0.0,
            scale=OUTLIER_SIGMA,
            size=contamination_count,
        )

    return (
        sample
        / CONTAMINATED_SIGMA
    )


def contaminated_gaussian_cdf(
    standardized_value: float,
) -> float:

    raw_value = (
        standardized_value
        * CONTAMINATED_SIGMA
    )

    core_probability = norm.cdf(
        raw_value
    )

    outlier_probability = norm.cdf(
        raw_value
        / OUTLIER_SIGMA
    )

    return float(
        (
            1.0
            - CONTAMINATION_PROBABILITY
        )
        * core_probability
        + CONTAMINATION_PROBABILITY
        * outlier_probability
    )


def contaminated_gaussian_true_quantile(
    probability: float,
) -> float:

    return float(
        brentq(
            lambda value:
                contaminated_gaussian_cdf(
                    value
                )
                - probability,
            -20.0,
            20.0,
        )
    )


# =============================================================================
# DISTRIBUTION REGISTRY
# =============================================================================

FAMILIES = (
    Family(
        name="Shifted Exponential",
        seed_offset=100,
        generator=generate_exponential,
        true_quantile=exponential_true_quantile,
    ),

    Family(
        name="Discrete Repeated",
        seed_offset=200,
        generator=generate_discrete,
        true_quantile=discrete_true_quantile,
    ),

    Family(
        name="Contaminated Gaussian",
        seed_offset=300,
        generator=generate_contaminated_gaussian,
        true_quantile=contaminated_gaussian_true_quantile,
    ),
)


# =============================================================================
# ERROR HELPERS
# =============================================================================

def mae(
    errors: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(
                errors
            )
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
# MONTE-CARLO EXPERIMENT
# =============================================================================

def run_experiment() -> list[Result]:

    results: list[Result] = []

    for family in FAMILIES:

        rng = np.random.default_rng(
            SEED
            + family.seed_offset
        )

        population_quantiles = {
            probability:
                family.true_quantile(
                    probability
                )
            for probability
            in PROBABILITIES
        }

        for sample_size in SAMPLE_SIZES:

            p2_type7_errors = {
                probability: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for probability
                in PROBABILITIES
            }

            p2_true_errors = {
                probability: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for probability
                in PROBABILITIES
            }

            type7_true_errors = {
                probability: np.empty(
                    TRIALS,
                    dtype=float,
                )
                for probability
                in PROBABILITIES
            }

            for trial in range(TRIALS):

                sample = family.generator(
                    rng,
                    sample_size,
                )

                p2_estimates = p2_quantiles(
                    sample
                )

                type7_estimates = type7_quantiles(
                    sample
                )

                for probability in PROBABILITIES:

                    p2_estimate = (
                        p2_estimates[
                            probability
                        ]
                    )

                    type7_estimate = (
                        type7_estimates[
                            probability
                        ]
                    )

                    population_quantile = (
                        population_quantiles[
                            probability
                        ]
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

                compression_absolute = np.abs(
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
                        family=family.name,
                        probability=probability,
                        sample_size=sample_size,

                        true_quantile=(
                            population_quantiles[
                                probability
                            ]
                        ),

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
                                compression_absolute,
                                0.95,
                                method="linear",
                            )
                        ),

                        p2_type7_max_abs=float(
                            np.max(
                                compression_absolute
                            )
                        ),

                        p2_true_bias=float(
                            np.mean(
                                p2_population_errors
                            )
                        ),

                        p2_true_mae=(
                            p2_true_mae
                        ),

                        type7_true_bias=float(
                            np.mean(
                                type7_population_errors
                            )
                        ),

                        type7_true_mae=(
                            type7_true_mae
                        ),

                        p2_type7_mae_ratio=(
                            ratio
                        ),
                    )
                )

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(
    results: list[Result],
) -> None:

    for family in FAMILIES:

        print(
            "\n"
            "P² Stress Validation\n"
            f"Distribution: {family.name}\n"
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

        print(
            header
        )

        print(
            "-"
            * len(
                header
            )
        )

        for result in results:

            if result.family != family.name:
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
                "family",
                "probability",
                "sample_size",
                "true_quantile",
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
                    result.family,
                    result.probability,
                    result.sample_size,
                    result.true_quantile,
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
