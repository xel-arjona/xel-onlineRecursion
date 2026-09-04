from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from src.adaptive_quantile import (
    AdaptiveQuantileState,
)
from src.p2 import (
    P2QuantileState,
)


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

CHECKPOINTS = (
    20,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
)

STREAM_TRIALS = 80


# -----------------------------------------------------------------------------
# Order/path-dependence experiment
# -----------------------------------------------------------------------------

ORDER_SAMPLE_SIZE = 1000

ORDER_POPULATIONS = 8

PERMUTATIONS_PER_POPULATION = 60


SEED = 1729


CHECKPOINT_RESULTS_PATH = Path(
    "results/p2_vs_adaptive_cumulative.csv"
)

ORDER_RESULTS_PATH = Path(
    "results/p2_vs_adaptive_order.csv"
)


# =============================================================================
# DISTRIBUTION REGISTRY
# =============================================================================

@dataclass(frozen=True)
class Family:
    name: str

    seed_offset: int

    generator: Callable[
        [
            np.random.Generator,
            int,
        ],
        np.ndarray,
    ]

    true_quantile: Callable[
        [float],
        float,
    ]


# =============================================================================
# NORMAL(0,1)
# =============================================================================

def generate_normal(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=sample_size,
    )


def normal_true_quantile(
    probability: float,
) -> float:

    return float(
        norm.ppf(
            probability
        )
    )


# =============================================================================
# UNIT-VARIANCE STUDENT-t(df=3)
# =============================================================================

STUDENT_T3_SCALE = float(
    np.sqrt(
        3.0
        / (
            3.0
            - 2.0
        )
    )
)


def generate_student_t3(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return (
        rng.standard_t(
            df=3,
            size=sample_size,
        )
        / STUDENT_T3_SCALE
    )


def student_t3_true_quantile(
    probability: float,
) -> float:

    return float(
        student_t.ppf(
            probability,
            df=3,
        )
        / STUDENT_T3_SCALE
    )


FAMILIES = (
    Family(
        name="Normal(0,1)",
        seed_offset=100,
        generator=generate_normal,
        true_quantile=normal_true_quantile,
    ),

    Family(
        name="Student-t(df=3), unit variance",
        seed_offset=200,
        generator=generate_student_t3,
        true_quantile=student_t3_true_quantile,
    ),
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class CheckpointResult:
    family: str

    probability: float
    sample_size: int

    p2_type7_bias: float
    p2_type7_mae: float

    adaptive_type7_bias: float
    adaptive_type7_mae: float

    adaptive_over_p2_type7_mae: float

    p2_true_bias: float
    p2_true_mae: float

    adaptive_true_bias: float
    adaptive_true_mae: float

    adaptive_over_p2_true_mae: float


@dataclass
class OrderResult:
    family: str

    probability: float
    sample_size: int

    p2_permutation_sd: float
    adaptive_permutation_sd: float
    adaptive_over_p2_sd: float

    p2_permutation_range: float
    adaptive_permutation_range: float

    p2_permutation_type7_mae: float
    adaptive_permutation_type7_mae: float

    p2_adversarial_error: float
    adaptive_adversarial_error: float


# =============================================================================
# BASIC HELPERS
# =============================================================================

def mean(
    values: list[float],
) -> float:

    return float(
        np.mean(
            values
        )
    )


def mae(
    values: list[float],
) -> float:

    return float(
        np.mean(
            np.abs(
                values
            )
        )
    )


def ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0.0:
        return float(
            "nan"
        )

    return (
        numerator
        / denominator
    )


# =============================================================================
# CUMULATIVE GAIN SCHEDULE
#
# This is the bridge experiment:
#
#       alpha_N = 1 / N
#
# It is the count-normalized cumulative schedule.
#
# It DOES NOT make AdaptiveQuantile mathematically identical to P².
# The experiment is explicitly designed to measure that difference.
# =============================================================================

def cumulative_alpha(
    observation_count: int,
) -> float:

    return (
        1.0
        / float(
            observation_count
        )
    )


# =============================================================================
# RUN BOTH FIXED-MEMORY ESTIMATORS OVER ONE COMPLETE STREAM
# =============================================================================

def run_estimators(
    sample: np.ndarray,
) -> tuple[
    dict[float, float],
    dict[float, float],
]:

    p2_states = {
        probability:
            P2QuantileState()
        for probability
        in PROBABILITIES
    }

    adaptive_states = {
        probability:
            AdaptiveQuantileState()
        for probability
        in PROBABILITIES
    }

    for observation_count, raw_observation in enumerate(
        sample,
        start=1,
    ):

        observation = float(
            raw_observation
        )

        alpha = cumulative_alpha(
            observation_count
        )

        for probability in PROBABILITIES:

            p2_states[
                probability
            ].update(
                observation,
                probability,
            )

            adaptive_states[
                probability
            ].update(
                observation,
                probability,
                alpha,
            )

    p2_estimates = {
        probability:
            p2_states[
                probability
            ].value
        for probability
        in PROBABILITIES
    }

    adaptive_estimates = {
        probability:
            adaptive_states[
                probability
            ].value
        for probability
        in PROBABILITIES
    }

    return (
        p2_estimates,
        adaptive_estimates,
    )


# =============================================================================
# CHECKPOINT EXPERIMENT
#
# SAME STREAM:
#
#                         ┌── P²
#       x_1 ... x_N ──────┤
#                         └── AdaptiveQuantile(alpha=1/N)
#
# FULL-MEMORY REFERENCES:
#
#       Type-7 empirical sample quantile
#       known true population quantile
#
# This distinguishes:
#
#       empirical-quantile approximation
#
# from:
#
#       population-quantile estimation
# =============================================================================

def run_checkpoint_experiment(
) -> list[CheckpointResult]:

    maximum_sample_size = max(
        CHECKPOINTS
    )

    checkpoint_set = set(
        CHECKPOINTS
    )

    results: list[CheckpointResult] = []

    for family in FAMILIES:

        rng = np.random.default_rng(
            SEED
            + family.seed_offset
        )

        true_quantiles = {
            probability:
                family.true_quantile(
                    probability
                )
            for probability
            in PROBABILITIES
        }

        errors = {
            (
                probability,
                sample_size,
            ): {
                "p2_type7": [],
                "adaptive_type7": [],
                "p2_true": [],
                "adaptive_true": [],
            }
            for probability
            in PROBABILITIES
            for sample_size
            in CHECKPOINTS
        }

        for _ in range(
            STREAM_TRIALS
        ):

            sample = family.generator(
                rng,
                maximum_sample_size,
            )

            p2_states = {
                probability:
                    P2QuantileState()
                for probability
                in PROBABILITIES
            }

            adaptive_states = {
                probability:
                    AdaptiveQuantileState()
                for probability
                in PROBABILITIES
            }

            for observation_count, raw_observation in enumerate(
                sample,
                start=1,
            ):

                observation = float(
                    raw_observation
                )

                alpha = cumulative_alpha(
                    observation_count
                )

                for probability in PROBABILITIES:

                    p2_states[
                        probability
                    ].update(
                        observation,
                        probability,
                    )

                    adaptive_states[
                        probability
                    ].update(
                        observation,
                        probability,
                        alpha,
                    )

                if (
                    observation_count
                    not in checkpoint_set
                ):
                    continue

                sample_prefix = sample[
                    :observation_count
                ]

                type7_values = np.quantile(
                    sample_prefix,
                    PROBABILITIES,
                    method="linear",
                )

                for index, probability in enumerate(
                    PROBABILITIES
                ):

                    type7_estimate = float(
                        type7_values[
                            index
                        ]
                    )

                    p2_estimate = (
                        p2_states[
                            probability
                        ].value
                    )

                    adaptive_estimate = (
                        adaptive_states[
                            probability
                        ].value
                    )

                    population_quantile = (
                        true_quantiles[
                            probability
                        ]
                    )

                    cell = errors[
                        (
                            probability,
                            observation_count,
                        )
                    ]

                    cell[
                        "p2_type7"
                    ].append(
                        p2_estimate
                        - type7_estimate
                    )

                    cell[
                        "adaptive_type7"
                    ].append(
                        adaptive_estimate
                        - type7_estimate
                    )

                    cell[
                        "p2_true"
                    ].append(
                        p2_estimate
                        - population_quantile
                    )

                    cell[
                        "adaptive_true"
                    ].append(
                        adaptive_estimate
                        - population_quantile
                    )

        for probability in PROBABILITIES:

            for sample_size in CHECKPOINTS:

                cell = errors[
                    (
                        probability,
                        sample_size,
                    )
                ]

                p2_type7_mae = mae(
                    cell[
                        "p2_type7"
                    ]
                )

                adaptive_type7_mae = mae(
                    cell[
                        "adaptive_type7"
                    ]
                )

                p2_true_mae = mae(
                    cell[
                        "p2_true"
                    ]
                )

                adaptive_true_mae = mae(
                    cell[
                        "adaptive_true"
                    ]
                )

                results.append(
                    CheckpointResult(
                        family=family.name,

                        probability=probability,
                        sample_size=sample_size,

                        p2_type7_bias=mean(
                            cell[
                                "p2_type7"
                            ]
                        ),

                        p2_type7_mae=(
                            p2_type7_mae
                        ),

                        adaptive_type7_bias=mean(
                            cell[
                                "adaptive_type7"
                            ]
                        ),

                        adaptive_type7_mae=(
                            adaptive_type7_mae
                        ),

                        adaptive_over_p2_type7_mae=ratio(
                            adaptive_type7_mae,
                            p2_type7_mae,
                        ),

                        p2_true_bias=mean(
                            cell[
                                "p2_true"
                            ]
                        ),

                        p2_true_mae=(
                            p2_true_mae
                        ),

                        adaptive_true_bias=mean(
                            cell[
                                "adaptive_true"
                            ]
                        ),

                        adaptive_true_mae=(
                            adaptive_true_mae
                        ),

                        adaptive_over_p2_true_mae=ratio(
                            adaptive_true_mae,
                            p2_true_mae,
                        ),
                    )
                )

    return results


# =============================================================================
# ORDER / PATH-DEPENDENCE EXPERIMENT
#
# For each fixed multiset:
#
#       exact Type 7
#
# is invariant under permutation.
#
# We then feed the identical observations to both recursive estimators in:
#
#       many random permutations
#       strictly ascending order
#       strictly descending order
#
# This directly measures how much final output depends on observation path.
# =============================================================================

def run_order_experiment(
) -> list[OrderResult]:

    results: list[OrderResult] = []

    for family in FAMILIES:

        rng = np.random.default_rng(
            SEED
            + 10000
            + family.seed_offset
        )

        aggregate = {
            probability: {
                "p2_sd": [],
                "adaptive_sd": [],

                "p2_range": [],
                "adaptive_range": [],

                "p2_perm_mae": [],
                "adaptive_perm_mae": [],

                "p2_adversarial": [],
                "adaptive_adversarial": [],
            }
            for probability
            in PROBABILITIES
        }

        for _ in range(
            ORDER_POPULATIONS
        ):

            population = family.generator(
                rng,
                ORDER_SAMPLE_SIZE,
            )

            type7_values = np.quantile(
                population,
                PROBABILITIES,
                method="linear",
            )

            type7 = {
                probability:
                    float(
                        type7_values[
                            index
                        ]
                    )
                for index, probability
                in enumerate(
                    PROBABILITIES
                )
            }

            p2_permutations = {
                probability: []
                for probability
                in PROBABILITIES
            }

            adaptive_permutations = {
                probability: []
                for probability
                in PROBABILITIES
            }

            # -----------------------------------------------------------------
            # RANDOM PERMUTATIONS
            # -----------------------------------------------------------------

            for _ in range(
                PERMUTATIONS_PER_POPULATION
            ):

                permutation = rng.permutation(
                    population
                )

                (
                    p2_estimates,
                    adaptive_estimates,
                ) = run_estimators(
                    permutation
                )

                for probability in PROBABILITIES:

                    p2_permutations[
                        probability
                    ].append(
                        p2_estimates[
                            probability
                        ]
                    )

                    adaptive_permutations[
                        probability
                    ].append(
                        adaptive_estimates[
                            probability
                        ]
                    )

            # -----------------------------------------------------------------
            # ADVERSARIAL MONOTONIC ORDERS
            # -----------------------------------------------------------------

            ascending = np.sort(
                population
            )

            descending = ascending[
                ::-1
            ]

            (
                p2_ascending,
                adaptive_ascending,
            ) = run_estimators(
                ascending
            )

            (
                p2_descending,
                adaptive_descending,
            ) = run_estimators(
                descending
            )

            # -----------------------------------------------------------------
            # POPULATION-LEVEL PATH METRICS
            # -----------------------------------------------------------------

            for probability in PROBABILITIES:

                p2_values = np.asarray(
                    p2_permutations[
                        probability
                    ],
                    dtype=float,
                )

                adaptive_values = np.asarray(
                    adaptive_permutations[
                        probability
                    ],
                    dtype=float,
                )

                reference = type7[
                    probability
                ]

                p2_adversarial_error = max(
                    abs(
                        p2_ascending[
                            probability
                        ]
                        - reference
                    ),
                    abs(
                        p2_descending[
                            probability
                        ]
                        - reference
                    ),
                )

                adaptive_adversarial_error = max(
                    abs(
                        adaptive_ascending[
                            probability
                        ]
                        - reference
                    ),
                    abs(
                        adaptive_descending[
                            probability
                        ]
                        - reference
                    ),
                )

                cell = aggregate[
                    probability
                ]

                cell[
                    "p2_sd"
                ].append(
                    float(
                        np.std(
                            p2_values,
                            ddof=1,
                        )
                    )
                )

                cell[
                    "adaptive_sd"
                ].append(
                    float(
                        np.std(
                            adaptive_values,
                            ddof=1,
                        )
                    )
                )

                cell[
                    "p2_range"
                ].append(
                    float(
                        np.max(
                            p2_values
                        )
                        - np.min(
                            p2_values
                        )
                    )
                )

                cell[
                    "adaptive_range"
                ].append(
                    float(
                        np.max(
                            adaptive_values
                        )
                        - np.min(
                            adaptive_values
                        )
                    )
                )

                cell[
                    "p2_perm_mae"
                ].append(
                    float(
                        np.mean(
                            np.abs(
                                p2_values
                                - reference
                            )
                        )
                    )
                )

                cell[
                    "adaptive_perm_mae"
                ].append(
                    float(
                        np.mean(
                            np.abs(
                                adaptive_values
                                - reference
                            )
                        )
                    )
                )

                cell[
                    "p2_adversarial"
                ].append(
                    p2_adversarial_error
                )

                cell[
                    "adaptive_adversarial"
                ].append(
                    adaptive_adversarial_error
                )

        for probability in PROBABILITIES:

            cell = aggregate[
                probability
            ]

            p2_sd = mean(
                cell[
                    "p2_sd"
                ]
            )

            adaptive_sd = mean(
                cell[
                    "adaptive_sd"
                ]
            )

            results.append(
                OrderResult(
                    family=family.name,

                    probability=probability,
                    sample_size=ORDER_SAMPLE_SIZE,

                    p2_permutation_sd=(
                        p2_sd
                    ),

                    adaptive_permutation_sd=(
                        adaptive_sd
                    ),

                    adaptive_over_p2_sd=ratio(
                        adaptive_sd,
                        p2_sd,
                    ),

                    p2_permutation_range=mean(
                        cell[
                            "p2_range"
                        ]
                    ),

                    adaptive_permutation_range=mean(
                        cell[
                            "adaptive_range"
                        ]
                    ),

                    p2_permutation_type7_mae=mean(
                        cell[
                            "p2_perm_mae"
                        ]
                    ),

                    adaptive_permutation_type7_mae=mean(
                        cell[
                            "adaptive_perm_mae"
                        ]
                    ),

                    p2_adversarial_error=mean(
                        cell[
                            "p2_adversarial"
                        ]
                    ),

                    adaptive_adversarial_error=mean(
                        cell[
                            "adaptive_adversarial"
                        ]
                    ),
                )
            )

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_checkpoint_results(
    results: list[CheckpointResult],
) -> None:

    for family in FAMILIES:

        print(
            "\n"
            "P² vs AdaptiveQuantile(alpha = 1/N)\n"
            f"Distribution: {family.name}\n"
            f"Trials: {STREAM_TRIALS}\n"
        )

        header = (
            f"{'p':>6}"
            f"{'N':>8}"
            f"{'P2-T7':>11}"
            f"{'AQ-T7':>11}"
            f"{'AQ/P2 E':>10}"
            f"{'P2-True':>11}"
            f"{'AQ-True':>11}"
            f"{'AQ/P2 T':>10}"
            f"{'AQ Bias':>11}"
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
                f"{result.p2_type7_mae:11.6f}"
                f"{result.adaptive_type7_mae:11.6f}"
                f"{result.adaptive_over_p2_type7_mae:10.3f}"
                f"{result.p2_true_mae:11.6f}"
                f"{result.adaptive_true_mae:11.6f}"
                f"{result.adaptive_over_p2_true_mae:10.3f}"
                f"{result.adaptive_true_bias:11.6f}"
            )


def print_order_results(
    results: list[OrderResult],
) -> None:

    for family in FAMILIES:

        print(
            "\n"
            "Order / Path-Dependence Validation\n"
            f"Distribution: {family.name}\n"
            f"Fixed population N: {ORDER_SAMPLE_SIZE}\n"
            f"Independent populations: {ORDER_POPULATIONS}\n"
            f"Random permutations each: {PERMUTATIONS_PER_POPULATION}\n"
        )

        header = (
            f"{'p':>6}"
            f"{'P2 SD':>11}"
            f"{'AQ SD':>11}"
            f"{'AQ/P2 SD':>11}"
            f"{'P2 Perm':>11}"
            f"{'AQ Perm':>11}"
            f"{'P2 Adv':>11}"
            f"{'AQ Adv':>11}"
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
                f"{result.p2_permutation_sd:11.6f}"
                f"{result.adaptive_permutation_sd:11.6f}"
                f"{result.adaptive_over_p2_sd:11.3f}"
                f"{result.p2_permutation_type7_mae:11.6f}"
                f"{result.adaptive_permutation_type7_mae:11.6f}"
                f"{result.p2_adversarial_error:11.6f}"
                f"{result.adaptive_adversarial_error:11.6f}"
            )


# =============================================================================
# CSV
# =============================================================================

def save_dataclass_csv(
    path: Path,
    rows: list,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        asdict(
            rows[0]
        ).keys()
    )

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                asdict(
                    row
                )
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    checkpoint_results = (
        run_checkpoint_experiment()
    )

    order_results = (
        run_order_experiment()
    )

    print_checkpoint_results(
        checkpoint_results
    )

    print_order_results(
        order_results
    )

    save_dataclass_csv(
        CHECKPOINT_RESULTS_PATH,
        checkpoint_results,
    )

    save_dataclass_csv(
        ORDER_RESULTS_PATH,
        order_results,
    )

    print(
        "\n"
        f"Saved cumulative comparison to: "
        f"{CHECKPOINT_RESULTS_PATH}\n"
        f"Saved order comparison to: "
        f"{ORDER_RESULTS_PATH}\n"
    )


if __name__ == "__main__":
    main()
