from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from src.adaptive_quantile import (
    AdaptiveQuantileState,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROBABILITIES = (
    0.05,
    0.50,
    0.95,
)

ALPHAS = (
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
)


# -----------------------------------------------------------------------------
# Stationary experiment
# -----------------------------------------------------------------------------

STATIONARY_TRIALS = 60
STATIONARY_LENGTH = 3500
STATIONARY_BURN_IN = 2000


# -----------------------------------------------------------------------------
# Regime-shift experiment
#
# Normal(0,1)
#       ↓
# Normal(+2,1)
# -----------------------------------------------------------------------------

SHIFT_TRIALS = 120

SHIFT_PRE_LENGTH = 2000
SHIFT_POST_LENGTH = 1000

SHIFT_LOCATION = 2.0

SHIFT_LATE_WINDOW = 200


SEED = 1729


STATIONARY_RESULTS_PATH = Path(
    "results/adaptive_quantile_stationary.csv"
)

SHIFT_RESULTS_PATH = Path(
    "results/adaptive_quantile_shift.csv"
)


# =============================================================================
# DISTRIBUTION TYPES
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

def student_t_scale(
    degrees_of_freedom: int,
) -> float:

    return float(
        np.sqrt(
            degrees_of_freedom
            / (
                degrees_of_freedom
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
        / student_t_scale(
            3
        )
    )


def student_t3_true_quantile(
    probability: float,
) -> float:

    return float(
        student_t.ppf(
            probability,
            df=3,
        )
        / student_t_scale(
            3
        )
    )


# =============================================================================
# UNIT-VARIANCE STUDENT-t(df=5)
# =============================================================================

def generate_student_t5(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return (
        rng.standard_t(
            df=5,
            size=sample_size,
        )
        / student_t_scale(
            5
        )
    )


def student_t5_true_quantile(
    probability: float,
) -> float:

    return float(
        student_t.ppf(
            probability,
            df=5,
        )
        / student_t_scale(
            5
        )
    )


# =============================================================================
# DISTRIBUTION REGISTRY
# =============================================================================

FAMILIES = (
    Family(
        name="Normal(0,1)",
        seed_offset=100,
        generator=generate_normal,
        true_quantile=normal_true_quantile,
    ),

    Family(
        name="Student-t(df=5), unit variance",
        seed_offset=200,
        generator=generate_student_t5,
        true_quantile=student_t5_true_quantile,
    ),

    Family(
        name="Student-t(df=3), unit variance",
        seed_offset=300,
        generator=generate_student_t3,
        true_quantile=student_t3_true_quantile,
    ),
)


# =============================================================================
# ONLINE METRIC ACCUMULATOR
# =============================================================================

@dataclass
class Metrics:
    count: int = 0

    error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    coverage_count: int = 0

    scale_sum: float = 0.0

    def add(
        self,
        observation: float,
        estimate: float,
        scale: float,
        true_quantile: float,
    ) -> None:

        error = (
            estimate
            - true_quantile
        )

        self.count += 1

        self.error_sum += (
            error
        )

        self.absolute_error_sum += abs(
            error
        )

        self.squared_error_sum += (
            error
            * error
        )

        if observation <= estimate:
            self.coverage_count += 1

        self.scale_sum += scale

    def bias(
        self,
    ) -> float:

        return (
            self.error_sum
            / self.count
        )

    def mae(
        self,
    ) -> float:

        return (
            self.absolute_error_sum
            / self.count
        )

    def rmse(
        self,
    ) -> float:

        return float(
            np.sqrt(
                self.squared_error_sum
                / self.count
            )
        )

    def coverage(
        self,
    ) -> float:

        return (
            self.coverage_count
            / self.count
        )

    def mean_scale(
        self,
    ) -> float:

        return (
            self.scale_sum
            / self.count
        )


# =============================================================================
# STATIONARY RESULT
# =============================================================================

@dataclass
class StationaryResult:
    family: str

    probability: float
    alpha: float

    true_quantile: float

    bias: float
    mae: float
    rmse: float

    coverage: float
    coverage_error: float

    mean_scale: float


# =============================================================================
# STATIONARY EXPERIMENT
#
# IMPORTANT:
#
# Metrics use q_(t-1) against x_t BEFORE the current observation updates
# the estimator.
#
# This preserves the actual online interpretation:
#
#       state known from past
#               ↓
#       current observation arrives
#               ↓
#       evaluate coverage / tracking
#               ↓
#       update state
# =============================================================================

def run_stationary_experiment(
) -> list[StationaryResult]:

    results: list[StationaryResult] = []

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

        metrics = {
            (
                probability,
                alpha,
            ): Metrics()
            for probability
            in PROBABILITIES
            for alpha
            in ALPHAS
        }

        for _ in range(
            STATIONARY_TRIALS
        ):

            states = {
                (
                    probability,
                    alpha,
                ):
                    AdaptiveQuantileState()
                for probability
                in PROBABILITIES
                for alpha
                in ALPHAS
            }

            sample = family.generator(
                rng,
                STATIONARY_LENGTH,
            )

            for index, raw_observation in enumerate(
                sample
            ):

                observation = float(
                    raw_observation
                )

                for (
                    probability,
                    alpha,
                ), state in states.items():

                    # -------------------------------------------------
                    # Evaluate the estimator BEFORE the current x_t
                    # updates q or scale.
                    # -------------------------------------------------

                    if (
                        index
                        >= STATIONARY_BURN_IN
                        and not isnan(
                            state.value
                        )
                    ):

                        metrics[
                            (
                                probability,
                                alpha,
                            )
                        ].add(
                            observation=observation,
                            estimate=state.value,
                            scale=state.scale,
                            true_quantile=(
                                true_quantiles[
                                    probability
                                ]
                            ),
                        )

                    # -------------------------------------------------
                    # Now process x_t.
                    # -------------------------------------------------

                    state.update(
                        observation,
                        probability,
                        alpha,
                    )

        for probability in PROBABILITIES:

            for alpha in ALPHAS:

                metric = metrics[
                    (
                        probability,
                        alpha,
                    )
                ]

                results.append(
                    StationaryResult(
                        family=family.name,

                        probability=probability,
                        alpha=alpha,

                        true_quantile=(
                            true_quantiles[
                                probability
                            ]
                        ),

                        bias=metric.bias(),

                        mae=metric.mae(),

                        rmse=metric.rmse(),

                        coverage=(
                            metric.coverage()
                        ),

                        coverage_error=(
                            metric.coverage()
                            - probability
                        ),

                        mean_scale=(
                            metric.mean_scale()
                        ),
                    )
                )

    return results


# =============================================================================
# SHIFT RESULT
# =============================================================================

@dataclass
class ShiftResult:
    probability: float
    alpha: float

    initial_mean: float
    target_quantile: float

    steps_to_50_percent: int | None
    steps_to_90_percent: int | None

    late_bias: float
    late_mae: float

    late_coverage: float
    late_coverage_error: float


# =============================================================================
# THRESHOLD HELPER
# =============================================================================

def steps_to_fraction(
    trajectory: np.ndarray,
    target: float,
    fraction: float,
) -> int | None:

    initial = float(
        trajectory[0]
    )

    threshold = (
        initial
        + fraction
        * (
            target
            - initial
        )
    )

    if target >= initial:

        matches = np.flatnonzero(
            trajectory >= threshold
        )

    else:

        matches = np.flatnonzero(
            trajectory <= threshold
        )

    if len(
        matches
    ) == 0:
        return None

    return int(
        matches[0]
    )


# =============================================================================
# REGIME-SHIFT EXPERIMENT
#
# We warm the estimator on:
#
#       X ~ Normal(0,1)
#
# and then abruptly switch to:
#
#       X ~ Normal(+2,1)
#
# The complete population quantile therefore moves by exactly +2.
#
# We average the quantile trajectory across trials and measure the number
# of post-shift observations required to traverse 50% and 90% of the
# distance from its actual ensemble starting value to the new target.
# =============================================================================

def run_shift_experiment(
) -> list[ShiftResult]:

    rng = np.random.default_rng(
        SEED
        + 1000
    )

    keys = [
        (
            probability,
            alpha,
        )
        for probability
        in PROBABILITIES
        for alpha
        in ALPHAS
    ]

    trajectory_sums = {
        key: np.zeros(
            SHIFT_POST_LENGTH,
            dtype=float,
        )
        for key in keys
    }

    late_metrics = {
        key: Metrics()
        for key in keys
    }

    post_quantiles = {
        probability:
            normal_true_quantile(
                probability
            )
            + SHIFT_LOCATION
        for probability
        in PROBABILITIES
    }

    for _ in range(
        SHIFT_TRIALS
    ):

        states = {
            key:
                AdaptiveQuantileState()
            for key in keys
        }

        # ---------------------------------------------------------------------
        # PRE-SHIFT WARMUP
        # ---------------------------------------------------------------------

        pre_sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=SHIFT_PRE_LENGTH,
        )

        for raw_observation in pre_sample:

            observation = float(
                raw_observation
            )

            for (
                probability,
                alpha,
            ), state in states.items():

                state.update(
                    observation,
                    probability,
                    alpha,
                )

        # ---------------------------------------------------------------------
        # POST-SHIFT POPULATION
        # ---------------------------------------------------------------------

        post_sample = rng.normal(
            loc=SHIFT_LOCATION,
            scale=1.0,
            size=SHIFT_POST_LENGTH,
        )

        for index, raw_observation in enumerate(
            post_sample
        ):

            observation = float(
                raw_observation
            )

            for (
                probability,
                alpha,
            ), state in states.items():

                key = (
                    probability,
                    alpha,
                )

                # -------------------------------------------------------------
                # Record PRE-UPDATE q_t.
                # -------------------------------------------------------------

                trajectory_sums[
                    key
                ][index] += (
                    state.value
                )

                # -------------------------------------------------------------
                # Late post-shift steady-state metrics.
                # -------------------------------------------------------------

                if (
                    index
                    >= (
                        SHIFT_POST_LENGTH
                        - SHIFT_LATE_WINDOW
                    )
                ):

                    late_metrics[
                        key
                    ].add(
                        observation=observation,
                        estimate=state.value,
                        scale=state.scale,
                        true_quantile=(
                            post_quantiles[
                                probability
                            ]
                        ),
                    )

                # -------------------------------------------------------------
                # Now update with current observation.
                # -------------------------------------------------------------

                state.update(
                    observation,
                    probability,
                    alpha,
                )

    results: list[ShiftResult] = []

    for probability in PROBABILITIES:

        for alpha in ALPHAS:

            key = (
                probability,
                alpha,
            )

            trajectory = (
                trajectory_sums[
                    key
                ]
                / SHIFT_TRIALS
            )

            target = (
                post_quantiles[
                    probability
                ]
            )

            metric = (
                late_metrics[
                    key
                ]
            )

            results.append(
                ShiftResult(
                    probability=probability,
                    alpha=alpha,

                    initial_mean=float(
                        trajectory[0]
                    ),

                    target_quantile=target,

                    steps_to_50_percent=(
                        steps_to_fraction(
                            trajectory,
                            target,
                            0.50,
                        )
                    ),

                    steps_to_90_percent=(
                        steps_to_fraction(
                            trajectory,
                            target,
                            0.90,
                        )
                    ),

                    late_bias=(
                        metric.bias()
                    ),

                    late_mae=(
                        metric.mae()
                    ),

                    late_coverage=(
                        metric.coverage()
                    ),

                    late_coverage_error=(
                        metric.coverage()
                        - probability
                    ),
                )
            )

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_stationary_results(
    results: list[StationaryResult],
) -> None:

    for family in FAMILIES:

        print(
            "\n"
            "Adaptive Quantile Stationary Validation\n"
            f"Distribution: {family.name}\n"
            f"Trials: {STATIONARY_TRIALS}\n"
            f"Length: {STATIONARY_LENGTH}\n"
            f"Burn-in: {STATIONARY_BURN_IN}\n"
        )

        header = (
            f"{'p':>6}"
            f"{'alpha':>10}"
            f"{'Bias':>12}"
            f"{'MAE':>12}"
            f"{'RMSE':>12}"
            f"{'Coverage':>12}"
            f"{'Cov Err':>12}"
            f"{'Mean Scale':>14}"
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
                f"{result.alpha:10.3f}"
                f"{result.bias:12.6f}"
                f"{result.mae:12.6f}"
                f"{result.rmse:12.6f}"
                f"{result.coverage:12.6f}"
                f"{result.coverage_error:12.6f}"
                f"{result.mean_scale:14.6f}"
            )


def format_steps(
    value: int | None,
) -> str:

    if value is None:
        return (
            f">{SHIFT_POST_LENGTH}"
        )

    return str(
        value
    )


def print_shift_results(
    results: list[ShiftResult],
) -> None:

    print(
        "\n"
        "Adaptive Quantile Regime-Shift Validation\n"
        "Distribution transition: Normal(0,1) -> Normal(+2,1)\n"
        f"Trials: {SHIFT_TRIALS}\n"
        f"Pre-shift observations: {SHIFT_PRE_LENGTH}\n"
        f"Post-shift observations: {SHIFT_POST_LENGTH}\n"
        f"Late window: {SHIFT_LATE_WINDOW}\n"
    )

    header = (
        f"{'p':>6}"
        f"{'alpha':>10}"
        f"{'Initial':>12}"
        f"{'Target':>12}"
        f"{'T50':>10}"
        f"{'T90':>10}"
        f"{'Late Bias':>12}"
        f"{'Late MAE':>12}"
        f"{'Coverage':>12}"
        f"{'Cov Err':>12}"
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

        print(
            f"{result.probability:6.2f}"
            f"{result.alpha:10.3f}"
            f"{result.initial_mean:12.6f}"
            f"{result.target_quantile:12.6f}"
            f"{format_steps(result.steps_to_50_percent):>10}"
            f"{format_steps(result.steps_to_90_percent):>10}"
            f"{result.late_bias:12.6f}"
            f"{result.late_mae:12.6f}"
            f"{result.late_coverage:12.6f}"
            f"{result.late_coverage_error:12.6f}"
        )


# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_stationary_csv(
    results: list[StationaryResult],
) -> None:

    STATIONARY_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATIONARY_RESULTS_PATH.open(
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
                "alpha",
                "true_quantile",
                "bias",
                "mae",
                "rmse",
                "coverage",
                "coverage_error",
                "mean_scale",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.family,
                    result.probability,
                    result.alpha,
                    result.true_quantile,
                    result.bias,
                    result.mae,
                    result.rmse,
                    result.coverage,
                    result.coverage_error,
                    result.mean_scale,
                ]
            )


def save_shift_csv(
    results: list[ShiftResult],
) -> None:

    SHIFT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SHIFT_RESULTS_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "probability",
                "alpha",
                "initial_mean",
                "target_quantile",
                "steps_to_50_percent",
                "steps_to_90_percent",
                "late_bias",
                "late_mae",
                "late_coverage",
                "late_coverage_error",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.probability,
                    result.alpha,
                    result.initial_mean,
                    result.target_quantile,
                    result.steps_to_50_percent,
                    result.steps_to_90_percent,
                    result.late_bias,
                    result.late_mae,
                    result.late_coverage,
                    result.late_coverage_error,
                ]
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    stationary_results = (
        run_stationary_experiment()
    )

    shift_results = (
        run_shift_experiment()
    )

    print_stationary_results(
        stationary_results
    )

    print_shift_results(
        shift_results
    )

    save_stationary_csv(
        stationary_results
    )

    save_shift_csv(
        shift_results
    )

    print(
        "\n"
        f"Saved stationary results to: "
        f"{STATIONARY_RESULTS_PATH}\n"
        f"Saved shift results to: "
        f"{SHIFT_RESULTS_PATH}\n"
    )


if __name__ == "__main__":
    main()
