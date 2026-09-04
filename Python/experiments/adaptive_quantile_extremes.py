from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.adaptive_quantile import (
    AdaptiveQuantileState,
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

STATIONARY_ALPHAS = (
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
)

SHIFT_ALPHAS = (
    0.010,
    0.020,
    0.050,
    0.100,
)

STATIONARY_TRIALS = 20
STATIONARY_LENGTH = 12000
STATIONARY_BURN_IN = 8000

SHIFT_TRIALS = 60
SHIFT_PRE_LENGTH = 10000
SHIFT_POST_LENGTH = 1500

SHIFT_MAGNITUDES = (
    2.0,
    -2.0,
)

SEED = 1729

STATIONARY_PATH = Path(
    "results/adaptive_quantile_extreme_stationary.csv"
)

SHIFT_PATH = Path(
    "results/adaptive_quantile_bidirectional_shift.csv"
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class StationaryResult:
    probability: float
    alpha: float

    bias: float
    mae: float

    coverage: float
    coverage_error: float

    mean_scale: float
    theoretical_scale: float
    scale_error: float


@dataclass
class ShiftResult:
    shift: float

    probability: float
    alpha: float

    initial: float
    target: float

    t50: int | None
    t90: int | None


# =============================================================================
# NORMAL POPULATION HELPERS
# =============================================================================

def true_quantile(
    probability: float,
    location: float = 0.0,
) -> float:

    return float(
        location
        + norm.ppf(
            probability
        )
    )


def theoretical_absolute_innovation_scale(
    probability: float,
) -> float:
    """
    For X ~ Normal(0,1) and q = Phi^-1(p):

        E|X - q|
            =
            2 phi(q)
            + q (2 Phi(q) - 1)
    """

    q = float(
        norm.ppf(
            probability
        )
    )

    return float(
        2.0
        * norm.pdf(
            q
        )
        + q
        * (
            2.0
            * probability
            - 1.0
        )
    )


# =============================================================================
# STATIONARY EXPERIMENT
# =============================================================================

def run_stationary(
) -> list[StationaryResult]:

    rng = np.random.default_rng(
        SEED + 100
    )

    keys = [
        (
            probability,
            alpha,
        )
        for probability
        in PROBABILITIES
        for alpha
        in STATIONARY_ALPHAS
    ]

    error_sum = {
        key: 0.0
        for key in keys
    }

    absolute_error_sum = {
        key: 0.0
        for key in keys
    }

    coverage_count = {
        key: 0
        for key in keys
    }

    scale_sum = {
        key: 0.0
        for key in keys
    }

    count = {
        key: 0
        for key in keys
    }

    targets = {
        probability:
            true_quantile(
                probability
            )
        for probability
        in PROBABILITIES
    }

    for _ in range(
        STATIONARY_TRIALS
    ):

        states = {
            key:
                AdaptiveQuantileState()
            for key in keys
        }

        sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=STATIONARY_LENGTH,
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

                key = (
                    probability,
                    alpha,
                )

                if (
                    index
                    >= STATIONARY_BURN_IN
                ):

                    error = (
                        state.value
                        - targets[
                            probability
                        ]
                    )

                    error_sum[
                        key
                    ] += error

                    absolute_error_sum[
                        key
                    ] += abs(
                        error
                    )

                    if (
                        observation
                        <= state.value
                    ):
                        coverage_count[
                            key
                        ] += 1

                    scale_sum[
                        key
                    ] += state.scale

                    count[
                        key
                    ] += 1

                state.update(
                    observation,
                    probability,
                    alpha,
                )

    results: list[StationaryResult] = []

    for probability in PROBABILITIES:

        theoretical_scale = (
            theoretical_absolute_innovation_scale(
                probability
            )
        )

        for alpha in STATIONARY_ALPHAS:

            key = (
                probability,
                alpha,
            )

            n = count[
                key
            ]

            bias = (
                error_sum[
                    key
                ]
                / n
            )

            mae = (
                absolute_error_sum[
                    key
                ]
                / n
            )

            coverage = (
                coverage_count[
                    key
                ]
                / n
            )

            mean_scale = (
                scale_sum[
                    key
                ]
                / n
            )

            results.append(
                StationaryResult(
                    probability=probability,
                    alpha=alpha,

                    bias=bias,
                    mae=mae,

                    coverage=coverage,

                    coverage_error=(
                        coverage
                        - probability
                    ),

                    mean_scale=mean_scale,

                    theoretical_scale=(
                        theoretical_scale
                    ),

                    scale_error=(
                        mean_scale
                        - theoretical_scale
                    ),
                )
            )

    return results


# =============================================================================
# SHIFT HELPERS
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
            trajectory
            >= threshold
        )

    else:

        matches = np.flatnonzero(
            trajectory
            <= threshold
        )

    if len(
        matches
    ) == 0:
        return None

    return int(
        matches[0]
    )


# =============================================================================
# BIDIRECTIONAL REGIME SHIFT
# =============================================================================

def run_shifts(
) -> list[ShiftResult]:

    results: list[ShiftResult] = []

    keys = [
        (
            probability,
            alpha,
        )
        for probability
        in PROBABILITIES
        for alpha
        in SHIFT_ALPHAS
    ]

    for shift_index, shift in enumerate(
        SHIFT_MAGNITUDES
    ):

        rng = np.random.default_rng(
            SEED
            + 1000
            + shift_index
        )

        trajectory_sums = {
            key: np.zeros(
                SHIFT_POST_LENGTH,
                dtype=float,
            )
            for key in keys
        }

        for _ in range(
            SHIFT_TRIALS
        ):

            states = {
                key:
                    AdaptiveQuantileState()
                for key in keys
            }

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

            post_sample = rng.normal(
                loc=shift,
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

                    trajectory_sums[
                        key
                    ][index] += (
                        state.value
                    )

                    state.update(
                        observation,
                        probability,
                        alpha,
                    )

        for probability in PROBABILITIES:

            target = true_quantile(
                probability,
                location=shift,
            )

            for alpha in SHIFT_ALPHAS:

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

                results.append(
                    ShiftResult(
                        shift=shift,

                        probability=probability,
                        alpha=alpha,

                        initial=float(
                            trajectory[0]
                        ),

                        target=target,

                        t50=steps_to_fraction(
                            trajectory,
                            target,
                            0.50,
                        ),

                        t90=steps_to_fraction(
                            trajectory,
                            target,
                            0.90,
                        ),
                    )
                )

    return results


# =============================================================================
# REPORTING
# =============================================================================

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


def print_stationary(
    results: list[StationaryResult],
) -> None:

    print(
        "\n"
        "Adaptive Quantile Extreme-Tail Stationary Validation\n"
        "Distribution: Normal(0,1)\n"
        f"Trials: {STATIONARY_TRIALS}\n"
        f"Length: {STATIONARY_LENGTH}\n"
        f"Burn-in: {STATIONARY_BURN_IN}\n"
    )

    header = (
        f"{'p':>6}"
        f"{'alpha':>10}"
        f"{'Bias':>12}"
        f"{'MAE':>12}"
        f"{'Coverage':>12}"
        f"{'Cov Err':>12}"
        f"{'Scale':>12}"
        f"{'Theory':>12}"
        f"{'Scale Err':>12}"
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
            f"{result.bias:12.6f}"
            f"{result.mae:12.6f}"
            f"{result.coverage:12.6f}"
            f"{result.coverage_error:12.6f}"
            f"{result.mean_scale:12.6f}"
            f"{result.theoretical_scale:12.6f}"
            f"{result.scale_error:12.6f}"
        )


def print_shifts(
    results: list[ShiftResult],
) -> None:

    for shift in SHIFT_MAGNITUDES:

        print(
            "\n"
            "Adaptive Quantile Bidirectional Shift Validation\n"
            f"Normal(0,1) -> Normal({shift:+.1f},1)\n"
            f"Trials: {SHIFT_TRIALS}\n"
            f"Warmup: {SHIFT_PRE_LENGTH}\n"
        )

        header = (
            f"{'p':>6}"
            f"{'alpha':>10}"
            f"{'Initial':>12}"
            f"{'Target':>12}"
            f"{'T50':>10}"
            f"{'T90':>10}"
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

            if result.shift != shift:
                continue

            print(
                f"{result.probability:6.2f}"
                f"{result.alpha:10.3f}"
                f"{result.initial:12.6f}"
                f"{result.target:12.6f}"
                f"{format_steps(result.t50):>10}"
                f"{format_steps(result.t90):>10}"
            )


# =============================================================================
# CSV
# =============================================================================

def save_stationary(
    results: list[StationaryResult],
) -> None:

    STATIONARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATIONARY_PATH.open(
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
                "bias",
                "mae",
                "coverage",
                "coverage_error",
                "mean_scale",
                "theoretical_scale",
                "scale_error",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.probability,
                    result.alpha,
                    result.bias,
                    result.mae,
                    result.coverage,
                    result.coverage_error,
                    result.mean_scale,
                    result.theoretical_scale,
                    result.scale_error,
                ]
            )


def save_shifts(
    results: list[ShiftResult],
) -> None:

    SHIFT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SHIFT_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "shift",
                "probability",
                "alpha",
                "initial",
                "target",
                "t50",
                "t90",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.shift,
                    result.probability,
                    result.alpha,
                    result.initial,
                    result.target,
                    result.t50,
                    result.t90,
                ]
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    stationary_results = (
        run_stationary()
    )

    shift_results = (
        run_shifts()
    )

    print_stationary(
        stationary_results
    )

    print_shifts(
        shift_results
    )

    save_stationary(
        stationary_results
    )

    save_shifts(
        shift_results
    )

    print(
        "\n"
        f"Saved stationary results to: "
        f"{STATIONARY_PATH}\n"
        f"Saved shift results to: "
        f"{SHIFT_PATH}\n"
    )


if __name__ == "__main__":
    main()
