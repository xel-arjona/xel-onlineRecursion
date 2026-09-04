from __future__ import annotations

import csv
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t as student_t

from src.adaptive_expectile import (
    AdaptiveExpectileState,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

LEVELS = (
    0.01,
    0.99,
)

ALPHAS = (
    0.010,
    0.020,
    0.050,
    0.100,
)

STATIONARY_TRIALS = 50
STATIONARY_LENGTH = 10000
STATIONARY_BURN_IN = 6000

SHIFT_TRIALS = 50
SHIFT_PRE_LENGTH = 8000
SHIFT_POST_LENGTH = 12000

SHIFT_MAGNITUDES = (
    2.0,
    -2.0,
)

SHIFT_LATE_WINDOW = 1000

SEED = 1729

STATIONARY_PATH = Path(
    "results/adaptive_expectile_extreme_stationary.csv"
)

SHIFT_PATH = Path(
    "results/adaptive_expectile_extreme_shift.csv"
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

    pdf: Callable[
        [float],
        float,
    ]


# =============================================================================
# NORMAL
# =============================================================================

def generate_normal(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return rng.normal(
        0.0,
        1.0,
        sample_size,
    )


def normal_pdf(
    value: float,
) -> float:

    return float(
        norm.pdf(
            value
        )
    )


# =============================================================================
# STUDENT-t
# =============================================================================

def student_scale(
    df: float,
) -> float:

    return float(
        np.sqrt(
            df
            / (
                df - 2.0
            )
        )
    )


T5_SD = student_scale(
    5.0
)

T3_SD = student_scale(
    3.0
)


def generate_t5(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return (
        rng.standard_t(
            5,
            sample_size,
        )
        / T5_SD
    )


def t5_pdf(
    value: float,
) -> float:

    return float(
        T5_SD
        * student_t.pdf(
            T5_SD
            * value,
            df=5,
        )
    )


def generate_t3(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    return (
        rng.standard_t(
            3,
            sample_size,
        )
        / T3_SD
    )


def t3_pdf(
    value: float,
) -> float:

    return float(
        T3_SD
        * student_t.pdf(
            T3_SD
            * value,
            df=3,
        )
    )


# =============================================================================
# CONTAMINATED GAUSSIAN
#
# Raw population:
#
#     99% N(0,1)
#      1% N(0,10)
#
# Raw variance:
#
#     .99 * 1
#     + .01 * 100
#     = 1.99
#
# Standardize to unit variance.
# =============================================================================

CONTAMINATION_SD = float(
    np.sqrt(
        1.99
    )
)


def generate_contaminated_normal(
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:

    contaminated = (
        rng.random(
            sample_size
        )
        < 0.01
    )

    sample = rng.normal(
        0.0,
        1.0,
        sample_size,
    )

    count = int(
        np.sum(
            contaminated
        )
    )

    if count > 0:

        sample[
            contaminated
        ] = rng.normal(
            0.0,
            10.0,
            count,
        )

    return (
        sample
        / CONTAMINATION_SD
    )


def contaminated_normal_pdf(
    value: float,
) -> float:

    raw_value = (
        CONTAMINATION_SD
        * value
    )

    raw_pdf = (
        0.99
        * norm.pdf(
            raw_value,
            loc=0.0,
            scale=1.0,
        )
        + 0.01
        * norm.pdf(
            raw_value,
            loc=0.0,
            scale=10.0,
        )
    )

    return float(
        CONTAMINATION_SD
        * raw_pdf
    )


FAMILIES = (
    Family(
        name="Normal(0,1)",
        seed_offset=100,
        generator=generate_normal,
        pdf=normal_pdf,
    ),

    Family(
        name="Student-t(df=5), unit variance",
        seed_offset=200,
        generator=generate_t5,
        pdf=t5_pdf,
    ),

    Family(
        name="Student-t(df=3), unit variance",
        seed_offset=300,
        generator=generate_t3,
        pdf=t3_pdf,
    ),

    Family(
        name="Contaminated Gaussian, unit variance",
        seed_offset=400,
        generator=generate_contaminated_normal,
        pdf=contaminated_normal_pdf,
    ),
)


# =============================================================================
# INDEPENDENT EXPECTILE SOLVER
# =============================================================================

def expectile_moment(
    estimate: float,
    level: float,
    pdf: Callable[
        [float],
        float,
    ],
) -> float:

    lower, _ = quad(
        lambda value:
            (
                1.0
                - level
            )
            * (
                value
                - estimate
            )
            * pdf(
                value
            ),
        -inf,
        estimate,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=400,
    )

    upper, _ = quad(
        lambda value:
            level
            * (
                value
                - estimate
            )
            * pdf(
                value
            ),
        estimate,
        inf,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=400,
    )

    return float(
        lower
        + upper
    )


def solve_expectile(
    level: float,
    pdf: Callable[
        [float],
        float,
    ],
) -> float:

    lower = -1.0
    upper = 1.0

    while (
        expectile_moment(
            lower,
            level,
            pdf,
        )
        <= 0.0
    ):
        lower *= 2.0

    while (
        expectile_moment(
            upper,
            level,
            pdf,
        )
        >= 0.0
    ):
        upper *= 2.0

    return float(
        brentq(
            lambda estimate:
                expectile_moment(
                    estimate,
                    level,
                    pdf,
                ),
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
        )
    )


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class Metrics:
    count: int = 0

    error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    moment_sum: float = 0.0
    absolute_residual_sum: float = 0.0

    def add(
        self,
        observation: float,
        estimate: float,
        target: float,
        level: float,
    ) -> None:

        error = (
            estimate
            - target
        )

        self.error_sum += error

        self.absolute_error_sum += abs(
            error
        )

        self.squared_error_sum += (
            error
            * error
        )

        residual = (
            observation
            - estimate
        )

        weighted_residual = (
            level
            * residual
            if residual > 0.0
            else (
                1.0
                - level
            )
            * residual
            if residual < 0.0
            else 0.0
        )

        self.moment_sum += (
            weighted_residual
        )

        self.absolute_residual_sum += abs(
            residual
        )

        self.count += 1

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

    def moment(
        self,
    ) -> float:

        return (
            self.moment_sum
            / self.count
        )

    def normalized_balance(
        self,
    ) -> float:

        if (
            self.absolute_residual_sum
            == 0.0
        ):
            return 0.0

        return (
            self.moment_sum
            / self.absolute_residual_sum
        )


# =============================================================================
# STATIONARY
# =============================================================================

@dataclass
class StationaryResult:
    family: str

    level: float
    alpha: float

    target: float

    bias: float
    mae: float
    rmse: float

    moment: float
    normalized_balance: float


def run_stationary(
) -> list[StationaryResult]:

    results: list[StationaryResult] = []

    for family in FAMILIES:

        targets = {
            level:
                solve_expectile(
                    level,
                    family.pdf,
                )
            for level
            in LEVELS
        }

        keys = [
            (
                level,
                alpha,
            )
            for level
            in LEVELS
            for alpha
            in ALPHAS
        ]

        aggregate = {
            key:
                Metrics()
            for key
            in keys
        }

        rng = np.random.default_rng(
            SEED
            + family.seed_offset
        )

        for _ in range(
            STATIONARY_TRIALS
        ):

            states = {
                key:
                    AdaptiveExpectileState()
                for key
                in keys
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
                    level,
                    alpha,
                ), state in states.items():

                    if (
                        index
                        >= STATIONARY_BURN_IN
                    ):

                        aggregate[
                            (
                                level,
                                alpha,
                            )
                        ].add(
                            observation,
                            state.value,
                            targets[
                                level
                            ],
                            level,
                        )

                    state.update(
                        observation,
                        level,
                        alpha,
                    )

        for level in LEVELS:

            for alpha in ALPHAS:

                metric = aggregate[
                    (
                        level,
                        alpha,
                    )
                ]

                results.append(
                    StationaryResult(
                        family=family.name,

                        level=level,
                        alpha=alpha,

                        target=targets[
                            level
                        ],

                        bias=metric.bias(),
                        mae=metric.mae(),
                        rmse=metric.rmse(),

                        moment=metric.moment(),

                        normalized_balance=(
                            metric.normalized_balance()
                        ),
                    )
                )

    return results


# =============================================================================
# SHIFT
# =============================================================================

@dataclass
class ShiftResult:
    shift: float

    level: float
    alpha: float

    initial: float
    target: float

    t50: int | None
    t90: int | None

    late_bias: float
    late_mae: float

    late_moment: float
    late_normalized_balance: float


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


def run_shift(
) -> list[ShiftResult]:

    normal_targets = {
        level:
            solve_expectile(
                level,
                normal_pdf,
            )
        for level
        in LEVELS
    }

    keys = [
        (
            level,
            alpha,
        )
        for level
        in LEVELS
        for alpha
        in ALPHAS
    ]

    results: list[ShiftResult] = []

    for shift_index, shift in enumerate(
        SHIFT_MAGNITUDES
    ):

        trajectory_sums = {
            key:
                np.zeros(
                    SHIFT_POST_LENGTH,
                    dtype=float,
                )
            for key
            in keys
        }

        late_metrics = {
            key:
                Metrics()
            for key
            in keys
        }

        rng = np.random.default_rng(
            SEED
            + 10000
            + shift_index
        )

        for _ in range(
            SHIFT_TRIALS
        ):

            states = {
                key:
                    AdaptiveExpectileState()
                for key
                in keys
            }

            pre_sample = rng.normal(
                0.0,
                1.0,
                SHIFT_PRE_LENGTH,
            )

            for raw_observation in pre_sample:

                observation = float(
                    raw_observation
                )

                for (
                    level,
                    alpha,
                ), state in states.items():

                    state.update(
                        observation,
                        level,
                        alpha,
                    )

            post_sample = rng.normal(
                shift,
                1.0,
                SHIFT_POST_LENGTH,
            )

            for index, raw_observation in enumerate(
                post_sample
            ):

                observation = float(
                    raw_observation
                )

                for (
                    level,
                    alpha,
                ), state in states.items():

                    key = (
                        level,
                        alpha,
                    )

                    trajectory_sums[
                        key
                    ][index] += (
                        state.value
                    )

                    if (
                        index
                        >= SHIFT_POST_LENGTH
                        - SHIFT_LATE_WINDOW
                    ):

                        target = (
                            normal_targets[
                                level
                            ]
                            + shift
                        )

                        late_metrics[
                            key
                        ].add(
                            observation,
                            state.value,
                            target,
                            level,
                        )

                    state.update(
                        observation,
                        level,
                        alpha,
                    )

        for level in LEVELS:

            target = (
                normal_targets[
                    level
                ]
                + shift
            )

            for alpha in ALPHAS:

                key = (
                    level,
                    alpha,
                )

                trajectory = (
                    trajectory_sums[
                        key
                    ]
                    / SHIFT_TRIALS
                )

                metric = late_metrics[
                    key
                ]

                results.append(
                    ShiftResult(
                        shift=shift,

                        level=level,
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

                        late_bias=metric.bias(),
                        late_mae=metric.mae(),

                        late_moment=metric.moment(),

                        late_normalized_balance=(
                            metric.normalized_balance()
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

    for family in FAMILIES:

        print(
            "\n"
            "Adaptive Expectile Extreme-Level Stress Validation\n"
            f"Distribution: {family.name}\n"
            f"Trials: {STATIONARY_TRIALS}\n"
            f"Length: {STATIONARY_LENGTH}\n"
            f"Burn-in: {STATIONARY_BURN_IN}\n"
        )

        header = (
            f"{'Level':>7}"
            f"{'Alpha':>9}"
            f"{'Target':>12}"
            f"{'Bias':>12}"
            f"{'MAE':>12}"
            f"{'RMSE':>12}"
            f"{'Moment':>12}"
            f"{'Norm Bal':>12}"
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
                f"{result.level:7.2f}"
                f"{result.alpha:9.3f}"
                f"{result.target:12.6f}"
                f"{result.bias:12.6f}"
                f"{result.mae:12.6f}"
                f"{result.rmse:12.6f}"
                f"{result.moment:12.6f}"
                f"{result.normalized_balance:12.6f}"
            )


def print_shift(
    results: list[ShiftResult],
) -> None:

    for shift in SHIFT_MAGNITUDES:

        print(
            "\n"
            "Adaptive Expectile Extreme-Level Shift Validation\n"
            f"Normal(0,1) -> Normal({shift:+.1f},1)\n"
            f"Trials: {SHIFT_TRIALS}\n"
            f"Warmup: {SHIFT_PRE_LENGTH}\n"
            f"Post-shift: {SHIFT_POST_LENGTH}\n"
        )

        header = (
            f"{'Level':>7}"
            f"{'Alpha':>9}"
            f"{'Initial':>12}"
            f"{'Target':>12}"
            f"{'T50':>10}"
            f"{'T90':>10}"
            f"{'Late Bias':>12}"
            f"{'Late MAE':>12}"
            f"{'Moment':>12}"
            f"{'Norm Bal':>12}"
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
                f"{result.level:7.2f}"
                f"{result.alpha:9.3f}"
                f"{result.initial:12.6f}"
                f"{result.target:12.6f}"
                f"{format_steps(result.t50):>10}"
                f"{format_steps(result.t90):>10}"
                f"{result.late_bias:12.6f}"
                f"{result.late_mae:12.6f}"
                f"{result.late_moment:12.6f}"
                f"{result.late_normalized_balance:12.6f}"
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
                "family",
                "level",
                "alpha",
                "target",
                "bias",
                "mae",
                "rmse",
                "moment",
                "normalized_balance",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.family,
                    result.level,
                    result.alpha,
                    result.target,
                    result.bias,
                    result.mae,
                    result.rmse,
                    result.moment,
                    result.normalized_balance,
                ]
            )


def save_shift(
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
                "level",
                "alpha",
                "initial",
                "target",
                "t50",
                "t90",
                "late_bias",
                "late_mae",
                "late_moment",
                "late_normalized_balance",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.shift,
                    result.level,
                    result.alpha,
                    result.initial,
                    result.target,
                    result.t50,
                    result.t90,
                    result.late_bias,
                    result.late_mae,
                    result.late_moment,
                    result.late_normalized_balance,
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
        run_shift()
    )

    print_stationary(
        stationary_results
    )

    print_shift(
        shift_results
    )

    save_stationary(
        stationary_results
    )

    save_shift(
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
