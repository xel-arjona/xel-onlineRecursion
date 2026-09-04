"""
Exponentially Weighted Conditional Tail Mean validation.

Candidate O(1) adaptive tail-mean estimator:

    indicator =
        1 if observation belongs to the selected tail
        0 otherwise

    tail_mass += alpha * (indicator - tail_mass)

    weighted_tail += alpha * (
        indicator * observation
        - weighted_tail
    )

    tail_mean = weighted_tail / tail_mass

The estimator uses ordinary observation-clock exponential forgetting.

For a fixed threshold q:

    E[indicator * X] / E[indicator]

equals the conditional tail mean.

When q is the corresponding quantile of a continuous distribution, this is
Expected Shortfall / tail mean under the selected sign convention.

The adaptive coupled implementation always uses the PREVIOUS quantile state as
the current observation's threshold.

This experiment evaluates:

1. stationary bias / MAE / RMSE;
2. retained tail-mass calibration;
3. shift T50/T90;
4. ensemble-mean overshoot;
5. settling behavior.

Only lower-tail Monte Carlo cases are required after an explicit deterministic
reflection-symmetry check.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    exp,
    isfinite,
    isnan,
    pi,
    sqrt,
)
from statistics import NormalDist

import numpy as np


STATIONARY_TRIALS = 50
STATIONARY_LENGTH = 12_000
STATIONARY_BURN_IN = 7_000

SHIFT_TRIALS = 100
SHIFT_WARMUP = 10_000
SHIFT_LENGTH = 14_000

SETTLING_FRACTION = 0.10
SETTLING_RUN = 500

MIN_FLOAT = 1e-16

BASE_SEED = 20260906

NORMAL = NormalDist()


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:

    return max(
        lower,
        min(
            upper,
            value,
        ),
    )


def normal_pdf(
    value: float,
) -> float:

    return (
        exp(
            -0.5
            * value
            * value
        )
        / sqrt(
            2.0
            * pi
        )
    )


def lower_normal_tail_mean(
    probability: float,
) -> float:

    quantile = NORMAL.inv_cdf(
        probability
    )

    return (
        -normal_pdf(
            quantile
        )
        / probability
    )


def upper_normal_tail_mean(
    probability: float,
) -> float:

    quantile = NORMAL.inv_cdf(
        probability
    )

    return (
        normal_pdf(
            quantile
        )
        / (
            1.0
            - probability
        )
    )


@dataclass
class AdaptiveQuantileState:

    value: float = float(
        "nan"
    )

    probability: float = float(
        "nan"
    )

    scale: float = float(
        "nan"
    )

    def update(
        self,
        input_value: float,
        probability: float,
        alpha: float,
    ) -> "AdaptiveQuantileState":

        if (
            probability <= 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "Quantile probability must be strictly between zero and one."
            )

        if (
            not isnan(
                self.probability
            )
            and probability
            != self.probability
        ):
            raise ValueError(
                "Quantile probability cannot change inside an active state."
            )

        if (
            not isfinite(
                input_value
            )
            or not isfinite(
                alpha
            )
        ):
            return self

        coefficient = clamp(
            alpha,
            0.0,
            1.0,
        )

        if isnan(
            self.value
        ):
            self.value = input_value
            self.probability = probability
            self.scale = 0.0

            return self

        innovation = (
            input_value
            - self.value
        )

        previous_scale = (
            self.scale
        )

        if innovation > 0.0:
            score = probability

        elif innovation < 0.0:
            score = (
                probability
                - 1.0
            )

        else:
            score = 0.0

        self.value += (
            coefficient
            * previous_scale
            * score
        )

        self.scale = (
            previous_scale
            + coefficient
            * (
                abs(
                    innovation
                )
                - previous_scale
            )
        )

        return self


@dataclass
class ConditionalTailMeanState:

    value: float = float(
        "nan"
    )

    weighted_tail: float = 0.0
    tail_mass: float = 0.0

    side: str | None = None

    def reset(
        self,
    ) -> "ConditionalTailMeanState":

        self.value = float(
            "nan"
        )

        self.weighted_tail = 0.0
        self.tail_mass = 0.0
        self.side = None

        return self

    def update(
        self,
        input_value: float,
        threshold: float,
        alpha: float,
        side: str,
    ) -> "ConditionalTailMeanState":

        if side not in (
            "lower",
            "upper",
        ):
            raise ValueError(
                "Tail side must be 'lower' or 'upper'."
            )

        if (
            self.side is not None
            and side
            != self.side
        ):
            raise ValueError(
                "Tail side cannot change inside an active state."
            )

        if (
            not isfinite(
                input_value
            )
            or not isfinite(
                threshold
            )
            or not isfinite(
                alpha
            )
        ):
            return self

        coefficient = clamp(
            alpha,
            0.0,
            1.0,
        )

        if side == "lower":

            indicator = (
                1.0
                if input_value <= threshold
                else 0.0
            )

        else:

            indicator = (
                1.0
                if input_value >= threshold
                else 0.0
            )

        retention = (
            1.0
            - coefficient
        )

        self.weighted_tail = (
            retention
            * self.weighted_tail
            + coefficient
            * indicator
            * input_value
        )

        self.tail_mass = (
            retention
            * self.tail_mass
            + coefficient
            * indicator
        )

        if self.tail_mass > MIN_FLOAT:

            self.value = (
                self.weighted_tail
                / self.tail_mass
            )

            if self.side is None:
                self.side = side

        return self


@dataclass
class CoupledState:

    quantile: AdaptiveQuantileState
    tail: ConditionalTailMeanState

    @classmethod
    def new(
        cls,
    ) -> "CoupledState":

        return cls(
            quantile=AdaptiveQuantileState(),
            tail=ConditionalTailMeanState(),
        )

    def update(
        self,
        input_value: float,
        probability: float,
        quantile_alpha: float,
        tail_alpha: float,
        side: str,
    ) -> "CoupledState":

        if isnan(
            self.quantile.value
        ):

            self.quantile.update(
                input_value,
                probability,
                quantile_alpha,
            )

            self.tail.update(
                input_value,
                self.quantile.value,
                tail_alpha,
                side,
            )

            return self

        previous_quantile = (
            self.quantile.value
        )

        self.tail.update(
            input_value,
            previous_quantile,
            tail_alpha,
            side,
        )

        self.quantile.update(
            input_value,
            probability,
            quantile_alpha,
        )

        return self


@dataclass
class Metrics:

    count: int = 0

    error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    def add(
        self,
        value: float,
        target: float,
    ) -> None:

        error = (
            value
            - target
        )

        self.count += 1

        self.error_sum += error

        self.absolute_error_sum += abs(
            error
        )

        self.squared_error_sum += (
            error
            * error
        )

    @property
    def bias(
        self,
    ) -> float:

        return (
            self.error_sum
            / self.count
        )

    @property
    def mae(
        self,
    ) -> float:

        return (
            self.absolute_error_sum
            / self.count
        )

    @property
    def rmse(
        self,
    ) -> float:

        return sqrt(
            self.squared_error_sum
            / self.count
        )


def tail_targets(
    probability: float,
    side: str,
    location: float = 0.0,
) -> tuple[
    float,
    float,
]:

    quantile = (
        location
        + NORMAL.inv_cdf(
            probability
        )
    )

    if side == "lower":

        tail = (
            location
            + lower_normal_tail_mean(
                probability
            )
        )

    else:

        tail = (
            location
            + upper_normal_tail_mean(
                probability
            )
        )

    return (
        quantile,
        tail,
    )


def assert_reflection_symmetry() -> None:

    sample = [
        -3.0,
        2.0,
        7.0,
        -1.0,
        4.0,
        -8.0,
        6.0,
        0.5,
        9.0,
        -2.5,
    ]

    lower = (
        CoupledState.new()
    )

    mirror = (
        CoupledState.new()
    )

    for observation in sample:

        lower.update(
            observation,
            0.05,
            0.20,
            0.01,
            "lower",
        )

        mirror.update(
            -observation,
            0.95,
            0.20,
            0.01,
            "upper",
        )

        if not np.isclose(
            lower.quantile.value,
            -mirror.quantile.value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Quantile reflection symmetry failed."
            )

        if not np.isclose(
            lower.tail.value,
            -mirror.tail.value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Conditional tail-mean reflection symmetry failed."
            )

        if not np.isclose(
            lower.tail.tail_mass,
            mirror.tail.tail_mass,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Conditional tail-mass reflection symmetry failed."
            )


def stationary_validation(
    probability: float,
    quantile_alpha: float,
    tail_alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    (
        quantile_target,
        tail_target,
    ) = tail_targets(
        probability,
        "lower",
    )

    q_metrics = Metrics()
    tail_metrics = Metrics()

    mass_sum = 0.0
    mass_count = 0

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        STATIONARY_TRIALS
    ):

        state = (
            CoupledState.new()
        )

        sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=STATIONARY_LENGTH,
        )

        for index, raw_observation in enumerate(
            sample
        ):

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
                "lower",
            )

            if (
                index >= STATIONARY_BURN_IN
                and not isnan(
                    state.tail.value
                )
            ):

                q_metrics.add(
                    state.quantile.value,
                    quantile_target,
                )

                tail_metrics.add(
                    state.tail.value,
                    tail_target,
                )

                mass_sum += (
                    state.tail.tail_mass
                )

                mass_count += 1

    return {
        "probability": probability,
        "quantile_alpha": quantile_alpha,
        "tail_alpha": tail_alpha,

        "q_bias": q_metrics.bias,
        "q_rmse": q_metrics.rmse,

        "tail_bias": tail_metrics.bias,
        "tail_mae": tail_metrics.mae,
        "tail_rmse": tail_metrics.rmse,

        "mean_mass": (
            mass_sum
            / mass_count
        ),
    }


def crossing_time(
    path: np.ndarray,
    initial_target: float,
    final_target: float,
    fraction: float,
) -> float:

    target = (
        initial_target
        + fraction
        * (
            final_target
            - initial_target
        )
    )

    if final_target > initial_target:

        indices = np.flatnonzero(
            path >= target
        )

    else:

        indices = np.flatnonzero(
            path <= target
        )

    if indices.size == 0:

        return float(
            "nan"
        )

    return float(
        indices[0]
        + 1
    )


def overshoot(
    path: np.ndarray,
    final_target: float,
    shift: float,
) -> float:

    if shift > 0.0:

        excursion = (
            np.max(
                path
            )
            - final_target
        )

    else:

        excursion = (
            final_target
            - np.min(
                path
            )
        )

    return max(
        float(
            excursion
        ),
        0.0,
    )


def settling_time(
    path: np.ndarray,
    final_target: float,
    shift: float,
) -> float:

    tolerance = (
        SETTLING_FRACTION
        * abs(
            shift
        )
    )

    inside = (
        np.abs(
            path
            - final_target
        )
        <= tolerance
    )

    rolling_count = np.convolve(
        inside.astype(
            int
        ),
        np.ones(
            SETTLING_RUN,
            dtype=int,
        ),
        mode="valid",
    )

    indices = np.flatnonzero(
        rolling_count
        == SETTLING_RUN
    )

    if indices.size == 0:

        return float(
            "nan"
        )

    return float(
        indices[0]
        + 1
    )


def shift_validation(
    probability: float,
    quantile_alpha: float,
    tail_alpha: float,
    shift: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    (
        q_initial_target,
        tail_initial_target,
    ) = tail_targets(
        probability,
        "lower",
        0.0,
    )

    (
        q_final_target,
        tail_final_target,
    ) = tail_targets(
        probability,
        "lower",
        shift,
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    quantile_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    tail_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    for trial in range(
        SHIFT_TRIALS
    ):

        state = (
            CoupledState.new()
        )

        warmup = rng.normal(
            loc=0.0,
            scale=1.0,
            size=SHIFT_WARMUP,
        )

        for raw_observation in warmup:

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
                "lower",
            )

        post = rng.normal(
            loc=shift,
            scale=1.0,
            size=SHIFT_LENGTH,
        )

        for index, raw_observation in enumerate(
            post
        ):

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
                "lower",
            )

            quantile_paths[
                trial,
                index,
            ] = (
                state.quantile.value
            )

            tail_paths[
                trial,
                index,
            ] = (
                state.tail.value
            )

    q_mean = np.mean(
        quantile_paths,
        axis=0,
    )

    tail_mean = np.mean(
        tail_paths,
        axis=0,
    )

    tail_median = np.median(
        tail_paths,
        axis=0,
    )

    mean_overshoot = overshoot(
        tail_mean,
        tail_final_target,
        shift,
    )

    median_overshoot = overshoot(
        tail_median,
        tail_final_target,
        shift,
    )

    return {
        "probability": probability,
        "quantile_alpha": quantile_alpha,
        "tail_alpha": tail_alpha,
        "shift": shift,

        "q_t50": crossing_time(
            q_mean,
            q_initial_target,
            q_final_target,
            0.50,
        ),

        "q_t90": crossing_time(
            q_mean,
            q_initial_target,
            q_final_target,
            0.90,
        ),

        "tail_mean_t50": crossing_time(
            tail_mean,
            tail_initial_target,
            tail_final_target,
            0.50,
        ),

        "tail_mean_t90": crossing_time(
            tail_mean,
            tail_initial_target,
            tail_final_target,
            0.90,
        ),

        "tail_median_t50": crossing_time(
            tail_median,
            tail_initial_target,
            tail_final_target,
            0.50,
        ),

        "tail_median_t90": crossing_time(
            tail_median,
            tail_initial_target,
            tail_final_target,
            0.90,
        ),

        "mean_overshoot": mean_overshoot,

        "mean_overshoot_percent": (
            100.0
            * mean_overshoot
            / abs(
                shift
            )
        ),

        "median_overshoot": median_overshoot,

        "median_overshoot_percent": (
            100.0
            * median_overshoot
            / abs(
                shift
            )
        ),

        "mean_settle": settling_time(
            tail_mean,
            tail_final_target,
            shift,
        ),

        "median_settle": settling_time(
            tail_median,
            tail_final_target,
            shift,
        ),
    }


def print_stationary(
    probability: float,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Exponentially Weighted Conditional Tail Mean"
    )

    print(
        "Stationary Normal(0,1)"
    )

    print(
        f"Lower-tail probability: {probability:.2f}"
    )

    print()

    print(
        "  QAlpha"
        "  TAlpha"
        "       QBias"
        "      QRMSE"
        "    TailBias"
        "     TailMAE"
        "    TailRMSE"
        "    MeanMass"
    )

    print(
        "-" * 94
    )

    for row in rows:

        print(
            f"{row['quantile_alpha']:8.4f}"
            f"{row['tail_alpha']:8.4f}"
            f"{row['q_bias']:12.6f}"
            f"{row['q_rmse']:12.6f}"
            f"{row['tail_bias']:12.6f}"
            f"{row['tail_mae']:12.6f}"
            f"{row['tail_rmse']:12.6f}"
            f"{row['mean_mass']:12.6f}"
        )


def print_shift(
    probability: float,
    shift: float,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Exponentially Weighted Conditional Tail Mean"
    )

    print(
        f"Normal(0,1) -> Normal({shift:+.1f},1)"
    )

    print(
        f"Lower-tail probability: {probability:.2f}"
    )

    print()

    print(
        "  QAlpha"
        "  TAlpha"
        "    Q-T50"
        "    Q-T90"
        "   TM-T50"
        "   TM-T90"
        "  Med-T50"
        "  Med-T90"
        "  MeanOver"
        " MeanOver%"
        "   MedOver"
        "  MedOver%"
        " MeanSettle"
        "  MedSettle"
    )

    print(
        "-" * 146
    )

    for row in rows:

        print(
            f"{row['quantile_alpha']:8.4f}"
            f"{row['tail_alpha']:8.4f}"
            f"{row['q_t50']:9.0f}"
            f"{row['q_t90']:9.0f}"
            f"{row['tail_mean_t50']:9.0f}"
            f"{row['tail_mean_t90']:9.0f}"
            f"{row['tail_median_t50']:9.0f}"
            f"{row['tail_median_t90']:9.0f}"
            f"{row['mean_overshoot']:10.4f}"
            f"{row['mean_overshoot_percent']:10.2f}"
            f"{row['median_overshoot']:10.4f}"
            f"{row['median_overshoot_percent']:10.2f}"
            f"{row['mean_settle']:11.0f}"
            f"{row['median_settle']:11.0f}"
        )


def main() -> None:

    assert_reflection_symmetry()

    stationary_configurations = {
        0.05: [
            (
                0.02,
                0.010,
            ),
            (
                0.02,
                0.005,
            ),
            (
                0.02,
                0.002,
            ),
            (
                0.05,
                0.005,
            ),
        ],

        0.01: [
            (
                0.02,
                0.005,
            ),
            (
                0.02,
                0.002,
            ),
            (
                0.02,
                0.001,
            ),
            (
                0.02,
                0.0005,
            ),
            (
                0.05,
                0.001,
            ),
        ],
    }

    seed_offset = 0

    for (
        probability,
        configurations,
    ) in stationary_configurations.items():

        rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        for (
            quantile_alpha,
            tail_alpha,
        ) in configurations:

            rows.append(
                stationary_validation(
                    probability,
                    quantile_alpha,
                    tail_alpha,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_stationary(
            probability,
            rows,
        )

    shift_configurations = {
        0.05: [
            (
                0.02,
                0.010,
            ),
            (
                0.02,
                0.005,
            ),
            (
                0.02,
                0.002,
            ),
            (
                0.05,
                0.005,
            ),
        ],

        0.01: [
            (
                0.02,
                0.005,
            ),
            (
                0.02,
                0.002,
            ),
            (
                0.02,
                0.001,
            ),
            (
                0.02,
                0.0005,
            ),
            (
                0.05,
                0.001,
            ),
        ],
    }

    for probability, configurations in (
        shift_configurations.items()
    ):

        for shift in (
            2.0,
            -2.0,
        ):

            rows = []

            for (
                quantile_alpha,
                tail_alpha,
            ) in configurations:

                rows.append(
                    shift_validation(
                        probability,
                        quantile_alpha,
                        tail_alpha,
                        shift,
                        seed_offset,
                    )
                )

                seed_offset += 1

            print_shift(
                probability,
                shift,
                rows,
            )


if __name__ == "__main__":
    main()
