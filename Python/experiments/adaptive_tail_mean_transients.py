"""
Adaptive Tail Mean transient-shape validation.

This experiment follows the stationary and first-crossing shift validations.

The previous shift experiment showed strongly directional adaptation:
tailward distribution shifts can cross the new tail-mean target almost
immediately, especially at extreme probabilities.

First-crossing T50/T90 alone cannot distinguish controlled adaptation from
overshoot.

This experiment therefore evaluates ensemble transient geometry:

    - ensemble-mean T50
    - ensemble-mean T90
    - ensemble-median T50
    - ensemble-median T90
    - maximum ensemble-mean overshoot beyond the new target
    - normalized overshoot relative to the true location shift
    - settling time inside a +/-10% shift-error band

The adaptive tail mean uses the PREVIOUS adaptive quantile as its threshold.

Only lower-tail cases are required here because upper-tail behavior has
already been validated as the sign/complement reflection of the lower tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    exp,
    isnan,
    pi,
    sqrt,
)
from statistics import NormalDist

import numpy as np


TRIALS = 100

WARMUP = 8_000
POST_SHIFT = 12_000

SETTLING_FRACTION = 0.10
SETTLING_RUN = 500

BASE_SEED = 20260905

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


def lower_tail_mean_standard_normal(
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
class AdaptiveTailMeanState:

    value: float = float(
        "nan"
    )

    probability: float = float(
        "nan"
    )

    def update(
        self,
        input_value: float,
        threshold: float,
        probability: float,
        alpha: float,
    ) -> "AdaptiveTailMeanState":

        if (
            probability <= 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "Tail probability must be strictly between zero and one."
            )

        if (
            not isnan(
                self.probability
            )
            and probability
            != self.probability
        ):
            raise ValueError(
                "Tail probability cannot change inside an active state."
            )

        coefficient = clamp(
            alpha,
            0.0,
            1.0,
        )

        pseudo_observation = (
            threshold
            - max(
                threshold
                - input_value,
                0.0,
            )
            / probability
        )

        if isnan(
            self.value
        ):
            self.value = pseudo_observation
            self.probability = probability

            return self

        self.value += (
            coefficient
            * (
                pseudo_observation
                - self.value
            )
        )

        return self


@dataclass
class CoupledState:

    quantile: AdaptiveQuantileState
    tail: AdaptiveTailMeanState

    @classmethod
    def new(
        cls,
    ) -> "CoupledState":

        return cls(
            quantile=AdaptiveQuantileState(),
            tail=AdaptiveTailMeanState(),
        )

    def update(
        self,
        input_value: float,
        probability: float,
        quantile_alpha: float,
        tail_alpha: float,
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
                probability,
                tail_alpha,
            )

            return self

        previous_quantile = (
            self.quantile.value
        )

        self.tail.update(
            input_value,
            previous_quantile,
            probability,
            tail_alpha,
        )

        self.quantile.update(
            input_value,
            probability,
            quantile_alpha,
        )

        return self


def population_targets(
    probability: float,
    location: float,
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

    tail_mean = (
        location
        + lower_tail_mean_standard_normal(
            probability
        )
    )

    return (
        quantile,
        tail_mean,
    )


def crossing_time(
    path: np.ndarray,
    initial_target: float,
    final_target: float,
    fraction: float,
) -> float:

    crossing_target = (
        initial_target
        + fraction
        * (
            final_target
            - initial_target
        )
    )

    if final_target > initial_target:

        indices = np.flatnonzero(
            path >= crossing_target
        )

    else:

        indices = np.flatnonzero(
            path <= crossing_target
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

    if len(
        path
    ) < SETTLING_RUN:

        return float(
            "nan"
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


def validate_case(
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
    ) = population_targets(
        probability,
        0.0,
    )

    (
        q_final_target,
        tail_final_target,
    ) = population_targets(
        probability,
        shift,
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    quantile_paths = np.empty(
        (
            TRIALS,
            POST_SHIFT,
        ),
        dtype=float,
    )

    tail_paths = np.empty(
        (
            TRIALS,
            POST_SHIFT,
        ),
        dtype=float,
    )

    for trial in range(
        TRIALS
    ):

        state = (
            CoupledState.new()
        )

        warmup = rng.normal(
            loc=0.0,
            scale=1.0,
            size=WARMUP,
        )

        for raw_observation in warmup:

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
            )

        post = rng.normal(
            loc=shift,
            scale=1.0,
            size=POST_SHIFT,
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

    q_mean_path = np.mean(
        quantile_paths,
        axis=0,
    )

    tail_mean_path = np.mean(
        tail_paths,
        axis=0,
    )

    tail_median_path = np.median(
        tail_paths,
        axis=0,
    )

    mean_overshoot = overshoot(
        tail_mean_path,
        tail_final_target,
        shift,
    )

    median_overshoot = overshoot(
        tail_median_path,
        tail_final_target,
        shift,
    )

    shift_size = abs(
        shift
    )

    return {
        "probability": probability,
        "quantile_alpha": quantile_alpha,
        "tail_alpha": tail_alpha,
        "shift": shift,

        "q_mean_t50": crossing_time(
            q_mean_path,
            q_initial_target,
            q_final_target,
            0.50,
        ),

        "q_mean_t90": crossing_time(
            q_mean_path,
            q_initial_target,
            q_final_target,
            0.90,
        ),

        "tail_mean_t50": crossing_time(
            tail_mean_path,
            tail_initial_target,
            tail_final_target,
            0.50,
        ),

        "tail_mean_t90": crossing_time(
            tail_mean_path,
            tail_initial_target,
            tail_final_target,
            0.90,
        ),

        "tail_median_t50": crossing_time(
            tail_median_path,
            tail_initial_target,
            tail_final_target,
            0.50,
        ),

        "tail_median_t90": crossing_time(
            tail_median_path,
            tail_initial_target,
            tail_final_target,
            0.90,
        ),

        "mean_overshoot": (
            mean_overshoot
        ),

        "mean_overshoot_fraction": (
            mean_overshoot
            / shift_size
        ),

        "median_overshoot": (
            median_overshoot
        ),

        "median_overshoot_fraction": (
            median_overshoot
            / shift_size
        ),

        "mean_settle": settling_time(
            tail_mean_path,
            tail_final_target,
            shift,
        ),

        "median_settle": settling_time(
            tail_median_path,
            tail_final_target,
            shift,
        ),
    }


def print_case(
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
        "Adaptive Lower Tail Mean Transient Shape"
    )

    print(
        f"Probability: {probability:.2f}"
    )

    print(
        f"Normal(0,1) -> Normal({shift:+.1f},1)"
    )

    print(
        f"Trials: {TRIALS}"
    )

    print(
        f"Warmup: {WARMUP}"
    )

    print(
        f"Post-shift: {POST_SHIFT}"
    )

    print()

    print(
        "  QAlpha  TAlpha"
        "  QMean50"
        "  QMean90"
        "  TMean50"
        "  TMean90"
        "   TMed50"
        "   TMed90"
        "  MeanOver"
        " MeanOver%"
        "   MedOver"
        "  MedOver%"
        " MeanSettle"
        "  MedSettle"
    )

    print(
        "-" * 145
    )

    for row in rows:

        print(
            f"{row['quantile_alpha']:8.3f}"
            f"{row['tail_alpha']:8.3f}"
            f"{row['q_mean_t50']:9.0f}"
            f"{row['q_mean_t90']:9.0f}"
            f"{row['tail_mean_t50']:9.0f}"
            f"{row['tail_mean_t90']:9.0f}"
            f"{row['tail_median_t50']:9.0f}"
            f"{row['tail_median_t90']:9.0f}"
            f"{row['mean_overshoot']:10.4f}"
            f"{100.0 * row['mean_overshoot_fraction']:10.2f}"
            f"{row['median_overshoot']:10.4f}"
            f"{100.0 * row['median_overshoot_fraction']:10.2f}"
            f"{row['mean_settle']:11.0f}"
            f"{row['median_settle']:11.0f}"
        )


def main() -> None:

    configurations = [
        (
            0.01,
            0.01,
        ),
        (
            0.02,
            0.02,
        ),
        (
            0.02,
            0.01,
        ),
        (
            0.02,
            0.005,
        ),
        (
            0.05,
            0.01,
        ),
    ]

    cases = [
        (
            0.05,
            2.0,
        ),
        (
            0.05,
            -2.0,
        ),
        (
            0.01,
            2.0,
        ),
        (
            0.01,
            -2.0,
        ),
    ]

    seed_offset = 0

    for (
        probability,
        shift,
    ) in cases:

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
                validate_case(
                    probability,
                    quantile_alpha,
                    tail_alpha,
                    shift,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_case(
            probability,
            shift,
            rows,
        )


if __name__ == "__main__":
    main()
