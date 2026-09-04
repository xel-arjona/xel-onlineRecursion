"""
Adaptive Tail Mean regime-shift validation.

Tests the coupled adaptive-quantile + adaptive-tail-mean system under
location shifts.

The purpose is to measure the bias/variance versus adaptation-speed tradeoff
created by using a tail alpha that is independent from the quantile alpha.

For Normal(mu, 1), both the population quantile and corresponding tail mean
shift by exactly mu. This provides analytically known pre- and post-shift
targets.

The tail companion always processes the current observation using the
PREVIOUS quantile estimate.
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


TRIALS = 50

WARMUP = 8_000
POST_SHIFT = 16_000

BASE_SEED = 20260904

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


def upper_tail_mean_standard_normal(
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

    side: str | None = None

    def update(
        self,
        input_value: float,
        threshold: float,
        probability: float,
        alpha: float,
        side: str,
    ) -> "AdaptiveTailMeanState":

        if (
            probability <= 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "Tail probability must be strictly between zero and one."
            )

        if side not in (
            "lower",
            "upper",
        ):
            raise ValueError(
                "Tail side must be 'lower' or 'upper'."
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

        if (
            self.side is not None
            and side
            != self.side
        ):
            raise ValueError(
                "Tail side cannot change inside an active state."
            )

        coefficient = clamp(
            alpha,
            0.0,
            1.0,
        )

        if side == "lower":

            pseudo_observation = (
                threshold
                - max(
                    threshold
                    - input_value,
                    0.0,
                )
                / probability
            )

        else:

            pseudo_observation = (
                threshold
                + max(
                    input_value
                    - threshold,
                    0.0,
                )
                / (
                    1.0
                    - probability
                )
            )

        if isnan(
            self.value
        ):
            self.value = pseudo_observation
            self.probability = probability
            self.side = side

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
                probability,
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
            probability,
            tail_alpha,
            side,
        )

        self.quantile.update(
            input_value,
            probability,
            quantile_alpha,
        )

        return self


def targets(
    probability: float,
    side: str,
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

    if side == "lower":

        tail_mean = (
            location
            + lower_tail_mean_standard_normal(
                probability
            )
        )

    else:

        tail_mean = (
            location
            + upper_tail_mean_standard_normal(
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

    if len(
        indices
    ) == 0:
        return float(
            "nan"
        )

    return float(
        indices[0]
        + 1
    )


def validate_shift(
    probability: float,
    side: str,
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
    ) = targets(
        probability,
        side,
        0.0,
    )

    (
        q_final_target,
        tail_final_target,
    ) = targets(
        probability,
        side,
        shift,
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    q_initial_values: list[
        float
    ] = []

    tail_initial_values: list[
        float
    ] = []

    q_t50: list[
        float
    ] = []

    q_t90: list[
        float
    ] = []

    tail_t50: list[
        float
    ] = []

    tail_t90: list[
        float
    ] = []

    late_q_errors: list[
        float
    ] = []

    late_tail_errors: list[
        float
    ] = []

    late_start = (
        POST_SHIFT
        * 3
        // 4
    )

    for _ in range(
        TRIALS
    ):

        state = (
            CoupledState.new()
        )

        warmup_sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=WARMUP,
        )

        for raw_observation in warmup_sample:

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
                side,
            )

        q_initial_values.append(
            state.quantile.value
        )

        tail_initial_values.append(
            state.tail.value
        )

        post_sample = rng.normal(
            loc=shift,
            scale=1.0,
            size=POST_SHIFT,
        )

        q_path = np.empty(
            POST_SHIFT,
            dtype=float,
        )

        tail_path = np.empty(
            POST_SHIFT,
            dtype=float,
        )

        for index, raw_observation in enumerate(
            post_sample
        ):

            state.update(
                float(
                    raw_observation
                ),
                probability,
                quantile_alpha,
                tail_alpha,
                side,
            )

            q_path[
                index
            ] = (
                state.quantile.value
            )

            tail_path[
                index
            ] = (
                state.tail.value
            )

        q_t50.append(
            crossing_time(
                q_path,
                q_initial_target,
                q_final_target,
                0.50,
            )
        )

        q_t90.append(
            crossing_time(
                q_path,
                q_initial_target,
                q_final_target,
                0.90,
            )
        )

        tail_t50.append(
            crossing_time(
                tail_path,
                tail_initial_target,
                tail_final_target,
                0.50,
            )
        )

        tail_t90.append(
            crossing_time(
                tail_path,
                tail_initial_target,
                tail_final_target,
                0.90,
            )
        )

        late_q_errors.extend(
            (
                q_path[
                    late_start:
                ]
                - q_final_target
            ).tolist()
        )

        late_tail_errors.extend(
            (
                tail_path[
                    late_start:
                ]
                - tail_final_target
            ).tolist()
        )

    q_errors = np.asarray(
        late_q_errors,
        dtype=float,
    )

    tail_errors = np.asarray(
        late_tail_errors,
        dtype=float,
    )

    return {
        "probability": probability,
        "quantile_alpha": quantile_alpha,
        "tail_alpha": tail_alpha,
        "shift": shift,
        "q_initial": float(
            np.mean(
                q_initial_values
            )
        ),
        "tail_initial": float(
            np.mean(
                tail_initial_values
            )
        ),
        "q_target": q_final_target,
        "tail_target": tail_final_target,
        "q_t50": float(
            np.nanmedian(
                q_t50
            )
        ),
        "q_t90": float(
            np.nanmedian(
                q_t90
            )
        ),
        "tail_t50": float(
            np.nanmedian(
                tail_t50
            )
        ),
        "tail_t90": float(
            np.nanmedian(
                tail_t90
            )
        ),
        "q_late_bias": float(
            np.mean(
                q_errors
            )
        ),
        "q_late_mae": float(
            np.mean(
                np.abs(
                    q_errors
                )
            )
        ),
        "tail_late_bias": float(
            np.mean(
                tail_errors
            )
        ),
        "tail_late_mae": float(
            np.mean(
                np.abs(
                    tail_errors
                )
            )
        ),
    }


def print_results(
    side: str,
    probability: float,
    shift: float,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    direction = (
        f"{shift:+.1f}"
    )

    print()
    print(
        "Adaptive Quantile + Tail Mean Shift Validation"
    )

    print(
        f"Normal(0,1) -> Normal({direction},1)"
    )

    print(
        f"Tail Side: {side}"
    )

    print(
        f"Probability: {probability:.2f}"
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
        "      QInit"
        "     QTarget"
        "   Q-T50"
        "   Q-T90"
        "   TailInit"
        "  TailTarget"
        "   T-T50"
        "   T-T90"
        "   QLateBias"
        "    QLateMAE"
        "   TLateBias"
        "    TLateMAE"
    )

    print(
        "-" * 158
    )

    for row in rows:

        print(
            f"{row['quantile_alpha']:8.3f}"
            f"{row['tail_alpha']:8.3f}"
            f"{row['q_initial']:11.6f}"
            f"{row['q_target']:12.6f}"
            f"{row['q_t50']:8.0f}"
            f"{row['q_t90']:8.0f}"
            f"{row['tail_initial']:12.6f}"
            f"{row['tail_target']:12.6f}"
            f"{row['tail_t50']:8.0f}"
            f"{row['tail_t90']:8.0f}"
            f"{row['q_late_bias']:12.6f}"
            f"{row['q_late_mae']:12.6f}"
            f"{row['tail_late_bias']:12.6f}"
            f"{row['tail_late_mae']:12.6f}"
        )


def main() -> None:

    configurations = [
        # Equal timescale.
        (
            0.01,
            0.01,
        ),
        (
            0.02,
            0.02,
        ),
        (
            0.05,
            0.05,
        ),

        # Tail state slower than quantile.
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
            "lower",
            0.05,
            2.0,
        ),
        (
            "lower",
            0.05,
            -2.0,
        ),
        (
            "upper",
            0.95,
            2.0,
        ),
        (
            "upper",
            0.95,
            -2.0,
        ),
        (
            "lower",
            0.01,
            2.0,
        ),
        (
            "lower",
            0.01,
            -2.0,
        ),
        (
            "upper",
            0.99,
            2.0,
        ),
        (
            "upper",
            0.99,
            -2.0,
        ),
    ]

    seed_offset = 0

    for (
        side,
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
                validate_shift(
                    probability,
                    side,
                    quantile_alpha,
                    tail_alpha,
                    shift,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_results(
            side,
            probability,
            shift,
            rows,
        )


if __name__ == "__main__":
    main()
