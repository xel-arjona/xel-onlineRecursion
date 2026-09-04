"""
Adaptive Tail Mean candidate validation.

This experiment evaluates an O(1) recursive lower/upper tail-mean
estimator based on the Rockafellar-Uryasev / CVaR companion functional.

The experiment deliberately separates two cases:

1. ORACLE THRESHOLD
   The true population quantile is supplied to the tail recursion.

   This isolates the tail-mean recursion itself.

2. ADAPTIVE THRESHOLD
   The threshold is supplied by the same adaptive quantile recursion
   used by the onlineRecursion library.

   This tests the coupled adaptive quantile + tail-mean system.

For a quantile q_p:

Lower tail pseudo-observation:

    w_lower(q, x)
        = q - max(q - x, 0) / p

Upper tail pseudo-observation:

    w_upper(q, x)
        = q + max(x - q, 0) / (1 - p)

At the correct population quantile:

    E[w_lower(q_p, X)]
        = lower tail mean

    E[w_upper(q_p, X)]
        = upper tail mean

For continuous distributions these equal the corresponding conditional
tail expectations.

The adaptive companion recursion is:

    tail += tail_alpha * (pseudo - tail)

The adaptive quantile threshold supplied to the current tail observation
is always the PREVIOUS quantile state, so the current observation cannot
alter its own tail threshold.
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


TRIALS = 50
LENGTH = 10_000
BURN_IN = 6_000

BASE_SEED = 20260903

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

    def reset(
        self,
    ) -> "AdaptiveQuantileState":

        self.value = float(
            "nan"
        )

        self.probability = float(
            "nan"
        )

        self.scale = float(
            "nan"
        )

        return self

    def update(
        self,
        input_value: float,
        probability: float,
        alpha: float,
    ) -> "AdaptiveQuantileState":

        if (
            isnan(
                probability
            )
            or probability <= 0.0
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
            self.value = (
                input_value
            )

            self.probability = (
                probability
            )

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
            score = (
                probability
            )

        elif innovation < 0.0:
            score = (
                probability
                - 1.0
            )

        else:
            score = 0.0

        step = (
            coefficient
            * previous_scale
        )

        self.value += (
            step
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

    def reset(
        self,
    ) -> "AdaptiveTailMeanState":

        self.value = float(
            "nan"
        )

        self.probability = float(
            "nan"
        )

        self.side = None

        return self

    def update(
        self,
        input_value: float,
        threshold: float,
        probability: float,
        alpha: float,
        side: str,
    ) -> "AdaptiveTailMeanState":

        if (
            isnan(
                probability
            )
            or probability <= 0.0
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
            self.value = (
                pseudo_observation
            )

            self.probability = (
                probability
            )

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
class AdaptiveQuantileTailMeanState:

    quantile: AdaptiveQuantileState
    tail_mean: AdaptiveTailMeanState

    @classmethod
    def new(
        cls,
    ) -> "AdaptiveQuantileTailMeanState":

        return cls(
            quantile=AdaptiveQuantileState(),
            tail_mean=AdaptiveTailMeanState(),
        )

    def reset(
        self,
    ) -> "AdaptiveQuantileTailMeanState":

        self.quantile.reset()
        self.tail_mean.reset()

        return self

    def update(
        self,
        input_value: float,
        probability: float,
        quantile_alpha: float,
        tail_alpha: float,
        side: str,
    ) -> "AdaptiveQuantileTailMeanState":

        if isnan(
            self.quantile.value
        ):

            # The first valid observation initializes the quantile.
            self.quantile.update(
                input_value,
                probability,
                quantile_alpha,
            )

            # At initialization q == x, therefore both lower and upper
            # pseudo-observations equal x.
            self.tail_mean.update(
                input_value,
                self.quantile.value,
                probability,
                tail_alpha,
                side,
            )

            return self

        # The tail companion must use the PREVIOUS quantile.
        previous_quantile = (
            self.quantile.value
        )

        self.tail_mean.update(
            input_value,
            previous_quantile,
            probability,
            tail_alpha,
            side,
        )

        # Only after the tail observation has been processed may the
        # current observation update the quantile threshold.
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

        self.error_sum += (
            error
        )

        self.absolute_error_sum += (
            abs(
                error
            )
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


def target_values(
    probability: float,
    side: str,
) -> tuple[
    float,
    float,
]:

    quantile_target = (
        NORMAL.inv_cdf(
            probability
        )
    )

    if side == "lower":

        tail_target = (
            lower_normal_tail_mean(
                probability
            )
        )

    else:

        tail_target = (
            upper_normal_tail_mean(
                probability
            )
        )

    return (
        quantile_target,
        tail_target,
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
    ]

    lower = (
        AdaptiveQuantileTailMeanState.new()
    )

    mirrored_upper = (
        AdaptiveQuantileTailMeanState.new()
    )

    for observation in sample:

        lower.update(
            observation,
            0.05,
            0.20,
            0.10,
            "lower",
        )

        mirrored_upper.update(
            -observation,
            0.95,
            0.20,
            0.10,
            "upper",
        )

        if not np.isclose(
            lower.quantile.value,
            -mirrored_upper.quantile.value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Quantile reflection symmetry failed."
            )

        if not np.isclose(
            lower.tail_mean.value,
            -mirrored_upper.tail_mean.value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Tail-mean reflection symmetry failed."
            )


def validate_configuration(
    probability: float,
    side: str,
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
    ) = target_values(
        probability,
        side,
    )

    oracle_metrics = (
        Metrics()
    )

    adaptive_quantile_metrics = (
        Metrics()
    )

    adaptive_tail_metrics = (
        Metrics()
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        TRIALS
    ):

        sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=LENGTH,
        )

        oracle_tail = (
            AdaptiveTailMeanState()
        )

        adaptive_pair = (
            AdaptiveQuantileTailMeanState.new()
        )

        for index, raw_observation in enumerate(
            sample
        ):

            observation = float(
                raw_observation
            )

            oracle_tail.update(
                observation,
                quantile_target,
                probability,
                tail_alpha,
                side,
            )

            adaptive_pair.update(
                observation,
                probability,
                quantile_alpha,
                tail_alpha,
                side,
            )

            if index >= BURN_IN:

                oracle_metrics.add(
                    oracle_tail.value,
                    tail_target,
                )

                adaptive_quantile_metrics.add(
                    adaptive_pair.quantile.value,
                    quantile_target,
                )

                adaptive_tail_metrics.add(
                    adaptive_pair.tail_mean.value,
                    tail_target,
                )

    return {
        "probability": (
            probability
        ),
        "quantile_alpha": (
            quantile_alpha
        ),
        "tail_alpha": (
            tail_alpha
        ),
        "quantile_target": (
            quantile_target
        ),
        "tail_target": (
            tail_target
        ),
        "quantile_bias": (
            adaptive_quantile_metrics.bias
        ),
        "quantile_mae": (
            adaptive_quantile_metrics.mae
        ),
        "quantile_rmse": (
            adaptive_quantile_metrics.rmse
        ),
        "oracle_bias": (
            oracle_metrics.bias
        ),
        "oracle_mae": (
            oracle_metrics.mae
        ),
        "oracle_rmse": (
            oracle_metrics.rmse
        ),
        "adaptive_bias": (
            adaptive_tail_metrics.bias
        ),
        "adaptive_mae": (
            adaptive_tail_metrics.mae
        ),
        "adaptive_rmse": (
            adaptive_tail_metrics.rmse
        ),
    }


def print_results(
    side: str,
    results: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()
    print(
        "Adaptive Tail Mean Stationary Validation"
    )

    print(
        "Distribution: Normal(0,1)"
    )

    print(
        f"Tail Side: {side}"
    )

    print(
        f"Trials: {TRIALS}"
    )

    print(
        f"Length: {LENGTH}"
    )

    print(
        f"Burn-in: {BURN_IN}"
    )

    print()

    print(
        "    Prob   QAlpha   TAlpha"
        "     QTarget"
        "    TailTarget"
        "       QBias"
        "      QRMSE"
        "    OracleBias"
        "    OracleRMSE"
        "   AdaptiveBias"
        "   AdaptiveMAE"
        "   AdaptiveRMSE"
    )

    print(
        "-" * 154
    )

    for row in results:

        print(
            f"{row['probability']:8.2f}"
            f"{row['quantile_alpha']:9.3f}"
            f"{row['tail_alpha']:9.3f}"
            f"{row['quantile_target']:12.6f}"
            f"{row['tail_target']:14.6f}"
            f"{row['quantile_bias']:12.6f}"
            f"{row['quantile_rmse']:12.6f}"
            f"{row['oracle_bias']:14.6f}"
            f"{row['oracle_rmse']:14.6f}"
            f"{row['adaptive_bias']:15.6f}"
            f"{row['adaptive_mae']:14.6f}"
            f"{row['adaptive_rmse']:15.6f}"
        )


def main() -> None:

    assert_reflection_symmetry()

    configurations = [
        # Same quantile/tail adaptation coefficient.
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

        # Slower tail state than quantile state.
        (
            0.02,
            0.01,
        ),
    ]

    cases = [
        (
            "lower",
            0.05,
        ),
        (
            "upper",
            0.95,
        ),
        (
            "lower",
            0.01,
        ),
        (
            "upper",
            0.99,
        ),
    ]

    seed_offset = 0

    for side, probability in cases:

        results: list[
            dict[
                str,
                float,
            ]
        ] = []

        for (
            quantile_alpha,
            tail_alpha,
        ) in configurations:

            results.append(
                validate_configuration(
                    probability,
                    side,
                    quantile_alpha,
                    tail_alpha,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_results(
            side,
            results,
        )


if __name__ == "__main__":
    main()
