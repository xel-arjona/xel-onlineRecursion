"""
Adaptive Huber Location candidate validation.

Tests the O(1) recursive Huber-location update:

    innovation = x - location

    limit = tuning * scale

    correction = clamp(
        innovation,
        -limit,
        +limit,
    )

    location += alpha * correction

The scale is EXTERNAL to the Huber location state.

This first experiment deliberately supplies oracle scale = 1 so that the
location recursion can be validated independently from any adaptive-scale
estimator.

For a symmetric zero-location population, the population Huber location is
zero for every positive tuning constant.

Inside the clipping region, the recursion is exactly a first-order IIR:

    location += alpha * innovation

Outside the clipping region, one-observation influence is bounded by:

    abs(delta location)
        <= alpha * tuning * scale
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    isnan,
    sqrt,
)

import numpy as np


TRIALS = 50
LENGTH = 10_000
BURN_IN = 6_000

BASE_SEED = 20260908


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


@dataclass
class AdaptiveHuberState:

    value: float = float(
        "nan"
    )

    tuning: float = float(
        "nan"
    )

    def reset(
        self,
    ) -> "AdaptiveHuberState":

        self.value = float(
            "nan"
        )

        self.tuning = float(
            "nan"
        )

        return self

    def update(
        self,
        input_value: float,
        scale: float,
        tuning: float,
        alpha: float,
    ) -> "AdaptiveHuberState":

        if (
            isnan(
                tuning
            )
            or tuning <= 0.0
        ):
            raise ValueError(
                "Huber tuning must be strictly positive."
            )

        if (
            not isnan(
                self.tuning
            )
            and tuning
            != self.tuning
        ):
            raise ValueError(
                "Huber tuning cannot change inside an active state."
            )

        if scale < 0.0:
            raise ValueError(
                "Huber scale cannot be negative."
            )

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

            self.tuning = (
                tuning
            )

            return self

        innovation = (
            input_value
            - self.value
        )

        limit = (
            tuning
            * scale
        )

        correction = clamp(
            innovation,
            -limit,
            +limit,
        )

        self.value += (
            coefficient
            * correction
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


def assert_deterministic_properties() -> None:

    # ------------------------------------------------------------------
    # INNER REGION EXACTLY MATCHES IIR
    # ------------------------------------------------------------------

    huber = AdaptiveHuberState()

    huber.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    # Innovation = +1, which lies inside +/-1.5.
    huber.update(
        11.0,
        1.0,
        1.5,
        0.20,
    )

    expected = (
        10.0
        + 0.20
        * (
            11.0
            - 10.0
        )
    )

    if not np.isclose(
        huber.value,
        expected,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Inner-region IIR equivalence failed."
        )

    # ------------------------------------------------------------------
    # OUTLIER CORRECTION IS BOUNDED
    # ------------------------------------------------------------------

    previous = (
        huber.value
    )

    huber.update(
        1_000_000.0,
        1.0,
        1.5,
        0.20,
    )

    maximum_delta = (
        0.20
        * 1.5
        * 1.0
    )

    actual_delta = (
        huber.value
        - previous
    )

    if not np.isclose(
        actual_delta,
        maximum_delta,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Bounded positive influence failed."
        )

    # ------------------------------------------------------------------
    # REFLECTION SYMMETRY
    # ------------------------------------------------------------------

    sample = [
        3.0,
        -7.0,
        2.0,
        20.0,
        -1.0,
        -30.0,
        4.0,
    ]

    positive = AdaptiveHuberState()
    negative = AdaptiveHuberState()

    for observation in sample:

        positive.update(
            observation,
            2.0,
            1.5,
            0.10,
        )

        negative.update(
            -observation,
            2.0,
            1.5,
            0.10,
        )

        if not np.isclose(
            positive.value,
            -negative.value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Reflection symmetry failed."
            )

    # ------------------------------------------------------------------
    # TRANSLATION EQUIVARIANCE
    # ------------------------------------------------------------------

    offset = 100.0

    original = AdaptiveHuberState()
    translated = AdaptiveHuberState()

    for observation in sample:

        original.update(
            observation,
            2.0,
            1.5,
            0.10,
        )

        translated.update(
            observation
            + offset,
            2.0,
            1.5,
            0.10,
        )

    if not np.isclose(
        translated.value,
        original.value
        + offset,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Translation equivariance failed."
        )

    # ------------------------------------------------------------------
    # POSITIVE-SCALE EQUIVARIANCE
    # ------------------------------------------------------------------

    factor = 7.0

    original = AdaptiveHuberState()
    scaled = AdaptiveHuberState()

    for observation in sample:

        original.update(
            observation,
            2.0,
            1.5,
            0.10,
        )

        scaled.update(
            factor
            * observation,
            factor
            * 2.0,
            1.5,
            0.10,
        )

    if not np.isclose(
        scaled.value,
        factor
        * original.value,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Positive-scale equivariance failed."
        )


def generate_normal(
    rng: np.random.Generator,
) -> np.ndarray:

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=LENGTH,
    )


def generate_contaminated(
    rng: np.random.Generator,
) -> np.ndarray:

    sample = rng.normal(
        loc=0.0,
        scale=1.0,
        size=LENGTH,
    )

    contaminated = (
        rng.random(
            LENGTH
        )
        < 0.01
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
            loc=0.0,
            scale=20.0,
            size=count,
        )

    return sample


def validate_configuration(
    generator,
    tuning: float,
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    huber_metrics = Metrics()
    iir_metrics = Metrics()

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        TRIALS
    ):

        huber = AdaptiveHuberState()

        iir_value = float(
            "nan"
        )

        sample = generator(
            rng
        )

        for index, raw_observation in enumerate(
            sample
        ):

            observation = float(
                raw_observation
            )

            huber.update(
                observation,
                1.0,
                tuning,
                alpha,
            )

            if isnan(
                iir_value
            ):

                iir_value = (
                    observation
                )

            else:

                iir_value += (
                    alpha
                    * (
                        observation
                        - iir_value
                    )
                )

            if index >= BURN_IN:

                huber_metrics.add(
                    huber.value,
                    0.0,
                )

                iir_metrics.add(
                    iir_value,
                    0.0,
                )

    return {
        "tuning": tuning,
        "alpha": alpha,

        "huber_bias": (
            huber_metrics.bias
        ),

        "huber_mae": (
            huber_metrics.mae
        ),

        "huber_rmse": (
            huber_metrics.rmse
        ),

        "iir_bias": (
            iir_metrics.bias
        ),

        "iir_mae": (
            iir_metrics.mae
        ),

        "iir_rmse": (
            iir_metrics.rmse
        ),
    }


def print_results(
    distribution_name: str,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Adaptive Huber Oracle-Scale Validation"
    )

    print(
        f"Distribution: {distribution_name}"
    )

    print(
        "Oracle scale: 1.0"
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
        "  Tuning"
        "   Alpha"
        "   HuberBias"
        "    HuberMAE"
        "   HuberRMSE"
        "     IIRBias"
        "      IIRMAE"
        "     IIRRMSE"
        "  RMSE Ratio"
    )

    print(
        "-" * 107
    )

    for row in rows:

        ratio = (
            row[
                "huber_rmse"
            ]
            / row[
                "iir_rmse"
            ]
        )

        print(
            f"{row['tuning']:8.3f}"
            f"{row['alpha']:8.3f}"
            f"{row['huber_bias']:12.6f}"
            f"{row['huber_mae']:12.6f}"
            f"{row['huber_rmse']:12.6f}"
            f"{row['iir_bias']:12.6f}"
            f"{row['iir_mae']:12.6f}"
            f"{row['iir_rmse']:12.6f}"
            f"{ratio:12.4f}"
        )


def main() -> None:

    assert_deterministic_properties()

    configurations = [
        (
            0.75,
            0.01,
        ),
        (
            1.345,
            0.01,
        ),
        (
            2.0,
            0.01,
        ),
        (
            1.345,
            0.02,
        ),
        (
            1.345,
            0.05,
        ),
    ]

    distributions = [
        (
            "Normal(0,1)",
            generate_normal,
        ),
        (
            "1% N(0,20^2) contamination",
            generate_contaminated,
        ),
    ]

    seed_offset = 0

    for (
        distribution_name,
        generator,
    ) in distributions:

        rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        for (
            tuning,
            alpha,
        ) in configurations:

            rows.append(
                validate_configuration(
                    generator,
                    tuning,
                    alpha,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_results(
            distribution_name,
            rows,
        )


if __name__ == "__main__":
    main()
