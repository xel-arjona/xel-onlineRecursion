"""
Adaptive Huber Location bandwidth-normalized validation.

The previous oracle-scale experiment compared Huber and ordinary IIR using
the same raw alpha.

That comparison is not dynamically neutral because clipping reduces the
local restoring slope of the Huber recursion.

For standard Normal innovations and Huber tuning c:

    A(c) = P(|Z| < c)

Near the population location:

    E[clip(X - m, -c, +c)]
        ~= -A(c) * m

so the local Huber adaptation coefficient is approximately:

    effective_alpha ~= huber_alpha * A(c)

To compare against an IIR having coefficient beta at approximately equal
local bandwidth, use:

    huber_alpha = beta / A(c)

This experiment compares:

1. ordinary IIR with coefficient beta;
2. raw Huber with alpha = beta;
3. bandwidth-matched Huber with alpha = beta / A(c).

It evaluates:

    - stationary Normal(0,1);
    - symmetric 1% N(0,20^2) contamination;
    - ensemble-mean response to a +2 location shift.

The Huber scale remains oracle scale = 1.

The contaminated population is:

    99% N(location, 1)
     1% N(location, 20^2)

so both IIR and Huber have population location target = location.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    erf,
    exp,
    isnan,
    pi,
    sqrt,
)

import numpy as np


STATIONARY_TRIALS = 50
STATIONARY_LENGTH = 10_000
STATIONARY_BURN_IN = 6_000

SHIFT_TRIALS = 100
SHIFT_WARMUP = 8_000
SHIFT_LENGTH = 4_000

BASE_SEED = 20260909


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


def normal_cdf(
    value: float,
) -> float:

    return (
        0.5
        * (
            1.0
            + erf(
                value
                / sqrt(
                    2.0
                )
            )
        )
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


def huber_A(
    tuning: float,
) -> float:
    """
    E[psi'(Z)] for standard Normal Z and Huber clipping.
    """

    return (
        2.0
        * normal_cdf(
            tuning
        )
        - 1.0
    )


def huber_B(
    tuning: float,
) -> float:
    """
    E[psi(Z)^2] for standard Normal Z and Huber clipping.
    """

    central_second_moment = (
        2.0
        * (
            normal_cdf(
                tuning
            )
            - 0.5
            - tuning
            * normal_pdf(
                tuning
            )
        )
    )

    clipped_tail_probability = (
        2.0
        * (
            1.0
            - normal_cdf(
                tuning
            )
        )
    )

    return (
        central_second_moment
        + tuning
        * tuning
        * clipped_tail_probability
    )


@dataclass
class AdaptiveHuberState:

    value: float = float(
        "nan"
    )

    tuning: float = float(
        "nan"
    )

    def update(
        self,
        input_value: float,
        scale: float,
        tuning: float,
        alpha: float,
    ) -> "AdaptiveHuberState":

        if (
            tuning <= 0.0
            or isnan(
                tuning
            )
        ):
            raise ValueError(
                "Huber tuning must be strictly positive."
            )

        if scale < 0.0:
            raise ValueError(
                "Huber scale cannot be negative."
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
class IIRState:

    value: float = float(
        "nan"
    )

    def update(
        self,
        input_value: float,
        alpha: float,
    ) -> "IIRState":

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

            return self

        self.value += (
            coefficient
            * (
                input_value
                - self.value
            )
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


def generate_sample(
    rng: np.random.Generator,
    length: int,
    location: float,
    contaminated: bool,
) -> np.ndarray:

    sample = rng.normal(
        loc=location,
        scale=1.0,
        size=length,
    )

    if not contaminated:

        return sample

    contamination = (
        rng.random(
            length
        )
        < 0.01
    )

    contamination_count = int(
        np.sum(
            contamination
        )
    )

    if contamination_count > 0:

        sample[
            contamination
        ] = rng.normal(
            loc=location,
            scale=20.0,
            size=contamination_count,
        )

    return sample


def validate_stationary(
    tuning: float,
    beta: float,
    contaminated: bool,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    A = huber_A(
        tuning
    )

    B = huber_B(
        tuning
    )

    matched_alpha = clamp(
        beta
        / A,
        0.0,
        1.0,
    )

    iir_metrics = Metrics()

    raw_huber_metrics = (
        Metrics()
    )

    matched_huber_metrics = (
        Metrics()
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        STATIONARY_TRIALS
    ):

        iir = IIRState()

        raw_huber = (
            AdaptiveHuberState()
        )

        matched_huber = (
            AdaptiveHuberState()
        )

        sample = generate_sample(
            rng,
            STATIONARY_LENGTH,
            0.0,
            contaminated,
        )

        for index, raw_observation in enumerate(
            sample
        ):

            observation = float(
                raw_observation
            )

            iir.update(
                observation,
                beta,
            )

            raw_huber.update(
                observation,
                1.0,
                tuning,
                beta,
            )

            matched_huber.update(
                observation,
                1.0,
                tuning,
                matched_alpha,
            )

            if index >= STATIONARY_BURN_IN:

                iir_metrics.add(
                    iir.value,
                    0.0,
                )

                raw_huber_metrics.add(
                    raw_huber.value,
                    0.0,
                )

                matched_huber_metrics.add(
                    matched_huber.value,
                    0.0,
                )

    theoretical_rmse_ratio = sqrt(
        B
        / (
            A
            * A
        )
    )

    return {
        "tuning": tuning,
        "beta": beta,
        "A": A,
        "B": B,

        "matched_alpha": (
            matched_alpha
        ),

        "theoretical_ratio": (
            theoretical_rmse_ratio
        ),

        "iir_bias": (
            iir_metrics.bias
        ),

        "iir_rmse": (
            iir_metrics.rmse
        ),

        "raw_bias": (
            raw_huber_metrics.bias
        ),

        "raw_rmse": (
            raw_huber_metrics.rmse
        ),

        "matched_bias": (
            matched_huber_metrics.bias
        ),

        "matched_rmse": (
            matched_huber_metrics.rmse
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


def validate_shift(
    tuning: float,
    beta: float,
    contaminated: bool,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    A = huber_A(
        tuning
    )

    matched_alpha = clamp(
        beta
        / A,
        0.0,
        1.0,
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    iir_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    raw_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    matched_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    for trial in range(
        SHIFT_TRIALS
    ):

        iir = IIRState()

        raw_huber = (
            AdaptiveHuberState()
        )

        matched_huber = (
            AdaptiveHuberState()
        )

        warmup = generate_sample(
            rng,
            SHIFT_WARMUP,
            0.0,
            contaminated,
        )

        for raw_observation in warmup:

            observation = float(
                raw_observation
            )

            iir.update(
                observation,
                beta,
            )

            raw_huber.update(
                observation,
                1.0,
                tuning,
                beta,
            )

            matched_huber.update(
                observation,
                1.0,
                tuning,
                matched_alpha,
            )

        post_shift = generate_sample(
            rng,
            SHIFT_LENGTH,
            2.0,
            contaminated,
        )

        for index, raw_observation in enumerate(
            post_shift
        ):

            observation = float(
                raw_observation
            )

            iir.update(
                observation,
                beta,
            )

            raw_huber.update(
                observation,
                1.0,
                tuning,
                beta,
            )

            matched_huber.update(
                observation,
                1.0,
                tuning,
                matched_alpha,
            )

            iir_paths[
                trial,
                index,
            ] = (
                iir.value
            )

            raw_paths[
                trial,
                index,
            ] = (
                raw_huber.value
            )

            matched_paths[
                trial,
                index,
            ] = (
                matched_huber.value
            )

    iir_mean = np.mean(
        iir_paths,
        axis=0,
    )

    raw_mean = np.mean(
        raw_paths,
        axis=0,
    )

    matched_mean = np.mean(
        matched_paths,
        axis=0,
    )

    return {
        "tuning": tuning,
        "beta": beta,
        "matched_alpha": matched_alpha,

        "iir_t50": crossing_time(
            iir_mean,
            0.0,
            2.0,
            0.50,
        ),

        "iir_t90": crossing_time(
            iir_mean,
            0.0,
            2.0,
            0.90,
        ),

        "raw_t50": crossing_time(
            raw_mean,
            0.0,
            2.0,
            0.50,
        ),

        "raw_t90": crossing_time(
            raw_mean,
            0.0,
            2.0,
            0.90,
        ),

        "matched_t50": crossing_time(
            matched_mean,
            0.0,
            2.0,
            0.50,
        ),

        "matched_t90": crossing_time(
            matched_mean,
            0.0,
            2.0,
            0.90,
        ),
    }


def print_stationary(
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
        "Adaptive Huber Bandwidth-Normalized Stationary Validation"
    )

    print(
        f"Distribution: {distribution_name}"
    )

    print(
        "Oracle Huber scale: 1.0"
    )

    print()

    print(
        "  Tuning"
        "    Beta"
        "       A(c)"
        "  MatchAlpha"
        " TheoryRatio"
        "    IIRBias"
        "    IIRRMSE"
        "    RawBias"
        "    RawRMSE"
        "  MatchBias"
        "  MatchRMSE"
        " Match/IIR"
    )

    print(
        "-" * 134
    )

    for row in rows:

        observed_ratio = (
            row[
                "matched_rmse"
            ]
            / row[
                "iir_rmse"
            ]
        )

        print(
            f"{row['tuning']:8.3f}"
            f"{row['beta']:8.3f}"
            f"{row['A']:11.6f}"
            f"{row['matched_alpha']:12.6f}"
            f"{row['theoretical_ratio']:12.6f}"
            f"{row['iir_bias']:11.6f}"
            f"{row['iir_rmse']:12.6f}"
            f"{row['raw_bias']:11.6f}"
            f"{row['raw_rmse']:12.6f}"
            f"{row['matched_bias']:11.6f}"
            f"{row['matched_rmse']:12.6f}"
            f"{observed_ratio:11.4f}"
        )


def print_shift(
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
        "Adaptive Huber Bandwidth-Normalized Shift Validation"
    )

    print(
        f"Distribution: {distribution_name}"
    )

    print(
        "Location shift: 0 -> +2"
    )

    print(
        "Oracle Huber scale: 1.0"
    )

    print()

    print(
        "  Tuning"
        "    Beta"
        " MatchAlpha"
        "  IIR-T50"
        "  IIR-T90"
        "  Raw-T50"
        "  Raw-T90"
        "Match-T50"
        "Match-T90"
    )

    print(
        "-" * 91
    )

    for row in rows:

        print(
            f"{row['tuning']:8.3f}"
            f"{row['beta']:8.3f}"
            f"{row['matched_alpha']:12.6f}"
            f"{row['iir_t50']:9.0f}"
            f"{row['iir_t90']:9.0f}"
            f"{row['raw_t50']:9.0f}"
            f"{row['raw_t90']:9.0f}"
            f"{row['matched_t50']:10.0f}"
            f"{row['matched_t90']:10.0f}"
        )


def main() -> None:

    configurations = [
        (
            0.750,
            0.010,
        ),
        (
            1.345,
            0.010,
        ),
        (
            2.000,
            0.010,
        ),
        (
            1.345,
            0.020,
        ),
    ]

    distributions = [
        (
            "Normal(0,1)",
            False,
        ),
        (
            "99% N(0,1) + 1% N(0,20^2)",
            True,
        ),
    ]

    seed_offset = 0

    for (
        distribution_name,
        contaminated,
    ) in distributions:

        stationary_rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        shift_rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        for (
            tuning,
            beta,
        ) in configurations:

            stationary_rows.append(
                validate_stationary(
                    tuning,
                    beta,
                    contaminated,
                    seed_offset,
                )
            )

            seed_offset += 1

            shift_rows.append(
                validate_shift(
                    tuning,
                    beta,
                    contaminated,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_stationary(
            distribution_name,
            stationary_rows,
        )

        print_shift(
            distribution_name,
            shift_rows,
        )


if __name__ == "__main__":
    main()
