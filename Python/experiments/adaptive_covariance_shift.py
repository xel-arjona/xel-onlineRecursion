"""
Exponentially Weighted Covariance / Correlation regime-shift validation.

The underlying recursive state has already been proven exactly equivalent to
explicit weighted recomputation, including variable-alpha sequences.

This experiment only characterizes dynamic statistical behavior under two
structural changes:

1. CORRELATION REVERSAL

       rho:
           +0.70 -> -0.70

   Marginal means and variances remain unchanged.

2. SCALE EXPANSION

       sigma_y:
           3.0 -> 6.0

       rho:
           remains +0.70

   Therefore:

       covariance:
           4.2 -> 8.4

       correlation:
           remains 0.70

The experiment reports ensemble-mean T50/T90 together with late bias.

Corrected covariance uses:

    covariance / (1 - weight_square_sum)

Correlation uses the raw central moments directly because the common
finite-weight attenuation cancels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    isnan,
    sqrt,
)

import numpy as np


TRIALS = 100

WARMUP = 8_000
POST_SHIFT = 6_000

BASE_SEED = 20260913


MEAN_X = 2.0
MEAN_Y = -3.0

SIGMA_X = 2.0
SIGMA_Y_INITIAL = 3.0

RHO_INITIAL = 0.70


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
class EWCovarianceState:

    mean_x: float = float(
        "nan"
    )

    mean_y: float = float(
        "nan"
    )

    variance_x: float = 0.0
    variance_y: float = 0.0

    covariance: float = 0.0

    weight_square_sum: float = 1.0

    def update(
        self,
        x: float,
        y: float,
        alpha: float,
    ) -> "EWCovarianceState":

        coefficient = clamp(
            alpha,
            0.0,
            1.0,
        )

        if (
            isnan(
                self.mean_x
            )
            or isnan(
                self.mean_y
            )
        ):

            self.mean_x = x
            self.mean_y = y

            self.variance_x = 0.0
            self.variance_y = 0.0

            self.covariance = 0.0

            self.weight_square_sum = 1.0

            return self

        dx = (
            x
            - self.mean_x
        )

        dy = (
            y
            - self.mean_y
        )

        retention = (
            1.0
            - coefficient
        )

        self.mean_x += (
            coefficient
            * dx
        )

        self.mean_y += (
            coefficient
            * dy
        )

        self.variance_x = (
            retention
            * (
                self.variance_x
                + coefficient
                * dx
                * dx
            )
        )

        self.variance_y = (
            retention
            * (
                self.variance_y
                + coefficient
                * dy
                * dy
            )
        )

        self.covariance = (
            retention
            * (
                self.covariance
                + coefficient
                * dx
                * dy
            )
        )

        self.weight_square_sum = (
            retention
            * retention
            * self.weight_square_sum
            + coefficient
            * coefficient
        )

        return self

    @property
    def correction_denominator(
        self,
    ) -> float:

        return (
            1.0
            - self.weight_square_sum
        )

    @property
    def corrected_covariance(
        self,
    ) -> float:

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / denominator
        )

    @property
    def corrected_variance_x(
        self,
    ) -> float:

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.variance_x
            / denominator
        )

    @property
    def corrected_variance_y(
        self,
    ) -> float:

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.variance_y
            / denominator
        )

    @property
    def correlation(
        self,
    ) -> float:

        denominator = sqrt(
            self.variance_x
            * self.variance_y
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / denominator
        )


def covariance_matrix(
    sigma_y: float,
    correlation: float,
) -> np.ndarray:

    covariance = (
        correlation
        * SIGMA_X
        * sigma_y
    )

    return np.array(
        [
            [
                SIGMA_X
                * SIGMA_X,
                covariance,
            ],
            [
                covariance,
                sigma_y
                * sigma_y,
            ],
        ],
        dtype=float,
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

    if indices.size == 0:

        return float(
            "nan"
        )

    return float(
        indices[0]
        + 1
    )


def validate_correlation_reversal(
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    final_rho = -0.70

    initial_covariance = (
        RHO_INITIAL
        * SIGMA_X
        * SIGMA_Y_INITIAL
    )

    final_covariance = (
        final_rho
        * SIGMA_X
        * SIGMA_Y_INITIAL
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    covariance_paths = np.empty(
        (
            TRIALS,
            POST_SHIFT,
        ),
        dtype=float,
    )

    correlation_paths = np.empty(
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
            EWCovarianceState()
        )

        warmup = rng.multivariate_normal(
            mean=[
                MEAN_X,
                MEAN_Y,
            ],
            cov=covariance_matrix(
                SIGMA_Y_INITIAL,
                RHO_INITIAL,
            ),
            size=WARMUP,
        )

        for observation in warmup:

            state.update(
                float(
                    observation[
                        0
                    ]
                ),
                float(
                    observation[
                        1
                    ]
                ),
                alpha,
            )

        post = rng.multivariate_normal(
            mean=[
                MEAN_X,
                MEAN_Y,
            ],
            cov=covariance_matrix(
                SIGMA_Y_INITIAL,
                final_rho,
            ),
            size=POST_SHIFT,
        )

        for index, observation in enumerate(
            post
        ):

            state.update(
                float(
                    observation[
                        0
                    ]
                ),
                float(
                    observation[
                        1
                    ]
                ),
                alpha,
            )

            covariance_paths[
                trial,
                index,
            ] = (
                state.corrected_covariance
            )

            correlation_paths[
                trial,
                index,
            ] = (
                state.correlation
            )

    covariance_mean = np.mean(
        covariance_paths,
        axis=0,
    )

    correlation_mean = np.mean(
        correlation_paths,
        axis=0,
    )

    late_covariance = (
        covariance_paths[
            :,
            -2000:
        ]
    )

    late_correlation = (
        correlation_paths[
            :,
            -2000:
        ]
    )

    return {
        "alpha": alpha,

        "cov_t50": crossing_time(
            covariance_mean,
            initial_covariance,
            final_covariance,
            0.50,
        ),

        "cov_t90": crossing_time(
            covariance_mean,
            initial_covariance,
            final_covariance,
            0.90,
        ),

        "rho_t50": crossing_time(
            correlation_mean,
            RHO_INITIAL,
            final_rho,
            0.50,
        ),

        "rho_t90": crossing_time(
            correlation_mean,
            RHO_INITIAL,
            final_rho,
            0.90,
        ),

        "cov_late_bias": float(
            np.mean(
                late_covariance
                - final_covariance
            )
        ),

        "rho_late_bias": float(
            np.mean(
                late_correlation
                - final_rho
            )
        ),

        "rho_late_rmse": float(
            sqrt(
                np.mean(
                    (
                        late_correlation
                        - final_rho
                    )
                    ** 2
                )
            )
        ),
    }


def validate_scale_expansion(
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    final_sigma_y = 6.0

    initial_covariance = (
        RHO_INITIAL
        * SIGMA_X
        * SIGMA_Y_INITIAL
    )

    final_covariance = (
        RHO_INITIAL
        * SIGMA_X
        * final_sigma_y
    )

    initial_variance_y = (
        SIGMA_Y_INITIAL
        * SIGMA_Y_INITIAL
    )

    final_variance_y = (
        final_sigma_y
        * final_sigma_y
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    covariance_paths = np.empty(
        (
            TRIALS,
            POST_SHIFT,
        ),
        dtype=float,
    )

    variance_y_paths = np.empty(
        (
            TRIALS,
            POST_SHIFT,
        ),
        dtype=float,
    )

    correlation_paths = np.empty(
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
            EWCovarianceState()
        )

        warmup = rng.multivariate_normal(
            mean=[
                MEAN_X,
                MEAN_Y,
            ],
            cov=covariance_matrix(
                SIGMA_Y_INITIAL,
                RHO_INITIAL,
            ),
            size=WARMUP,
        )

        for observation in warmup:

            state.update(
                float(
                    observation[
                        0
                    ]
                ),
                float(
                    observation[
                        1
                    ]
                ),
                alpha,
            )

        post = rng.multivariate_normal(
            mean=[
                MEAN_X,
                MEAN_Y,
            ],
            cov=covariance_matrix(
                final_sigma_y,
                RHO_INITIAL,
            ),
            size=POST_SHIFT,
        )

        for index, observation in enumerate(
            post
        ):

            state.update(
                float(
                    observation[
                        0
                    ]
                ),
                float(
                    observation[
                        1
                    ]
                ),
                alpha,
            )

            covariance_paths[
                trial,
                index,
            ] = (
                state.corrected_covariance
            )

            variance_y_paths[
                trial,
                index,
            ] = (
                state.corrected_variance_y
            )

            correlation_paths[
                trial,
                index,
            ] = (
                state.correlation
            )

    covariance_mean = np.mean(
        covariance_paths,
        axis=0,
    )

    variance_y_mean = np.mean(
        variance_y_paths,
        axis=0,
    )

    correlation_mean = np.mean(
        correlation_paths,
        axis=0,
    )

    late_covariance = (
        covariance_paths[
            :,
            -2000:
        ]
    )

    late_variance_y = (
        variance_y_paths[
            :,
            -2000:
        ]
    )

    late_correlation = (
        correlation_paths[
            :,
            -2000:
        ]
    )

    return {
        "alpha": alpha,

        "cov_t50": crossing_time(
            covariance_mean,
            initial_covariance,
            final_covariance,
            0.50,
        ),

        "cov_t90": crossing_time(
            covariance_mean,
            initial_covariance,
            final_covariance,
            0.90,
        ),

        "var_y_t50": crossing_time(
            variance_y_mean,
            initial_variance_y,
            final_variance_y,
            0.50,
        ),

        "var_y_t90": crossing_time(
            variance_y_mean,
            initial_variance_y,
            final_variance_y,
            0.90,
        ),

        "cov_late_bias": float(
            np.mean(
                late_covariance
                - final_covariance
            )
        ),

        "var_y_late_bias": float(
            np.mean(
                late_variance_y
                - final_variance_y
            )
        ),

        "rho_late_bias": float(
            np.mean(
                late_correlation
                - RHO_INITIAL
            )
        ),

        "rho_max_deviation": float(
            np.max(
                np.abs(
                    correlation_mean
                    - RHO_INITIAL
                )
            )
        ),
    }


def print_correlation_reversal(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "EW Covariance / Correlation Shift Validation"
    )

    print(
        "Correlation: +0.70 -> -0.70"
    )

    print()

    print(
        "   Alpha"
        "   CovT50"
        "   CovT90"
        "   RhoT50"
        "   RhoT90"
        "  CovLateBias"
        "  RhoLateBias"
        "  RhoLateRMSE"
    )

    print(
        "-" * 94
    )

    for row in rows:

        print(
            f"{row['alpha']:8.3f}"
            f"{row['cov_t50']:9.0f}"
            f"{row['cov_t90']:9.0f}"
            f"{row['rho_t50']:9.0f}"
            f"{row['rho_t90']:9.0f}"
            f"{row['cov_late_bias']:13.6f}"
            f"{row['rho_late_bias']:13.6f}"
            f"{row['rho_late_rmse']:13.6f}"
        )


def print_scale_expansion(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "EW Covariance / Correlation Scale Validation"
    )

    print(
        "sigmaY: 3 -> 6, rho remains +0.70"
    )

    print()

    print(
        "   Alpha"
        "   CovT50"
        "   CovT90"
        "  VarYT50"
        "  VarYT90"
        "  CovLateBias"
        " VarYLateBias"
        "  RhoLateBias"
        "   RhoMaxDev"
    )

    print(
        "-" * 107
    )

    for row in rows:

        print(
            f"{row['alpha']:8.3f}"
            f"{row['cov_t50']:9.0f}"
            f"{row['cov_t90']:9.0f}"
            f"{row['var_y_t50']:9.0f}"
            f"{row['var_y_t90']:9.0f}"
            f"{row['cov_late_bias']:13.6f}"
            f"{row['var_y_late_bias']:14.6f}"
            f"{row['rho_late_bias']:13.6f}"
            f"{row['rho_max_deviation']:12.6f}"
        )


def main() -> None:

    alphas = [
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    correlation_rows: list[
        dict[
            str,
            float,
        ]
    ] = []

    scale_rows: list[
        dict[
            str,
            float,
        ]
    ] = []

    seed_offset = 0

    for alpha in alphas:

        correlation_rows.append(
            validate_correlation_reversal(
                alpha,
                seed_offset,
            )
        )

        seed_offset += 1

        scale_rows.append(
            validate_scale_expansion(
                alpha,
                seed_offset,
            )
        )

        seed_offset += 1

    print_correlation_reversal(
        correlation_rows
    )

    print_scale_expansion(
        scale_rows
    )


if __name__ == "__main__":
    main()
