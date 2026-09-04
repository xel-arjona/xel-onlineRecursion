"""
Exponentially Weighted Covariance / Correlation validation.

Candidate O(1) bivariate recursion.

For previous exponentially weighted means mx, my:

    dx = x - mx
    dy = y - my

    mx += alpha * dx
    my += alpha * dy

    covariance =
        (1-alpha)
        * (
            covariance
            + alpha * dx * dy
        )

    variance_x =
        (1-alpha)
        * (
            variance_x
            + alpha * dx * dx
        )

    variance_y =
        (1-alpha)
        * (
            variance_y
            + alpha * dy * dy
        )

The state also recursively tracks normalized-weight concentration:

    weight_square_sum =
        (1-alpha)^2 * previous_weight_square_sum
        + alpha^2

For deterministic/exogenous weights and IID observations:

    E[raw covariance]
        = (1 - weight_square_sum)
          * population covariance

so the finite-weight corrected covariance is:

    corrected_covariance =
        covariance
        / (1 - weight_square_sum)

when the denominator is positive.

The corresponding effective sample size is:

    effective_sample_size
        = 1 / weight_square_sum

Correlation needs no finite-weight correction because the common multiplicative
factor cancels between covariance and variances.

This experiment validates:

1. exact identity with explicit weighted recomputation;
2. variable-alpha weight semantics;
3. constant-alpha theoretical raw covariance bias;
4. finite-weight covariance correction;
5. correlation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    isnan,
    sqrt,
)

import numpy as np


TRIALS = 100
LENGTH = 12_000
BURN_IN = 7_000

BASE_SEED = 20260912


TRUE_MEAN_X = 2.0
TRUE_MEAN_Y = -3.0

TRUE_SIGMA_X = 2.0
TRUE_SIGMA_Y = 3.0

TRUE_CORRELATION = 0.70

TRUE_VARIANCE_X = (
    TRUE_SIGMA_X
    * TRUE_SIGMA_X
)

TRUE_VARIANCE_Y = (
    TRUE_SIGMA_Y
    * TRUE_SIGMA_Y
)

TRUE_COVARIANCE = (
    TRUE_CORRELATION
    * TRUE_SIGMA_X
    * TRUE_SIGMA_Y
)


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

    def reset(
        self,
    ) -> "EWCovarianceState":

        self.mean_x = float(
            "nan"
        )

        self.mean_y = float(
            "nan"
        )

        self.variance_x = 0.0
        self.variance_y = 0.0

        self.covariance = 0.0

        self.weight_square_sum = 1.0

        return self

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
    def effective_sample_size(
        self,
    ) -> float:

        if self.weight_square_sum <= 0.0:

            return float(
                "nan"
            )

        return (
            1.0
            / self.weight_square_sum
        )


def explicit_weighted_moments(
    x_values: list[
        float
    ],
    y_values: list[
        float
    ],
    alphas: list[
        float
    ],
) -> dict[
    str,
    float,
]:

    if (
        len(
            x_values
        )
        != len(
            y_values
        )
        or len(
            x_values
        )
        != len(
            alphas
        )
    ):
        raise ValueError(
            "Input lengths must match."
        )

    if len(
        x_values
    ) == 0:

        raise ValueError(
            "At least one observation is required."
        )

    weights = np.array(
        [
            1.0
        ],
        dtype=float,
    )

    xs = [
        x_values[
            0
        ]
    ]

    ys = [
        y_values[
            0
        ]
    ]

    for index in range(
        1,
        len(
            x_values
        ),
    ):

        coefficient = clamp(
            alphas[
                index
            ],
            0.0,
            1.0,
        )

        weights *= (
            1.0
            - coefficient
        )

        weights = np.append(
            weights,
            coefficient,
        )

        xs.append(
            x_values[
                index
            ]
        )

        ys.append(
            y_values[
                index
            ]
        )

    x_array = np.asarray(
        xs,
        dtype=float,
    )

    y_array = np.asarray(
        ys,
        dtype=float,
    )

    mean_x = float(
        np.sum(
            weights
            * x_array
        )
    )

    mean_y = float(
        np.sum(
            weights
            * y_array
        )
    )

    centered_x = (
        x_array
        - mean_x
    )

    centered_y = (
        y_array
        - mean_y
    )

    variance_x = float(
        np.sum(
            weights
            * centered_x
            * centered_x
        )
    )

    variance_y = float(
        np.sum(
            weights
            * centered_y
            * centered_y
        )
    )

    covariance = float(
        np.sum(
            weights
            * centered_x
            * centered_y
        )
    )

    weight_square_sum = float(
        np.sum(
            weights
            * weights
        )
    )

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": variance_x,
        "variance_y": variance_y,
        "covariance": covariance,
        "weight_square_sum": (
            weight_square_sum
        ),
    }


def assert_exact_weight_identity() -> None:

    x_values = [
        2.0,
        5.0,
        -1.0,
        8.0,
        4.0,
        10.0,
    ]

    y_values = [
        -3.0,
        7.0,
        2.0,
        -5.0,
        6.0,
        11.0,
    ]

    alphas = [
        # First alpha does not affect direct initialization.
        0.00,
        0.10,
        0.25,
        0.00,
        0.70,
        1.20,
    ]

    state = (
        EWCovarianceState()
    )

    for (
        x,
        y,
        alpha,
    ) in zip(
        x_values,
        y_values,
        alphas,
        strict=True,
    ):

        state.update(
            x,
            y,
            alpha,
        )

    explicit = (
        explicit_weighted_moments(
            x_values,
            y_values,
            alphas,
        )
    )

    comparisons = {
        "mean_x": state.mean_x,
        "mean_y": state.mean_y,
        "variance_x": (
            state.variance_x
        ),
        "variance_y": (
            state.variance_y
        ),
        "covariance": (
            state.covariance
        ),
        "weight_square_sum": (
            state.weight_square_sum
        ),
    }

    for (
        name,
        recursive_value,
    ) in comparisons.items():

        if not np.isclose(
            recursive_value,
            explicit[
                name
            ],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                f"Exact weighted identity failed for {name}: "
                f"{recursive_value} != {explicit[name]}"
            )


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


def validate_alpha(
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    raw_covariance_metrics = (
        Metrics()
    )

    corrected_covariance_metrics = (
        Metrics()
    )

    correlation_metrics = (
        Metrics()
    )

    raw_variance_x_metrics = (
        Metrics()
    )

    corrected_variance_x_metrics = (
        Metrics()
    )

    weight_square_sum_total = 0.0
    effective_sample_size_total = 0.0
    weight_count = 0

    covariance_matrix = np.array(
        [
            [
                TRUE_VARIANCE_X,
                TRUE_COVARIANCE,
            ],
            [
                TRUE_COVARIANCE,
                TRUE_VARIANCE_Y,
            ],
        ],
        dtype=float,
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        TRIALS
    ):

        state = (
            EWCovarianceState()
        )

        sample = rng.multivariate_normal(
            mean=[
                TRUE_MEAN_X,
                TRUE_MEAN_Y,
            ],
            cov=covariance_matrix,
            size=LENGTH,
        )

        for index, observation in enumerate(
            sample
        ):

            x = float(
                observation[
                    0
                ]
            )

            y = float(
                observation[
                    1
                ]
            )

            state.update(
                x,
                y,
                alpha,
            )

            if index >= BURN_IN:

                raw_covariance_metrics.add(
                    state.covariance,
                    TRUE_COVARIANCE,
                )

                corrected_covariance_metrics.add(
                    state.corrected_covariance,
                    TRUE_COVARIANCE,
                )

                correlation_metrics.add(
                    state.correlation,
                    TRUE_CORRELATION,
                )

                raw_variance_x_metrics.add(
                    state.variance_x,
                    TRUE_VARIANCE_X,
                )

                corrected_variance_x_metrics.add(
                    state.corrected_variance_x,
                    TRUE_VARIANCE_X,
                )

                weight_square_sum_total += (
                    state.weight_square_sum
                )

                effective_sample_size_total += (
                    state.effective_sample_size
                )

                weight_count += 1

    theoretical_weight_square_sum = (
        alpha
        / (
            2.0
            - alpha
        )
    )

    theoretical_raw_factor = (
        1.0
        - theoretical_weight_square_sum
    )

    theoretical_effective_sample_size = (
        1.0
        / theoretical_weight_square_sum
    )

    return {
        "alpha": alpha,

        "theoretical_s2": (
            theoretical_weight_square_sum
        ),

        "observed_s2": (
            weight_square_sum_total
            / weight_count
        ),

        "theoretical_raw_factor": (
            theoretical_raw_factor
        ),

        "observed_raw_factor": (
            (
                raw_covariance_metrics.bias
                + TRUE_COVARIANCE
            )
            / TRUE_COVARIANCE
        ),

        "raw_cov_bias": (
            raw_covariance_metrics.bias
        ),

        "raw_cov_rmse": (
            raw_covariance_metrics.rmse
        ),

        "corrected_cov_bias": (
            corrected_covariance_metrics.bias
        ),

        "corrected_cov_rmse": (
            corrected_covariance_metrics.rmse
        ),

        "correlation_bias": (
            correlation_metrics.bias
        ),

        "correlation_rmse": (
            correlation_metrics.rmse
        ),

        "raw_var_x_bias": (
            raw_variance_x_metrics.bias
        ),

        "corrected_var_x_bias": (
            corrected_variance_x_metrics.bias
        ),

        "theoretical_neff": (
            theoretical_effective_sample_size
        ),

        "observed_neff": (
            effective_sample_size_total
            / weight_count
        ),
    }


def print_results(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Exponentially Weighted Covariance / Correlation Validation"
    )

    print()

    print(
        f"True covariance: {TRUE_COVARIANCE:.6f}"
    )

    print(
        f"True correlation: {TRUE_CORRELATION:.6f}"
    )

    print(
        f"True variance X: {TRUE_VARIANCE_X:.6f}"
    )

    print()

    print(
        "   Alpha"
        "   TheoryS2"
        "      ObsS2"
        "  TheoryRaw"
        "     ObsRaw"
        "    RawCBias"
        "    RawCRMSE"
        "   CorrCBias"
        "   CorrCRMSE"
        "    RhoBias"
        "    RhoRMSE"
        "   RawVxBias"
        "  CorrVxBias"
        " TheoryNeff"
        "    ObsNeff"
    )

    print(
        "-" * 183
    )

    for row in rows:

        print(
            f"{row['alpha']:8.3f}"
            f"{row['theoretical_s2']:11.6f}"
            f"{row['observed_s2']:11.6f}"
            f"{row['theoretical_raw_factor']:11.6f}"
            f"{row['observed_raw_factor']:11.6f}"
            f"{row['raw_cov_bias']:12.6f}"
            f"{row['raw_cov_rmse']:12.6f}"
            f"{row['corrected_cov_bias']:12.6f}"
            f"{row['corrected_cov_rmse']:12.6f}"
            f"{row['correlation_bias']:11.6f}"
            f"{row['correlation_rmse']:11.6f}"
            f"{row['raw_var_x_bias']:12.6f}"
            f"{row['corrected_var_x_bias']:12.6f}"
            f"{row['theoretical_neff']:11.3f}"
            f"{row['observed_neff']:11.3f}"
        )


def main() -> None:

    assert_exact_weight_identity()

    alphas = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    ]

    rows: list[
        dict[
            str,
            float,
        ]
    ] = []

    for seed_offset, alpha in enumerate(
        alphas
    ):

        rows.append(
            validate_alpha(
                alpha,
                seed_offset,
            )
        )

    print_results(
        rows
    )


if __name__ == "__main__":
    main()
