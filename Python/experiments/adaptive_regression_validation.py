"""
Exponentially weighted linear-regression validation.

The candidate simple linear regression of Y on X is derived directly from
the already validated exponentially weighted covariance state.

For a common normalized weighted population:

    beta =
        covariance / variance_x

    intercept =
        mean_y - beta * mean_x

    r_squared =
        correlation^2

The finite-weight central-moment correction factor:

    1 - weight_square_sum

cancels exactly from beta because covariance and variance_x share the same
factor.

Therefore regression introduces no new recursive state.

This experiment validates:

1. exact identity with explicit weighted least squares under variable alpha;
2. affine transformation properties;
3. stationary estimation under a linear Gaussian model;
4. dynamic response to a slope regime change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    isnan,
    sqrt,
)

import numpy as np


STATIONARY_TRIALS = 100
STATIONARY_LENGTH = 12_000
STATIONARY_BURN_IN = 7_000

SHIFT_TRIALS = 100
SHIFT_WARMUP = 8_000
SHIFT_LENGTH = 5_000

BASE_SEED = 20260914


TRUE_MEAN_X = 2.0
TRUE_SIGMA_X = 2.0

TRUE_INTERCEPT = -1.0
TRUE_BETA = 1.50

NOISE_SIGMA = 1.0


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
    def beta_y_on_x(
        self,
    ) -> float:

        if self.variance_x <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / self.variance_x
        )

    @property
    def intercept_y_on_x(
        self,
    ) -> float:

        beta = (
            self.beta_y_on_x
        )

        if (
            isnan(
                beta
            )
            or isnan(
                self.mean_x
            )
            or isnan(
                self.mean_y
            )
        ):

            return float(
                "nan"
            )

        return (
            self.mean_y
            - beta
            * self.mean_x
        )

    @property
    def correlation(
        self,
    ) -> float:

        if (
            self.variance_x <= 0.0
            or self.variance_y <= 0.0
        ):

            return float(
                "nan"
            )

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
    def r_squared(
        self,
    ) -> float:

        rho = (
            self.correlation
        )

        if isnan(
            rho
        ):

            return float(
                "nan"
            )

        return (
            rho
            * rho
        )


def explicit_weighted_regression(
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

    weights = np.array(
        [
            1.0
        ],
        dtype=float,
    )

    for alpha in alphas[
        1:
    ]:

        coefficient = clamp(
            alpha,
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

    x_array = np.asarray(
        x_values,
        dtype=float,
    )

    y_array = np.asarray(
        y_values,
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

    if variance_x <= 0.0:

        beta = float(
            "nan"
        )

        intercept = float(
            "nan"
        )

    else:

        beta = (
            covariance
            / variance_x
        )

        intercept = (
            mean_y
            - beta
            * mean_x
        )

    if (
        variance_x <= 0.0
        or variance_y <= 0.0
    ):

        r_squared = float(
            "nan"
        )

    else:

        correlation = (
            covariance
            / sqrt(
                variance_x
                * variance_y
            )
        )

        r_squared = (
            correlation
            * correlation
        )

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": variance_x,
        "variance_y": variance_y,
        "covariance": covariance,
        "beta": beta,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def assert_exact_weighted_identity() -> None:

    x_values = [
        2.0,
        5.0,
        -1.0,
        8.0,
        4.0,
        10.0,
    ]

    y_values = [
        3.0,
        8.0,
        -2.0,
        14.0,
        5.0,
        17.0,
    ]

    alphas = [
        0.0,
        0.10,
        0.25,
        0.00,
        0.70,
        0.80,
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
        explicit_weighted_regression(
            x_values,
            y_values,
            alphas,
        )
    )

    comparisons = {
        "beta": (
            state.beta_y_on_x
        ),

        "intercept": (
            state.intercept_y_on_x
        ),

        "r_squared": (
            state.r_squared
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
                f"Exact weighted regression identity failed for {name}: "
                f"{recursive_value} != {explicit[name]}"
            )


def assert_affine_properties() -> None:

    sample = [
        (
            1.0,
            4.0,
        ),
        (
            2.0,
            7.0,
        ),
        (
            5.0,
            13.0,
        ),
        (
            -2.0,
            -1.0,
        ),
    ]

    # --------------------------------------------------------------
    # TRANSLATING Y CHANGES INTERCEPT ONLY
    # --------------------------------------------------------------

    original = (
        EWCovarianceState()
    )

    translated_y = (
        EWCovarianceState()
    )

    offset_y = 100.0

    for (
        x,
        y,
    ) in sample:

        original.update(
            x,
            y,
            0.20,
        )

        translated_y.update(
            x,
            y + offset_y,
            0.20,
        )

    if not np.isclose(
        translated_y.beta_y_on_x,
        original.beta_y_on_x,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Y translation changed regression beta."
        )

    if not np.isclose(
        translated_y.intercept_y_on_x,
        original.intercept_y_on_x
        + offset_y,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Y translation produced incorrect intercept."
        )

    # --------------------------------------------------------------
    # TRANSLATING X CHANGES INTERCEPT BY -BETA * OFFSET
    # --------------------------------------------------------------

    translated_x = (
        EWCovarianceState()
    )

    offset_x = 50.0

    for (
        x,
        y,
    ) in sample:

        translated_x.update(
            x + offset_x,
            y,
            0.20,
        )

    if not np.isclose(
        translated_x.beta_y_on_x,
        original.beta_y_on_x,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "X translation changed regression beta."
        )

    expected_intercept = (
        original.intercept_y_on_x
        - original.beta_y_on_x
        * offset_x
    )

    if not np.isclose(
        translated_x.intercept_y_on_x,
        expected_intercept,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "X translation produced incorrect intercept."
        )

    # --------------------------------------------------------------
    # POSITIVE RESCALING
    #
    # X' = ax X
    # Y' = ay Y
    #
    # beta' = (ay / ax) beta
    # --------------------------------------------------------------

    scaled = (
        EWCovarianceState()
    )

    factor_x = 3.0
    factor_y = 5.0

    for (
        x,
        y,
    ) in sample:

        scaled.update(
            factor_x
            * x,
            factor_y
            * y,
            0.20,
        )

    expected_beta = (
        factor_y
        / factor_x
        * original.beta_y_on_x
    )

    expected_intercept = (
        factor_y
        * original.intercept_y_on_x
    )

    if not np.isclose(
        scaled.beta_y_on_x,
        expected_beta,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Regression scale equivariance failed for beta."
        )

    if not np.isclose(
        scaled.intercept_y_on_x,
        expected_intercept,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Regression scale equivariance failed for intercept."
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


def true_r_squared(
    beta: float,
) -> float:

    explained_variance = (
        beta
        * beta
        * TRUE_SIGMA_X
        * TRUE_SIGMA_X
    )

    total_variance = (
        explained_variance
        + NOISE_SIGMA
        * NOISE_SIGMA
    )

    return (
        explained_variance
        / total_variance
    )


def validate_stationary(
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    beta_metrics = (
        Metrics()
    )

    intercept_metrics = (
        Metrics()
    )

    r_squared_metrics = (
        Metrics()
    )

    target_r_squared = (
        true_r_squared(
            TRUE_BETA
        )
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        STATIONARY_TRIALS
    ):

        state = (
            EWCovarianceState()
        )

        x_sample = rng.normal(
            loc=TRUE_MEAN_X,
            scale=TRUE_SIGMA_X,
            size=STATIONARY_LENGTH,
        )

        noise = rng.normal(
            loc=0.0,
            scale=NOISE_SIGMA,
            size=STATIONARY_LENGTH,
        )

        y_sample = (
            TRUE_INTERCEPT
            + TRUE_BETA
            * x_sample
            + noise
        )

        for index in range(
            STATIONARY_LENGTH
        ):

            state.update(
                float(
                    x_sample[
                        index
                    ]
                ),
                float(
                    y_sample[
                        index
                    ]
                ),
                alpha,
            )

            if (
                index >= STATIONARY_BURN_IN
                and not isnan(
                    state.beta_y_on_x
                )
            ):

                beta_metrics.add(
                    state.beta_y_on_x,
                    TRUE_BETA,
                )

                intercept_metrics.add(
                    state.intercept_y_on_x,
                    TRUE_INTERCEPT,
                )

                r_squared_metrics.add(
                    state.r_squared,
                    target_r_squared,
                )

    return {
        "alpha": alpha,

        "beta_bias": (
            beta_metrics.bias
        ),

        "beta_mae": (
            beta_metrics.mae
        ),

        "beta_rmse": (
            beta_metrics.rmse
        ),

        "intercept_bias": (
            intercept_metrics.bias
        ),

        "intercept_rmse": (
            intercept_metrics.rmse
        ),

        "r2_target": (
            target_r_squared
        ),

        "r2_bias": (
            r_squared_metrics.bias
        ),

        "r2_rmse": (
            r_squared_metrics.rmse
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
        indices[
            0
        ]
        + 1
    )


def validate_beta_shift(
    alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    initial_beta = 1.50
    final_beta = -0.50

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    beta_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    intercept_paths = np.empty(
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
            EWCovarianceState()
        )

        warmup_x = rng.normal(
            loc=TRUE_MEAN_X,
            scale=TRUE_SIGMA_X,
            size=SHIFT_WARMUP,
        )

        warmup_noise = rng.normal(
            loc=0.0,
            scale=NOISE_SIGMA,
            size=SHIFT_WARMUP,
        )

        warmup_y = (
            TRUE_INTERCEPT
            + initial_beta
            * warmup_x
            + warmup_noise
        )

        for index in range(
            SHIFT_WARMUP
        ):

            state.update(
                float(
                    warmup_x[
                        index
                    ]
                ),
                float(
                    warmup_y[
                        index
                    ]
                ),
                alpha,
            )

        post_x = rng.normal(
            loc=TRUE_MEAN_X,
            scale=TRUE_SIGMA_X,
            size=SHIFT_LENGTH,
        )

        post_noise = rng.normal(
            loc=0.0,
            scale=NOISE_SIGMA,
            size=SHIFT_LENGTH,
        )

        post_y = (
            TRUE_INTERCEPT
            + final_beta
            * post_x
            + post_noise
        )

        for index in range(
            SHIFT_LENGTH
        ):

            state.update(
                float(
                    post_x[
                        index
                    ]
                ),
                float(
                    post_y[
                        index
                    ]
                ),
                alpha,
            )

            beta_paths[
                trial,
                index,
            ] = (
                state.beta_y_on_x
            )

            intercept_paths[
                trial,
                index,
            ] = (
                state.intercept_y_on_x
            )

    beta_mean = np.mean(
        beta_paths,
        axis=0,
    )

    late_beta = (
        beta_paths[
            :,
            -1500:
        ]
    )

    late_intercept = (
        intercept_paths[
            :,
            -1500:
        ]
    )

    return {
        "alpha": alpha,

        "beta_t50": crossing_time(
            beta_mean,
            initial_beta,
            final_beta,
            0.50,
        ),

        "beta_t90": crossing_time(
            beta_mean,
            initial_beta,
            final_beta,
            0.90,
        ),

        "late_beta_bias": float(
            np.mean(
                late_beta
                - final_beta
            )
        ),

        "late_beta_rmse": float(
            sqrt(
                np.mean(
                    (
                        late_beta
                        - final_beta
                    )
                    ** 2
                )
            )
        ),

        "late_intercept_bias": float(
            np.mean(
                late_intercept
                - TRUE_INTERCEPT
            )
        ),

        "late_intercept_rmse": float(
            sqrt(
                np.mean(
                    (
                        late_intercept
                        - TRUE_INTERCEPT
                    )
                    ** 2
                )
            )
        ),
    }


def print_stationary(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Exponentially Weighted Linear Regression Stationary Validation"
    )

    print()

    print(
        f"True beta: {TRUE_BETA:.6f}"
    )

    print(
        f"True intercept: {TRUE_INTERCEPT:.6f}"
    )

    print()

    print(
        "   Alpha"
        "    BetaBias"
        "     BetaMAE"
        "    BetaRMSE"
        "    IntBias"
        "    IntRMSE"
        "    R2Target"
        "      R2Bias"
        "      R2RMSE"
    )

    print(
        "-" * 108
    )

    for row in rows:

        print(
            f"{row['alpha']:8.3f}"
            f"{row['beta_bias']:12.6f}"
            f"{row['beta_mae']:12.6f}"
            f"{row['beta_rmse']:12.6f}"
            f"{row['intercept_bias']:11.6f}"
            f"{row['intercept_rmse']:11.6f}"
            f"{row['r2_target']:12.6f}"
            f"{row['r2_bias']:12.6f}"
            f"{row['r2_rmse']:12.6f}"
        )


def print_shift(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Exponentially Weighted Linear Regression Beta Shift"
    )

    print(
        "beta: +1.50 -> -0.50"
    )

    print()

    print(
        "   Alpha"
        "   BetaT50"
        "   BetaT90"
        "  LateBetaBias"
        "  LateBetaRMSE"
        "   LateIntBias"
        "   LateIntRMSE"
    )

    print(
        "-" * 88
    )

    for row in rows:

        print(
            f"{row['alpha']:8.3f}"
            f"{row['beta_t50']:10.0f}"
            f"{row['beta_t90']:10.0f}"
            f"{row['late_beta_bias']:14.6f}"
            f"{row['late_beta_rmse']:14.6f}"
            f"{row['late_intercept_bias']:14.6f}"
            f"{row['late_intercept_rmse']:14.6f}"
        )


def main() -> None:

    assert_exact_weighted_identity()
    assert_affine_properties()

    alphas = [
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    stationary_rows = []
    shift_rows = []

    seed_offset = 0

    for alpha in alphas:

        stationary_rows.append(
            validate_stationary(
                alpha,
                seed_offset,
            )
        )

        seed_offset += 1

        shift_rows.append(
            validate_beta_shift(
                alpha,
                seed_offset,
            )
        )

        seed_offset += 1

    print_stationary(
        stationary_rows
    )

    print_shift(
        shift_rows
    )


if __name__ == "__main__":
    main()
