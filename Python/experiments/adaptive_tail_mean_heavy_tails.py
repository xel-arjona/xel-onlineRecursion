"""
Exponentially Weighted Conditional Tail Mean
Heavy-Tail Stationary Validation.

This experiment stress-tests the selected O(1) tail-mean candidate under:

    - Normal(0,1)
    - Student-t(df=5), unit variance
    - Student-t(df=3), unit variance
    - Contaminated Gaussian, unit variance

The contaminated Gaussian is:

    99% N(0, 1)
     1% N(0, 10^2)

rescaled to unit variance.

Two threshold cases are evaluated separately:

1. ORACLE THRESHOLD
   The true population quantile is supplied.

2. ADAPTIVE THRESHOLD
   The threshold is supplied by the onlineRecursion adaptive-quantile
   recursion and the tail estimator uses the PREVIOUS quantile state.

The exponentially weighted conditional tail state is:

    weighted_tail[t]
        = (1-alpha) * weighted_tail[t-1]
          + alpha * I[t] * X[t]

    tail_mass[t]
        = (1-alpha) * tail_mass[t-1]
          + alpha * I[t]

    tail_mean[t]
        = weighted_tail[t] / tail_mass[t]

where for a lower tail:

    I[t] = 1 when X[t] <= threshold
           0 otherwise

The configurations deliberately scale tail alpha approximately with
tail probability:

    p=.05 -> tail_alpha=.005
    p=.01 -> tail_alpha=.001

which produces approximately comparable effective gain per tail event.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    isfinite,
    isnan,
    sqrt,
)

import numpy as np

from scipy.optimize import brentq
from scipy.stats import (
    norm,
    t,
)


TRIALS = 50
LENGTH = 16_000
BURN_IN = 9_000

BASE_SEED = 20260907

MIN_FLOAT = 1e-16

CONTAMINATION_WEIGHT = 0.01
CONTAMINATION_SIGMA = 10.0

CONTAMINATED_VARIANCE = (
    (
        1.0
        - CONTAMINATION_WEIGHT
    )
    * 1.0
    + CONTAMINATION_WEIGHT
    * (
        CONTAMINATION_SIGMA
        * CONTAMINATION_SIGMA
    )
)

CONTAMINATED_SCALE = sqrt(
    CONTAMINATED_VARIANCE
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

    def reset(
        self,
    ) -> "ConditionalTailMeanState":

        self.value = float(
            "nan"
        )

        self.weighted_tail = 0.0
        self.tail_mass = 0.0

        return self

    def update_lower(
        self,
        input_value: float,
        threshold: float,
        alpha: float,
    ) -> "ConditionalTailMeanState":

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

        indicator = (
            1.0
            if input_value <= threshold
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
    ) -> "CoupledState":

        if isnan(
            self.quantile.value
        ):

            self.quantile.update(
                input_value,
                probability,
                quantile_alpha,
            )

            self.tail.update_lower(
                input_value,
                self.quantile.value,
                tail_alpha,
            )

            return self

        previous_quantile = (
            self.quantile.value
        )

        self.tail.update_lower(
            input_value,
            previous_quantile,
            tail_alpha,
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


@dataclass(frozen=True)
class Distribution:

    name: str
    kind: str
    df: float | None = None


DISTRIBUTIONS = [
    Distribution(
        name="Normal(0,1)",
        kind="normal",
    ),
    Distribution(
        name="Student-t(df=5), unit variance",
        kind="student",
        df=5.0,
    ),
    Distribution(
        name="Student-t(df=3), unit variance",
        kind="student",
        df=3.0,
    ),
    Distribution(
        name="Contaminated Gaussian, unit variance",
        kind="contaminated",
    ),
]


def contaminated_component_sigmas(
) -> tuple[
    float,
    float,
]:

    normal_sigma = (
        1.0
        / CONTAMINATED_SCALE
    )

    contamination_sigma = (
        CONTAMINATION_SIGMA
        / CONTAMINATED_SCALE
    )

    return (
        normal_sigma,
        contamination_sigma,
    )


def contaminated_cdf(
    value: float,
) -> float:

    (
        normal_sigma,
        contamination_sigma,
    ) = (
        contaminated_component_sigmas()
    )

    return (
        (
            1.0
            - CONTAMINATION_WEIGHT
        )
        * norm.cdf(
            value
            / normal_sigma
        )
        + CONTAMINATION_WEIGHT
        * norm.cdf(
            value
            / contamination_sigma
        )
    )


def contaminated_quantile(
    probability: float,
) -> float:

    return float(
        brentq(
            lambda value: (
                contaminated_cdf(
                    value
                )
                - probability
            ),
            -50.0,
            50.0,
        )
    )


def contaminated_lower_tail_mean(
    probability: float,
    quantile: float,
) -> float:

    (
        normal_sigma,
        contamination_sigma,
    ) = (
        contaminated_component_sigmas()
    )

    normal_partial_mean = (
        -normal_sigma
        * norm.pdf(
            quantile
            / normal_sigma
        )
    )

    contamination_partial_mean = (
        -contamination_sigma
        * norm.pdf(
            quantile
            / contamination_sigma
        )
    )

    lower_partial_mean = (
        (
            1.0
            - CONTAMINATION_WEIGHT
        )
        * normal_partial_mean
        + CONTAMINATION_WEIGHT
        * contamination_partial_mean
    )

    return (
        lower_partial_mean
        / probability
    )


def distribution_targets(
    distribution: Distribution,
    probability: float,
) -> tuple[
    float,
    float,
]:

    if distribution.kind == "normal":

        quantile = float(
            norm.ppf(
                probability
            )
        )

        tail_mean = (
            -norm.pdf(
                quantile
            )
            / probability
        )

        return (
            quantile,
            float(
                tail_mean
            ),
        )

    if distribution.kind == "student":

        if distribution.df is None:
            raise ValueError(
                "Student-t distribution requires df."
            )

        df = (
            distribution.df
        )

        unit_variance_scale = sqrt(
            (
                df
                - 2.0
            )
            / df
        )

        raw_quantile = float(
            t.ppf(
                probability,
                df,
            )
        )

        raw_pdf = float(
            t.pdf(
                raw_quantile,
                df,
            )
        )

        raw_tail_mean = (
            -(
                df
                + raw_quantile
                * raw_quantile
            )
            / (
                df
                - 1.0
            )
            * raw_pdf
            / probability
        )

        return (
            unit_variance_scale
            * raw_quantile,
            unit_variance_scale
            * raw_tail_mean,
        )

    if distribution.kind == "contaminated":

        quantile = (
            contaminated_quantile(
                probability
            )
        )

        tail_mean = (
            contaminated_lower_tail_mean(
                probability,
                quantile,
            )
        )

        return (
            quantile,
            tail_mean,
        )

    raise ValueError(
        "Unknown distribution kind."
    )


def generate_sample(
    rng: np.random.Generator,
    distribution: Distribution,
) -> np.ndarray:

    if distribution.kind == "normal":

        return rng.normal(
            loc=0.0,
            scale=1.0,
            size=LENGTH,
        )

    if distribution.kind == "student":

        if distribution.df is None:
            raise ValueError(
                "Student-t distribution requires df."
            )

        df = (
            distribution.df
        )

        unit_variance_scale = sqrt(
            (
                df
                - 2.0
            )
            / df
        )

        return (
            rng.standard_t(
                df,
                size=LENGTH,
            )
            * unit_variance_scale
        )

    if distribution.kind == "contaminated":

        contamination = (
            rng.random(
                LENGTH
            )
            < CONTAMINATION_WEIGHT
        )

        sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=LENGTH,
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
                loc=0.0,
                scale=CONTAMINATION_SIGMA,
                size=contamination_count,
            )

        return (
            sample
            / CONTAMINATED_SCALE
        )

    raise ValueError(
        "Unknown distribution kind."
    )


def validate_configuration(
    distribution: Distribution,
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
    ) = distribution_targets(
        distribution,
        probability,
    )

    oracle_tail_metrics = (
        Metrics()
    )

    adaptive_quantile_metrics = (
        Metrics()
    )

    adaptive_tail_metrics = (
        Metrics()
    )

    oracle_mass_sum = 0.0
    adaptive_mass_sum = 0.0

    mass_count = 0

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        TRIALS
    ):

        oracle_tail = (
            ConditionalTailMeanState()
        )

        adaptive = (
            CoupledState.new()
        )

        sample = generate_sample(
            rng,
            distribution,
        )

        for index, raw_observation in enumerate(
            sample
        ):

            observation = float(
                raw_observation
            )

            oracle_tail.update_lower(
                observation,
                quantile_target,
                tail_alpha,
            )

            adaptive.update(
                observation,
                probability,
                quantile_alpha,
                tail_alpha,
            )

            if (
                index >= BURN_IN
                and not isnan(
                    oracle_tail.value
                )
                and not isnan(
                    adaptive.tail.value
                )
            ):

                oracle_tail_metrics.add(
                    oracle_tail.value,
                    tail_target,
                )

                adaptive_quantile_metrics.add(
                    adaptive.quantile.value,
                    quantile_target,
                )

                adaptive_tail_metrics.add(
                    adaptive.tail.value,
                    tail_target,
                )

                oracle_mass_sum += (
                    oracle_tail.tail_mass
                )

                adaptive_mass_sum += (
                    adaptive.tail.tail_mass
                )

                mass_count += 1

    return {
        "probability": probability,

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

        "quantile_rmse": (
            adaptive_quantile_metrics.rmse
        ),

        "oracle_bias": (
            oracle_tail_metrics.bias
        ),

        "oracle_mae": (
            oracle_tail_metrics.mae
        ),

        "oracle_rmse": (
            oracle_tail_metrics.rmse
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

        "oracle_mass": (
            oracle_mass_sum
            / mass_count
        ),

        "adaptive_mass": (
            adaptive_mass_sum
            / mass_count
        ),
    }


def print_results(
    distribution: Distribution,
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
        "Conditional Tail Mean Heavy-Tail Validation"
    )

    print(
        f"Distribution: {distribution.name}"
    )

    print(
        f"Lower-tail probability: {probability:.2f}"
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
        "  QAlpha"
        "  TAlpha"
        "     QTarget"
        "    TailTarget"
        "       QBias"
        "      QRMSE"
        "   OracleBias"
        "    OracleMAE"
        "   OracleRMSE"
        " AdaptiveBias"
        "  AdaptiveMAE"
        " AdaptiveRMSE"
        " OracleMass"
        " AdaptMass"
    )

    print(
        "-" * 166
    )

    for row in rows:

        print(
            f"{row['quantile_alpha']:8.3f}"
            f"{row['tail_alpha']:8.3f}"
            f"{row['quantile_target']:12.6f}"
            f"{row['tail_target']:14.6f}"
            f"{row['quantile_bias']:12.6f}"
            f"{row['quantile_rmse']:12.6f}"
            f"{row['oracle_bias']:13.6f}"
            f"{row['oracle_mae']:13.6f}"
            f"{row['oracle_rmse']:13.6f}"
            f"{row['adaptive_bias']:13.6f}"
            f"{row['adaptive_mae']:13.6f}"
            f"{row['adaptive_rmse']:13.6f}"
            f"{row['oracle_mass']:11.6f}"
            f"{row['adaptive_mass']:11.6f}"
        )


def main() -> None:

    configurations = {
        0.05: [
            (
                0.02,
                0.005,
            ),
            (
                0.05,
                0.005,
            ),
        ],

        0.01: [
            (
                0.02,
                0.001,
            ),
            (
                0.05,
                0.001,
            ),
        ],
    }

    seed_offset = 0

    for distribution in DISTRIBUTIONS:

        for (
            probability,
            probability_configurations,
        ) in configurations.items():

            rows: list[
                dict[
                    str,
                    float,
                ]
            ] = []

            for (
                quantile_alpha,
                tail_alpha,
            ) in probability_configurations:

                rows.append(
                    validate_configuration(
                        distribution,
                        probability,
                        quantile_alpha,
                        tail_alpha,
                        seed_offset,
                    )
                )

                seed_offset += 1

            print_results(
                distribution,
                probability,
                rows,
            )


if __name__ == "__main__":
    main()
