"""
Adaptive Median Absolute Deviation validation.

The candidate robust scale is constructed from the already validated adaptive
quantile recursion.

For an externally supplied center m:

    deviation = abs(x - m)

and the raw adaptive MAD is the adaptive 0.5 quantile of that deviation stream.

For a symmetric continuous population whose center is its median:

    population MAD = median(abs(X - median(X)))

For a standard Normal population:

    MAD = Phi^{-1}(0.75)
        = 0.6744897501960817

and the conventional Gaussian-equivalent scale is:

    gaussian_scale
        = MAD / Phi^{-1}(0.75)
        ~= 1.482602218505602 * MAD

The experiment separates:

1. ORACLE CENTER
   Absolute deviations are formed around the true population center.

2. ADAPTIVE HUBER CENTER
   Absolute deviations use the PREVIOUS adaptive Huber location.

The adaptive Huber center itself is supplied oracle unit scale in this
experiment. This deliberately isolates center/MAD coupling before we test
a fully coupled Huber + MAD system.

All tested populations are symmetric and standardized to zero location and
unit variance where the variance exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
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

BASE_SEED = 20260910

MAD_NORMAL = float(
    norm.ppf(
        0.75
    )
)

MAD_TO_GAUSSIAN_SCALE = (
    1.0
    / MAD_NORMAL
)

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

        if tuning <= 0.0:
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
class AdaptiveMADState:

    deviation_quantile: AdaptiveQuantileState

    @classmethod
    def new(
        cls,
    ) -> "AdaptiveMADState":

        return cls(
            deviation_quantile=AdaptiveQuantileState()
        )

    @property
    def value(
        self,
    ) -> float:

        return (
            self.deviation_quantile.value
        )

    @property
    def gaussian_scale(
        self,
    ) -> float:

        if isnan(
            self.value
        ):

            return float(
                "nan"
            )

        return (
            self.value
            * MAD_TO_GAUSSIAN_SCALE
        )

    def update(
        self,
        input_value: float,
        center: float,
        alpha: float,
    ) -> "AdaptiveMADState":

        deviation = abs(
            input_value
            - center
        )

        self.deviation_quantile.update(
            deviation,
            0.5,
            alpha,
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

    return (
        1.0
        / CONTAMINATED_SCALE,
        CONTAMINATION_SIGMA
        / CONTAMINATED_SCALE,
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


def population_mad(
    distribution: Distribution,
) -> float:

    # For every symmetric zero-centered continuous population:
    #
    #     median(|X|)
    #         = Q_X(0.75)

    if distribution.kind == "normal":

        return MAD_NORMAL

    if distribution.kind == "student":

        if distribution.df is None:
            raise ValueError(
                "Student distribution requires df."
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

        return float(
            t.ppf(
                0.75,
                df,
            )
            * unit_variance_scale
        )

    if distribution.kind == "contaminated":

        return (
            contaminated_quantile(
                0.75
            )
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
                "Student distribution requires df."
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

        contaminated = (
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
                scale=CONTAMINATION_SIGMA,
                size=count,
            )

        return (
            sample
            / CONTAMINATED_SCALE
        )

    raise ValueError(
        "Unknown distribution kind."
    )


def assert_deterministic_properties() -> None:

    sample = [
        3.0,
        -7.0,
        2.0,
        20.0,
        -1.0,
        -30.0,
        4.0,
    ]

    # ------------------------------------------------------------------
    # TRANSLATION EQUIVARIANCE
    # ------------------------------------------------------------------

    original = (
        AdaptiveMADState.new()
    )

    translated = (
        AdaptiveMADState.new()
    )

    offset = 100.0

    for observation in sample:

        original.update(
            observation,
            0.0,
            0.10,
        )

        translated.update(
            observation
            + offset,
            offset,
            0.10,
        )

    if not np.isclose(
        original.value,
        translated.value,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "MAD translation equivariance failed."
        )

    # ------------------------------------------------------------------
    # REFLECTION INVARIANCE
    # ------------------------------------------------------------------

    positive = (
        AdaptiveMADState.new()
    )

    reflected = (
        AdaptiveMADState.new()
    )

    for observation in sample:

        positive.update(
            observation,
            0.0,
            0.10,
        )

        reflected.update(
            -observation,
            0.0,
            0.10,
        )

    if not np.isclose(
        positive.value,
        reflected.value,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "MAD reflection invariance failed."
        )

    # ------------------------------------------------------------------
    # POSITIVE-SCALE EQUIVARIANCE
    # ------------------------------------------------------------------

    factor = 7.0

    original = (
        AdaptiveMADState.new()
    )

    scaled = (
        AdaptiveMADState.new()
    )

    for observation in sample:

        original.update(
            observation,
            0.0,
            0.10,
        )

        scaled.update(
            factor
            * observation,
            0.0,
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
            "MAD positive-scale equivariance failed."
        )


def validate_configuration(
    distribution: Distribution,
    center_alpha: float,
    mad_alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    mad_target = population_mad(
        distribution
    )

    gaussian_scale_target = (
        mad_target
        * MAD_TO_GAUSSIAN_SCALE
    )

    oracle_metrics = Metrics()

    adaptive_center_metrics = (
        Metrics()
    )

    huber_metrics = Metrics()

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        TRIALS
    ):

        oracle_mad = (
            AdaptiveMADState.new()
        )

        adaptive_mad = (
            AdaptiveMADState.new()
        )

        huber = (
            AdaptiveHuberState()
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

            # ----------------------------------------------------------
            # ORACLE CENTER
            # ----------------------------------------------------------

            oracle_mad.update(
                observation,
                0.0,
                mad_alpha,
            )

            # ----------------------------------------------------------
            # ADAPTIVE CENTER
            #
            # The deviation must use the PREVIOUS Huber location.
            # ----------------------------------------------------------

            if isnan(
                huber.value
            ):

                huber.update(
                    observation,
                    1.0,
                    1.345,
                    center_alpha,
                )

                adaptive_mad.update(
                    observation,
                    huber.value,
                    mad_alpha,
                )

            else:

                previous_center = (
                    huber.value
                )

                adaptive_mad.update(
                    observation,
                    previous_center,
                    mad_alpha,
                )

                huber.update(
                    observation,
                    1.0,
                    1.345,
                    center_alpha,
                )

            if index >= BURN_IN:

                oracle_metrics.add(
                    oracle_mad.value,
                    mad_target,
                )

                adaptive_center_metrics.add(
                    adaptive_mad.value,
                    mad_target,
                )

                huber_metrics.add(
                    huber.value,
                    0.0,
                )

    return {
        "center_alpha": (
            center_alpha
        ),

        "mad_alpha": (
            mad_alpha
        ),

        "mad_target": (
            mad_target
        ),

        "gaussian_target": (
            gaussian_scale_target
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
            adaptive_center_metrics.bias
        ),

        "adaptive_mae": (
            adaptive_center_metrics.mae
        ),

        "adaptive_rmse": (
            adaptive_center_metrics.rmse
        ),

        "center_bias": (
            huber_metrics.bias
        ),

        "center_rmse": (
            huber_metrics.rmse
        ),
    }


def print_results(
    distribution: Distribution,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Adaptive MAD Validation"
    )

    print(
        f"Distribution: {distribution.name}"
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
        " CAlpha"
        " MAlpha"
        "   MADTarget"
        "  GaussScale"
        "  OracleBias"
        "   OracleMAE"
        "  OracleRMSE"
        "   AdaptBias"
        "    AdaptMAE"
        "   AdaptRMSE"
        "  CenterBias"
        "  CenterRMSE"
    )

    print(
        "-" * 142
    )

    for row in rows:

        print(
            f"{row['center_alpha']:7.3f}"
            f"{row['mad_alpha']:7.3f}"
            f"{row['mad_target']:12.6f}"
            f"{row['gaussian_target']:12.6f}"
            f"{row['oracle_bias']:12.6f}"
            f"{row['oracle_mae']:12.6f}"
            f"{row['oracle_rmse']:12.6f}"
            f"{row['adaptive_bias']:12.6f}"
            f"{row['adaptive_mae']:12.6f}"
            f"{row['adaptive_rmse']:12.6f}"
            f"{row['center_bias']:12.6f}"
            f"{row['center_rmse']:12.6f}"
        )


def main() -> None:

    assert_deterministic_properties()

    configurations = [
        (
            0.02,
            0.01,
        ),
        (
            0.02,
            0.02,
        ),
        (
            0.02,
            0.05,
        ),
        (
            0.05,
            0.02,
        ),
    ]

    seed_offset = 0

    for distribution in DISTRIBUTIONS:

        rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        for (
            center_alpha,
            mad_alpha,
        ) in configurations:

            rows.append(
                validate_configuration(
                    distribution,
                    center_alpha,
                    mad_alpha,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_results(
            distribution,
            rows,
        )


if __name__ == "__main__":
    main()
