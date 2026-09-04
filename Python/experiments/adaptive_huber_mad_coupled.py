"""
Fully coupled Adaptive Huber + MAD validation.

This experiment removes the remaining oracle quantity from the robust
location/scale pair.

The coupled system consists of:

    Huber location:
        innovation = x - location
        limit = tuning * robust_scale
        correction = clamp(innovation, -limit, +limit)
        location += center_alpha * correction

    Raw MAD:
        deviation = abs(x - previous_location)
        mad = adaptive_median(deviation)

    Gaussian-equivalent robust scale:
        robust_scale = MAD / Phi^{-1}(0.75)

The timing rule is essential.

For each observation after initialization:

    previous_location = huber.value
    previous_mad      = mad.value

    huber.update(
        input,
        previous_mad * MAD_TO_GAUSSIAN_SCALE,
        ...
    )

    mad.update(
        input,
        previous_location,
        ...
    )

Thus the current observation cannot inflate its own Huber clipping boundary
and cannot move its own center before its deviation is measured.

Bootstrap behavior
------------------

The first observation initializes Huber directly.

MAD then receives zero deviation because the first center equals the first
observation.

The adaptive MAD quantile initially remains zero, but its internal innovation
scale begins accumulating. Subsequent observations therefore allow MAD to
become positive, after which Huber obtains a non-zero clipping scale.

No arbitrary bootstrap scale is injected.

Validation
----------

1. Stationary:
   - Normal(0,1)
   - Student-t(df=5), unit variance
   - Student-t(df=3), unit variance
   - contaminated Gaussian, unit variance

   Compare the fully coupled Huber center with an oracle-scale Huber using the
   population Gaussian-equivalent MAD scale.

   Compare the fully coupled MAD with an oracle-center MAD.

2. Location shift:
       Normal(0,1) -> Normal(+2,1)

   Measure coupled Huber T50/T90 and compare with oracle-scale Huber.

3. Scale shifts:
       Normal(0,1) -> Normal(0,2)
       Normal(0,1) -> Normal(0,0.5)

   Measure robust-scale T50/T90 and late error while checking that the center
   remains stable.
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


STATIONARY_TRIALS = 50
STATIONARY_LENGTH = 16_000
STATIONARY_BURN_IN = 9_000

SHIFT_TRIALS = 50
SHIFT_WARMUP = 8_000
SHIFT_LENGTH = 8_000

BASE_SEED = 20260911


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
class CoupledHuberMADState:

    huber: AdaptiveHuberState
    mad: AdaptiveMADState

    @classmethod
    def new(
        cls,
    ) -> "CoupledHuberMADState":

        return cls(
            huber=AdaptiveHuberState(),
            mad=AdaptiveMADState.new(),
        )

    def update(
        self,
        input_value: float,
        tuning: float,
        center_alpha: float,
        mad_alpha: float,
    ) -> "CoupledHuberMADState":

        # --------------------------------------------------------------
        # INITIALIZATION
        #
        # Huber initializes directly from the first observation.
        # The first deviation is therefore zero.
        # --------------------------------------------------------------

        if isnan(
            self.huber.value
        ):

            self.huber.update(
                input_value,
                0.0,
                tuning,
                center_alpha,
            )

            self.mad.update(
                input_value,
                self.huber.value,
                mad_alpha,
            )

            return self

        # --------------------------------------------------------------
        # PREVIOUS-STATE TIMING
        # --------------------------------------------------------------

        previous_center = (
            self.huber.value
        )

        previous_mad = (
            self.mad.value
        )

        robust_scale = (
            0.0
            if isnan(
                previous_mad
            )
            else (
                previous_mad
                * MAD_TO_GAUSSIAN_SCALE
            )
        )

        # Current observation uses previous robust scale.
        self.huber.update(
            input_value,
            robust_scale,
            tuning,
            center_alpha,
        )

        # Current deviation uses previous center.
        self.mad.update(
            input_value,
            previous_center,
            mad_alpha,
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
    ) = contaminated_component_sigmas()

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

        return contaminated_quantile(
            0.75
        )

    raise ValueError(
        "Unknown distribution kind."
    )


def generate_stationary_sample(
    rng: np.random.Generator,
    distribution: Distribution,
) -> np.ndarray:

    if distribution.kind == "normal":

        return rng.normal(
            loc=0.0,
            scale=1.0,
            size=STATIONARY_LENGTH,
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
                size=STATIONARY_LENGTH,
            )
            * unit_variance_scale
        )

    if distribution.kind == "contaminated":

        contaminated = (
            rng.random(
                STATIONARY_LENGTH
            )
            < CONTAMINATION_WEIGHT
        )

        sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=STATIONARY_LENGTH,
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


def assert_bootstrap() -> None:

    state = (
        CoupledHuberMADState.new()
    )

    sample = [
        0.0,
        2.0,
        -2.0,
        1.0,
        -1.0,
    ]

    for observation in sample:

        state.update(
            observation,
            1.345,
            0.20,
            0.10,
        )

    if (
        isnan(
            state.mad.value
        )
        or state.mad.value <= 0.0
    ):
        raise AssertionError(
            "Coupled MAD failed to bootstrap above zero."
        )

    if (
        isnan(
            state.huber.value
        )
        or isnan(
            state.mad.gaussian_scale
        )
    ):
        raise AssertionError(
            "Coupled Huber/MAD bootstrap failed."
        )


def validate_stationary(
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

    robust_scale_target = (
        mad_target
        * MAD_TO_GAUSSIAN_SCALE
    )

    coupled_center_metrics = (
        Metrics()
    )

    oracle_scale_center_metrics = (
        Metrics()
    )

    coupled_mad_metrics = (
        Metrics()
    )

    oracle_center_mad_metrics = (
        Metrics()
    )

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    for _ in range(
        STATIONARY_TRIALS
    ):

        coupled = (
            CoupledHuberMADState.new()
        )

        oracle_huber = (
            AdaptiveHuberState()
        )

        oracle_mad = (
            AdaptiveMADState.new()
        )

        sample = generate_stationary_sample(
            rng,
            distribution,
        )

        for index, raw_observation in enumerate(
            sample
        ):

            observation = float(
                raw_observation
            )

            coupled.update(
                observation,
                1.345,
                center_alpha,
                mad_alpha,
            )

            oracle_huber.update(
                observation,
                robust_scale_target,
                1.345,
                center_alpha,
            )

            oracle_mad.update(
                observation,
                0.0,
                mad_alpha,
            )

            if index >= STATIONARY_BURN_IN:

                coupled_center_metrics.add(
                    coupled.huber.value,
                    0.0,
                )

                oracle_scale_center_metrics.add(
                    oracle_huber.value,
                    0.0,
                )

                coupled_mad_metrics.add(
                    coupled.mad.value,
                    mad_target,
                )

                oracle_center_mad_metrics.add(
                    oracle_mad.value,
                    mad_target,
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

        "scale_target": (
            robust_scale_target
        ),

        "oracle_center_rmse": (
            oracle_scale_center_metrics.rmse
        ),

        "coupled_center_bias": (
            coupled_center_metrics.bias
        ),

        "coupled_center_rmse": (
            coupled_center_metrics.rmse
        ),

        "oracle_mad_rmse": (
            oracle_center_mad_metrics.rmse
        ),

        "coupled_mad_bias": (
            coupled_mad_metrics.bias
        ),

        "coupled_mad_rmse": (
            coupled_mad_metrics.rmse
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


def validate_location_shift(
    center_alpha: float,
    mad_alpha: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    coupled_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    oracle_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    scale_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    for trial in range(
        SHIFT_TRIALS
    ):

        coupled = (
            CoupledHuberMADState.new()
        )

        oracle_huber = (
            AdaptiveHuberState()
        )

        warmup = rng.normal(
            loc=0.0,
            scale=1.0,
            size=SHIFT_WARMUP,
        )

        for raw_observation in warmup:

            observation = float(
                raw_observation
            )

            coupled.update(
                observation,
                1.345,
                center_alpha,
                mad_alpha,
            )

            oracle_huber.update(
                observation,
                1.0,
                1.345,
                center_alpha,
            )

        post = rng.normal(
            loc=2.0,
            scale=1.0,
            size=SHIFT_LENGTH,
        )

        for index, raw_observation in enumerate(
            post
        ):

            observation = float(
                raw_observation
            )

            coupled.update(
                observation,
                1.345,
                center_alpha,
                mad_alpha,
            )

            oracle_huber.update(
                observation,
                1.0,
                1.345,
                center_alpha,
            )

            coupled_paths[
                trial,
                index,
            ] = (
                coupled.huber.value
            )

            oracle_paths[
                trial,
                index,
            ] = (
                oracle_huber.value
            )

            scale_paths[
                trial,
                index,
            ] = (
                coupled.mad.gaussian_scale
            )

    coupled_mean = np.mean(
        coupled_paths,
        axis=0,
    )

    oracle_mean = np.mean(
        oracle_paths,
        axis=0,
    )

    scale_mean = np.mean(
        scale_paths,
        axis=0,
    )

    return {
        "center_alpha": center_alpha,
        "mad_alpha": mad_alpha,

        "oracle_t50": crossing_time(
            oracle_mean,
            0.0,
            2.0,
            0.50,
        ),

        "oracle_t90": crossing_time(
            oracle_mean,
            0.0,
            2.0,
            0.90,
        ),

        "coupled_t50": crossing_time(
            coupled_mean,
            0.0,
            2.0,
            0.50,
        ),

        "coupled_t90": crossing_time(
            coupled_mean,
            0.0,
            2.0,
            0.90,
        ),

        "maximum_scale": float(
            np.max(
                scale_mean
            )
        ),

        "late_scale_bias": float(
            np.mean(
                scale_mean[
                    -2000:
                ]
            )
            - 1.0
        ),
    }


def validate_scale_shift(
    center_alpha: float,
    mad_alpha: float,
    final_sigma: float,
    seed_offset: int,
) -> dict[
    str,
    float,
]:

    rng = np.random.default_rng(
        BASE_SEED
        + seed_offset
    )

    scale_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    center_paths = np.empty(
        (
            SHIFT_TRIALS,
            SHIFT_LENGTH,
        ),
        dtype=float,
    )

    for trial in range(
        SHIFT_TRIALS
    ):

        coupled = (
            CoupledHuberMADState.new()
        )

        warmup = rng.normal(
            loc=0.0,
            scale=1.0,
            size=SHIFT_WARMUP,
        )

        for raw_observation in warmup:

            coupled.update(
                float(
                    raw_observation
                ),
                1.345,
                center_alpha,
                mad_alpha,
            )

        post = rng.normal(
            loc=0.0,
            scale=final_sigma,
            size=SHIFT_LENGTH,
        )

        for index, raw_observation in enumerate(
            post
        ):

            coupled.update(
                float(
                    raw_observation
                ),
                1.345,
                center_alpha,
                mad_alpha,
            )

            scale_paths[
                trial,
                index,
            ] = (
                coupled.mad.gaussian_scale
            )

            center_paths[
                trial,
                index,
            ] = (
                coupled.huber.value
            )

    scale_mean = np.mean(
        scale_paths,
        axis=0,
    )

    center_errors = (
        center_paths
    )

    late_center = (
        center_errors[
            :,
            -2000:
        ]
    )

    late_scale = (
        scale_paths[
            :,
            -2000:
        ]
    )

    return {
        "center_alpha": center_alpha,
        "mad_alpha": mad_alpha,
        "final_sigma": final_sigma,

        "scale_t50": crossing_time(
            scale_mean,
            1.0,
            final_sigma,
            0.50,
        ),

        "scale_t90": crossing_time(
            scale_mean,
            1.0,
            final_sigma,
            0.90,
        ),

        "late_scale_bias": float(
            np.mean(
                late_scale
                - final_sigma
            )
        ),

        "late_scale_rmse": float(
            sqrt(
                np.mean(
                    (
                        late_scale
                        - final_sigma
                    )
                    ** 2
                )
            )
        ),

        "late_center_bias": float(
            np.mean(
                late_center
            )
        ),

        "late_center_rmse": float(
            sqrt(
                np.mean(
                    late_center
                    ** 2
                )
            )
        ),
    }


def print_stationary(
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
        "Fully Coupled Huber + MAD Stationary Validation"
    )

    print(
        f"Distribution: {distribution.name}"
    )

    print()

    print(
        " CAlpha"
        " MAlpha"
        "  MADTarget"
        " ScaleTarget"
        " OracleCRMSE"
        " CoupledCBias"
        " CoupledCRMSE"
        " OracleMRMSE"
        " CoupledMBias"
        " CoupledMRMSE"
    )

    print(
        "-" * 121
    )

    for row in rows:

        print(
            f"{row['center_alpha']:7.3f}"
            f"{row['mad_alpha']:7.3f}"
            f"{row['mad_target']:11.6f}"
            f"{row['scale_target']:12.6f}"
            f"{row['oracle_center_rmse']:13.6f}"
            f"{row['coupled_center_bias']:14.6f}"
            f"{row['coupled_center_rmse']:14.6f}"
            f"{row['oracle_mad_rmse']:13.6f}"
            f"{row['coupled_mad_bias']:14.6f}"
            f"{row['coupled_mad_rmse']:14.6f}"
        )


def print_location_shift(
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Fully Coupled Huber + MAD Location Shift"
    )

    print(
        "Normal(0,1) -> Normal(+2,1)"
    )

    print()

    print(
        " CAlpha"
        " MAlpha"
        " OracleT50"
        " OracleT90"
        " CoupledT50"
        " CoupledT90"
        " MaxScale"
        " LateScaleBias"
    )

    print(
        "-" * 93
    )

    for row in rows:

        print(
            f"{row['center_alpha']:7.3f}"
            f"{row['mad_alpha']:7.3f}"
            f"{row['oracle_t50']:10.0f}"
            f"{row['oracle_t90']:10.0f}"
            f"{row['coupled_t50']:11.0f}"
            f"{row['coupled_t90']:11.0f}"
            f"{row['maximum_scale']:10.6f}"
            f"{row['late_scale_bias']:14.6f}"
        )


def print_scale_shift(
    final_sigma: float,
    rows: list[
        dict[
            str,
            float,
        ]
    ],
) -> None:

    print()

    print(
        "Fully Coupled Huber + MAD Scale Shift"
    )

    print(
        f"Normal(0,1) -> Normal(0,{final_sigma})"
    )

    print()

    print(
        " CAlpha"
        " MAlpha"
        " ScaleT50"
        " ScaleT90"
        " ScaleBias"
        " ScaleRMSE"
        " CenterBias"
        " CenterRMSE"
    )

    print(
        "-" * 91
    )

    for row in rows:

        print(
            f"{row['center_alpha']:7.3f}"
            f"{row['mad_alpha']:7.3f}"
            f"{row['scale_t50']:9.0f}"
            f"{row['scale_t90']:9.0f}"
            f"{row['late_scale_bias']:11.6f}"
            f"{row['late_scale_rmse']:11.6f}"
            f"{row['late_center_bias']:12.6f}"
            f"{row['late_center_rmse']:12.6f}"
        )


def main() -> None:

    assert_bootstrap()

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
            0.05,
            0.02,
        ),
    ]

    seed_offset = 0

    # ------------------------------------------------------------------
    # STATIONARY
    # ------------------------------------------------------------------

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
                validate_stationary(
                    distribution,
                    center_alpha,
                    mad_alpha,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_stationary(
            distribution,
            rows,
        )

    # ------------------------------------------------------------------
    # LOCATION SHIFT
    # ------------------------------------------------------------------

    location_rows = []

    for (
        center_alpha,
        mad_alpha,
    ) in configurations:

        location_rows.append(
            validate_location_shift(
                center_alpha,
                mad_alpha,
                seed_offset,
            )
        )

        seed_offset += 1

    print_location_shift(
        location_rows
    )

    # ------------------------------------------------------------------
    # SCALE SHIFTS
    # ------------------------------------------------------------------

    for final_sigma in (
        2.0,
        0.5,
    ):

        scale_rows = []

        for (
            center_alpha,
            mad_alpha,
        ) in configurations:

            scale_rows.append(
                validate_scale_shift(
                    center_alpha,
                    mad_alpha,
                    final_sigma,
                    seed_offset,
                )
            )

            seed_offset += 1

        print_scale_shift(
            final_sigma,
            scale_rows,
        )


if __name__ == "__main__":
    main()
