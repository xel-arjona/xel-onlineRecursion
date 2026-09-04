from __future__ import annotations

from dataclasses import dataclass
from math import isnan, nan


def is_missing(
    value: float | None,
) -> bool:

    if value is None:
        return True

    return isnan(
        float(value)
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
    """
    Candidate recursively adaptive quantile estimator.

    Recurrence:

        innovation_t =
            x_t - q_(t-1)

        score_t =
            p          if innovation_t > 0
            p - 1      if innovation_t < 0
            0          if innovation_t = 0

        step_t =
            alpha_t * scale_(t-1)

        q_t =
            q_(t-1)
            + step_t * score_t

        scale_t =
            scale_(t-1)
            + alpha_t
              * (
                    |innovation_t|
                    - scale_(t-1)
                )

    The quantile update always uses the scale retained BEFORE
    processing the current observation.

    The current innovation affects scale only for subsequent
    observations.
    """

    value: float = nan
    probability: float = nan
    scale: float = nan

    def reset(
        self,
    ) -> None:

        self.value = nan
        self.probability = nan
        self.scale = nan

    def update(
        self,
        input_value: float | None,
        probability: float,
        alpha: float | None,
    ) -> AdaptiveQuantileState:

        # -------------------------------------------------------------
        # STATIC QUANTILE CONFIGURATION
        # -------------------------------------------------------------

        if (
            is_missing(
                probability
            )
            or probability <= 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "probability must be strictly between zero and one"
            )

        p = float(
            probability
        )

        if (
            not is_missing(
                self.probability
            )
            and p != self.probability
        ):
            raise ValueError(
                "probability cannot change inside an active population"
            )

        # -------------------------------------------------------------
        # OBSERVATION VALIDITY
        #
        # Missing input or alpha preserves the complete retained state.
        # -------------------------------------------------------------

        valid_observation = (
            not is_missing(
                input_value
            )
            and not is_missing(
                alpha
            )
        )

        if not valid_observation:
            return self

        observation = float(
            input_value
        )

        coefficient = clamp(
            float(alpha),
            0.0,
            1.0,
        )

        # -------------------------------------------------------------
        # INITIALIZATION
        #
        # The first valid observation seeds location directly.
        #
        # Scale begins at zero. This is intentional:
        #
        #     - no arbitrary unit-dependent initialization constant
        #     - alpha = 0 remains a true no-contribution coefficient
        #     - subsequent observations establish scale recursively
        # -------------------------------------------------------------

        if is_missing(
            self.value
        ):
            self.value = observation
            self.probability = p
            self.scale = 0.0

            return self

        # -------------------------------------------------------------
        # CURRENT INNOVATION AGAINST PREVIOUS LOCATION
        # -------------------------------------------------------------

        innovation = (
            observation
            - self.value
        )

        previous_scale = (
            self.scale
        )

        # -------------------------------------------------------------
        # QUANTILE SCORE
        #
        # The zero score at an exact tie is a valid subgradient choice
        # and prevents deterministic location drift at equality.
        # -------------------------------------------------------------

        score = (
            p
            if innovation > 0.0
            else p - 1.0
            if innovation < 0.0
            else 0.0
        )

        # -------------------------------------------------------------
        # QUANTILE UPDATE
        #
        # Crucially uses PREVIOUS scale.
        # -------------------------------------------------------------

        step = (
            coefficient
            * previous_scale
        )

        self.value += (
            step
            * score
        )

        # -------------------------------------------------------------
        # SCALE UPDATE
        #
        # Current innovation updates the scale only after the quantile
        # correction has been determined.
        # -------------------------------------------------------------

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
