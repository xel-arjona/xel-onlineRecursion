from __future__ import annotations

from dataclasses import dataclass
from math import floor, isnan, nan
from typing import Optional


def is_missing(value: float | None) -> bool:
    """Pine-like `na` test for numeric observations."""
    return value is None or (
        isinstance(value, float)
        and isnan(value)
    )


def round_half_to_even(value: float) -> int:
    """
    Round to the nearest integer using half-to-even midpoint resolution.

    This mirrors the explicit helper used by the frozen Pine P²
    implementation.
    """
    lower = floor(value)
    fraction = value - float(lower)

    if fraction < 0.5:
        return lower

    if fraction > 0.5:
        return lower + 1

    return lower if lower % 2 == 0 else lower + 1


@dataclass
class P2QuantileState:
    """
    Python transliteration of the frozen Pine P2QuantileState recurrence.

    One state represents one cumulative population and one fixed quantile
    probability.

    The estimator becomes defined after five valid observations.
    """

    value: float = nan
    probability: float = nan

    marker_heights: Optional[list[float]] = None
    marker_positions: Optional[list[Optional[int]]] = None
    desired_positions: Optional[list[float]] = None

    count: int = 0

    def reset(self) -> P2QuantileState:
        """
        Discard the current population while reusing five-marker storage.
        """
        self.value = nan
        self.probability = nan
        self.count = 0

        if self.marker_heights is None:
            self.marker_heights = [nan] * 5
        else:
            self.marker_heights[:] = [nan] * 5

        if self.marker_positions is None:
            self.marker_positions = [None] * 5
        else:
            self.marker_positions[:] = [None] * 5

        if self.desired_positions is None:
            self.desired_positions = [nan] * 5
        else:
            self.desired_positions[:] = [nan] * 5

        return self

    def update(
        self,
        input_value: float | None,
        probability: float,
    ) -> P2QuantileState:
        """
        Update the cumulative P² population with one observation.

        Missing observations preserve the retained estimator state.
        """

        # ------------------------------------------------------------------
        # CONFIGURATION VALIDATION
        # ------------------------------------------------------------------

        if (
            is_missing(probability)
            or probability <= 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "P² quantile probability must be strictly between zero and one."
            )

        if (
            not is_missing(self.probability)
            and self.probability != probability
        ):
            raise ValueError(
                "P² quantile probability cannot change within an active population."
            )

        valid_observation = not is_missing(input_value)

        if not valid_observation:
            return self

        # From this point forward input_value is known to be numeric.
        x = float(input_value)

        # Lazily allocate reusable five-marker storage.
        if self.marker_heights is None:
            self.reset()

        assert self.marker_heights is not None
        assert self.marker_positions is not None
        assert self.desired_positions is not None

        # Bind probability when the first valid observation starts
        # a new population.
        if is_missing(self.probability):
            self.probability = probability

        # ------------------------------------------------------------------
        # FIVE-OBSERVATION BOOTSTRAP
        # ------------------------------------------------------------------

        if self.count < 5:
            self.marker_heights[self.count] = x
            self.count += 1

            if self.count == 5:
                self.marker_heights.sort()

                n1 = round_half_to_even(
                    2.0 * probability
                )

                n2 = round_half_to_even(
                    4.0 * probability
                )

                n3 = round_half_to_even(
                    2.0 + 2.0 * probability
                )

                # Capture selected bootstrap values before changing
                # the three interior marker heights.
                initial_q1 = self.marker_heights[n1]
                initial_q2 = self.marker_heights[n2]
                initial_q3 = self.marker_heights[n3]

                self.marker_positions[0] = 0
                self.marker_positions[1] = n1
                self.marker_positions[2] = n2
                self.marker_positions[3] = n3
                self.marker_positions[4] = 4

                # Endpoints 0 and 4 remain the sorted minimum/maximum.
                self.marker_heights[1] = initial_q1
                self.marker_heights[2] = initial_q2
                self.marker_heights[3] = initial_q3

                self.desired_positions[0] = 0.0
                self.desired_positions[1] = (
                    2.0 * probability
                )
                self.desired_positions[2] = (
                    4.0 * probability
                )
                self.desired_positions[3] = (
                    2.0 + 2.0 * probability
                )
                self.desired_positions[4] = 4.0

                # q2 is the requested P² quantile marker.
                self.value = self.marker_heights[2]

            return self

        # ------------------------------------------------------------------
        # POST-BOOTSTRAP P² UPDATE
        # ------------------------------------------------------------------

        q0 = self.marker_heights[0]
        q1 = self.marker_heights[1]
        q2 = self.marker_heights[2]
        q3 = self.marker_heights[3]
        q4 = self.marker_heights[4]

        # Identify the marker interval containing x.
        if x < q0:
            self.marker_heights[0] = x
            k = 0

        elif x < q1:
            k = 0

        elif x < q2:
            k = 1

        elif x < q3:
            k = 2

        elif x < q4:
            k = 3

        else:
            self.marker_heights[4] = x
            k = 3

        # Every actual marker above the identified interval advances.
        for i in range(5):
            if i > k:
                marker_position = self.marker_positions[i]

                assert marker_position is not None

                self.marker_positions[i] = (
                    marker_position + 1
                )

        # Direct reconstruction of desired positions.
        #
        # count is the population size before incorporating the current
        # observation and therefore equals N - 1 for the expanded population.
        population_span = float(self.count)

        self.desired_positions[0] = 0.0

        self.desired_positions[1] = (
            population_span
            * probability
            / 2.0
        )

        self.desired_positions[2] = (
            population_span
            * probability
        )

        self.desired_positions[3] = (
            population_span
            * (1.0 + probability)
            / 2.0
        )

        self.desired_positions[4] = (
            population_span
        )

        # ------------------------------------------------------------------
        # INTERIOR MARKER ADJUSTMENT
        # ------------------------------------------------------------------

        for step in range(1, 4):
            i = (
                step
                if probability >= 0.5
                else 4 - step
            )

            previous_position = self.marker_positions[i - 1]
            current_position = self.marker_positions[i]
            next_position = self.marker_positions[i + 1]

            assert previous_position is not None
            assert current_position is not None
            assert next_position is not None

            desired_position = (
                self.desired_positions[i]
            )

            position_error = (
                desired_position
                - float(current_position)
            )

            move_up = (
                position_error >= 1.0
                and next_position - current_position > 1
            )

            move_down = (
                position_error <= -1.0
                and current_position - previous_position > 1
            )

            if not (move_up or move_down):
                continue

            direction = 1 if move_up else -1

            previous_height = (
                self.marker_heights[i - 1]
            )

            current_height = (
                self.marker_heights[i]
            )

            next_height = (
                self.marker_heights[i + 1]
            )

            left_gap = (
                current_position
                - previous_position
            )

            right_gap = (
                next_position
                - current_position
            )

            full_gap = (
                next_position
                - previous_position
            )

            parabolic_defined = (
                left_gap != 0
                and right_gap != 0
                and full_gap != 0
            )

            parabolic_height = nan

            if parabolic_defined:
                parabolic_height = (
                    current_height
                    + float(direction)
                    / float(full_gap)
                    * (
                        float(left_gap + direction)
                        * (
                            next_height
                            - current_height
                        )
                        / float(right_gap)
                        + float(right_gap - direction)
                        * (
                            current_height
                            - previous_height
                        )
                        / float(left_gap)
                    )
                )

            parabolic_valid = (
                not isnan(parabolic_height)
                and previous_height < parabolic_height
                and parabolic_height < next_height
            )

            adjusted_height = parabolic_height

            if not parabolic_valid:
                neighbor_index = (
                    i + direction
                )

                neighbor_position = (
                    self.marker_positions[
                        neighbor_index
                    ]
                )

                assert neighbor_position is not None

                neighbor_height = (
                    self.marker_heights[
                        neighbor_index
                    ]
                )

                adjusted_height = (
                    current_height
                    + float(direction)
                    * (
                        neighbor_height
                        - current_height
                    )
                    / float(
                        neighbor_position
                        - current_position
                    )
                )

            self.marker_heights[i] = (
                adjusted_height
            )

            self.marker_positions[i] = (
                current_position
                + direction
            )

        # Current observation has now entered the population.
        self.count += 1

        self.value = (
            self.marker_heights[2]
        )

        return self
