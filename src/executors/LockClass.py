import os
import sys
import copy

from collections import defaultdict


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../",
    )
)


from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor

from components.LockClass.src.models.PackageModel import (
    PackageModel,
)

from components.LockClass.src.utils.response import (
    build_response_lock_class,
)


# ============================================================
# TRACK STATE
# ============================================================

class TrackState:

    def __init__(self):

        # class_id -> vote count
        self.votes = defaultdict(int)

        # Currently locked class
        self.locked_class_id = None
        self.locked_class_label = None

        # Candidate class for switching
        self.switch_class_id = None
        self.switch_count = 0

        # Last frame where tracker was seen
        self.last_seen_frame = 0


# ============================================================
# LOCK CLASS
# ============================================================

class LockClass(Component):

    def __init__(self, request, bootstrap):

        super().__init__(
            request,
            bootstrap
        )

        self.request.model = PackageModel(
            **self.request.data
        )

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        self.min_votes = int(
            self._get_config_value(
                "MinVotes"
            )
        )

        self.vote_confidence = float(
            self._get_config_value(
                "VoteConfidence"
            )
        )

        self.lead_margin = int(
            self._get_config_value(
                "LeadMargin"
            )
        )

        self.switch_after = int(
            self._get_config_value(
                "SwitchAfter"
            )
        )

        self.state_ttl = int(
            self._get_config_value(
                "StateTTL"
            )
        )

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        self.tracked_detections = (
            self.request.get_param(
                "InputTrackedDetections"
            )
        )

        # ----------------------------------------------------
        # TRACK STATES
        # ----------------------------------------------------

        self.track_states = {}

        self.frame_index = 0


    # ========================================================
    # CONFIG HELPER
    # ========================================================

    def _get_config_value(self, name):

        value = self.request.get_param(
            name
        )

        if hasattr(value, "value"):
            return value.value

        return value


    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    # ========================================================
    # VALUE HELPER
    # ========================================================

    @staticmethod
    def _get_value(
        obj,
        *names,
        default=None
    ):

        if isinstance(obj, dict):

            for name in names:

                if name in obj:
                    return obj[name]

        else:

            for name in names:

                if hasattr(obj, name):

                    return getattr(
                        obj,
                        name
                    )

        return default


    # ========================================================
    # DETECTION FIELDS
    # ========================================================

    @classmethod
    def _get_tracker_id(cls, detection):

        return cls._get_value(
            detection,
            "trackerID",
            "tracker_id",
            default=None
        )


    @classmethod
    def _get_class_id(cls, detection):

        return cls._get_value(
            detection,
            "classId",
            "class_id",
            default=None
        )


    @classmethod
    def _get_class_label(cls, detection):

        return cls._get_value(
            detection,
            "classLabel",
            "class_label",
            default=None
        )


    @classmethod
    def _get_confidence(cls, detection):

        value = cls._get_value(
            detection,
            "confidence",
            default=0.0
        )

        try:

            return float(value)

        except (TypeError, ValueError):

            return 0.0


    # ========================================================
    # STATE
    # ========================================================

    def _get_state(self, tracker_id):

        if tracker_id not in self.track_states:

            self.track_states[
                tracker_id
            ] = TrackState()

        return self.track_states[
            tracker_id
        ]


    # ========================================================
    # VOTING
    # ========================================================

    def _add_vote(
        self,
        state,
        class_id,
        confidence
    ):

        if confidence >= self.vote_confidence:

            state.votes[
                class_id
            ] += 1


    # ========================================================
    # FIND WINNING CLASS
    # ========================================================

    def _get_winner(self, state):

        if not state.votes:

            return None

        sorted_votes = sorted(
            state.votes.items(),
            key=lambda item: item[1],
            reverse=True
        )

        winner_id = sorted_votes[0][0]
        winner_votes = sorted_votes[0][1]

        if len(sorted_votes) > 1:

            second_votes = sorted_votes[1][1]

        else:

            second_votes = 0

        margin = (
            winner_votes
            - second_votes
        )

        if (
            winner_votes >= self.min_votes
            and margin >= self.lead_margin
        ):

            return winner_id

        return None


    # ========================================================
    # CLASS LOCKING
    # ========================================================

    def _update_lock(
        self,
        state,
        class_id,
        class_label
    ):

        # ----------------------------------------------------
        # No class locked yet
        # ----------------------------------------------------

        if state.locked_class_id is None:

            winner = self._get_winner(
                state
            )

            if winner == class_id:

                state.locked_class_id = (
                    class_id
                )

                state.locked_class_label = (
                    class_label
                )

            return


        # ----------------------------------------------------
        # Same class as locked class
        # ----------------------------------------------------

        if class_id == state.locked_class_id:

            state.switch_class_id = None
            state.switch_count = 0

            state.locked_class_label = (
                class_label
            )

            return


        # ----------------------------------------------------
        # Different class
        # ----------------------------------------------------

        if (
            state.switch_class_id
            == class_id
        ):

            state.switch_count += 1

        else:

            state.switch_class_id = (
                class_id
            )

            state.switch_count = 1


        # ----------------------------------------------------
        # Switch class
        # ----------------------------------------------------

        if (
            state.switch_count
            >= self.switch_after
        ):

            state.locked_class_id = (
                class_id
            )

            state.locked_class_label = (
                class_label
            )

            state.switch_class_id = None
            state.switch_count = 0


    # ========================================================
    # PROCESS DETECTION
    # ========================================================

    def _process_detection(
        self,
        detection
    ):

        tracker_id = self._get_tracker_id(
            detection
        )

        # ----------------------------------------------------
        # No tracker ID
        # ----------------------------------------------------

        if tracker_id is None:

            return detection


        class_id = self._get_class_id(
            detection
        )

        class_label = self._get_class_label(
            detection
        )

        confidence = self._get_confidence(
            detection
        )


        # ----------------------------------------------------
        # No class ID
        # ----------------------------------------------------

        if class_id is None:

            return detection


        # ----------------------------------------------------
        # Get tracker state
        # ----------------------------------------------------

        state = self._get_state(
            tracker_id
        )

        state.last_seen_frame = (
            self.frame_index
        )


        # ----------------------------------------------------
        # Add vote
        # ----------------------------------------------------

        self._add_vote(
            state,
            class_id,
            confidence
        )


        # ----------------------------------------------------
        # Update locked class
        # ----------------------------------------------------

        self._update_lock(
            state,
            class_id,
            class_label
        )


        # ----------------------------------------------------
        # Apply locked class
        # ----------------------------------------------------

        if (
            state.locked_class_id
            is not None
        ):

            result = copy.deepcopy(
                detection
            )

            if isinstance(result, dict):

                result["classId"] = (
                    state.locked_class_id
                )

                result["classLabel"] = (
                    state.locked_class_label
                )

            else:

                if hasattr(
                    result,
                    "classId"
                ):

                    result.classId = (
                        state.locked_class_id
                    )

                if hasattr(
                    result,
                    "classLabel"
                ):

                    result.classLabel = (
                        state.locked_class_label
                    )

            return result


        return detection


    # ========================================================
    # CLEANUP
    # ========================================================

    def _cleanup_states(self):

        expired = []

        for tracker_id, state in (
            self.track_states.items()
        ):

            frames_since_seen = (
                self.frame_index
                - state.last_seen_frame
            )

            if (
                frames_since_seen
                > self.state_ttl
            ):

                expired.append(
                    tracker_id
                )


        for tracker_id in expired:

            del self.track_states[
                tracker_id
            ]


    # ========================================================
    # NORMALIZE INPUT
    # ========================================================

    def _normalize_detections(
        self,
        detections
    ):

        if detections is None:

            return []


        if isinstance(
            detections,
            str
        ):

            import json

            detections = json.loads(
                detections
            )


        if isinstance(
            detections,
            list
        ):

            return detections


        return [detections]


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            self.frame_index += 1

            detections = (
                self._normalize_detections(
                    self.tracked_detections
                )
            )


            output_detections = []


            for detection in detections:

                processed = (
                    self._process_detection(
                        detection
                    )
                )

                output_detections.append(
                    processed
                )


            self._cleanup_states()


            self.output_detections = (
                output_detections
            )


        except Exception as e:

            print(
                "LockClass Error:",
                repr(e),
                flush=True
            )

            raise


        return build_response_lock_class(
            context=self
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()