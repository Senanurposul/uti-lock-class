import os
import sys
import copy

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor

from components.LockClass.src.utils.response import build_response_lock_class
from components.LockClass.src.models.PackageModel import PackageModel


class LockClass(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.tracked_detections = self.request.get_param(
            "InputTrackedDetections"
        )

        self.min_votes = int(
            self.request.get_param("MinVotes")
        )

        self.vote_confidence = float(
            self.request.get_param("VoteConfidence")
        )

        self.lead_margin = int(
            self.request.get_param("LeadMargin")
        )

        self.switch_after = int(
            self.request.get_param("SwitchAfter")
        )

        self.state_ttl = int(
            self.request.get_param("StateTTL")
        )

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "track_states": {},
            "frame_index": 0
        }

    @staticmethod
    def get_value(detection, key, default=None):
        if isinstance(detection, dict):
            return detection.get(key, default)

        return getattr(detection, key, default)

    def get_tracker_id(self, detection):
        tracker_id = self.get_value(
            detection,
            "trackerID"
        )

        if tracker_id is None:
            tracker_id = self.get_value(
                detection,
                "tracker_id"
            )

        return tracker_id

    def get_class_id(self, detection):
        class_id = self.get_value(
            detection,
            "classId"
        )

        if class_id is None:
            class_id = self.get_value(
                detection,
                "class_id"
            )

        return class_id

    def get_class_label(self, detection):
        class_label = self.get_value(
            detection,
            "classLabel"
        )

        if class_label is None:
            class_label = self.get_value(
                detection,
                "class_label"
            )

        return class_label

    def get_confidence(self, detection):
        confidence = self.get_value(
            detection,
            "confidence",
            0.0
        )

        try:
            return float(confidence)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def create_state():
        return {
            "votes": {},
            "lockedClassId": None,
            "lockedClassLabel": None,
            "switchClassId": None,
            "switchCount": 0,
            "lastSeenFrame": 0
        }

    def get_state(self, tracker_id):
        states = self.bootstrap["track_states"]

        if tracker_id not in states:
            states[tracker_id] = self.create_state()

        return states[tracker_id]

    def add_vote(self, state, class_id, confidence):
        if confidence < self.vote_confidence:
            return

        if class_id not in state["votes"]:
            state["votes"][class_id] = 0

        state["votes"][class_id] += 1

    def find_winner(self, state):
        votes = state["votes"]

        if not votes:
            return None

        sorted_votes = sorted(
            votes.items(),
            key=lambda item: item[1],
            reverse=True
        )

        winner_id = sorted_votes[0][0]
        winner_votes = sorted_votes[0][1]

        second_votes = (
            sorted_votes[1][1]
            if len(sorted_votes) > 1
            else 0
        )

        margin = winner_votes - second_votes

        if (
            winner_votes >= self.min_votes
            and margin >= self.lead_margin
        ):
            return winner_id

        return None

    def update_lock(self, state, class_id, class_label):
        if state["lockedClassId"] is None:
            winner = self.find_winner(state)

            if winner == class_id:
                state["lockedClassId"] = class_id
                state["lockedClassLabel"] = class_label

            return

        if class_id == state["lockedClassId"]:
            state["lockedClassLabel"] = class_label
            state["switchClassId"] = None
            state["switchCount"] = 0
            return

        if state["switchClassId"] == class_id:
            state["switchCount"] += 1
        else:
            state["switchClassId"] = class_id
            state["switchCount"] = 1

        if state["switchCount"] >= self.switch_after:
            state["lockedClassId"] = class_id
            state["lockedClassLabel"] = class_label
            state["switchClassId"] = None
            state["switchCount"] = 0
            state["votes"] = {
                class_id: 1
            }

    def process_detection(self, detection):
        tracker_id = self.get_tracker_id(detection)

        if tracker_id is None:
            return copy.deepcopy(detection)

        class_id = self.get_class_id(detection)
        class_label = self.get_class_label(detection)
        confidence = self.get_confidence(detection)

        if class_id is None:
            return copy.deepcopy(detection)

        state = self.get_state(tracker_id)

        state["lastSeenFrame"] = self.bootstrap["frame_index"]

        self.add_vote(
            state,
            class_id,
            confidence
        )

        self.update_lock(
            state,
            class_id,
            class_label
        )

        result = copy.deepcopy(detection)

        if isinstance(result, dict):
            result["voteCount"] = state["votes"].get(
                class_id,
                0
            )

            result["frameIndex"] = self.bootstrap["frame_index"]

            result["lockedClassId"] = state["lockedClassId"]
            result["lockedClassLabel"] = state["lockedClassLabel"]
            result["isLocked"] = state["lockedClassId"] is not None

            if state["lockedClassId"] is not None:
                result["classId"] = state["lockedClassId"]
                result["classLabel"] = state["lockedClassLabel"]

        return result

    def cleanup_states(self):
        current_frame = self.bootstrap["frame_index"]
        states = self.bootstrap["track_states"]

        expired = []

        for tracker_id, state in states.items():
            frame_difference = (
                current_frame - state["lastSeenFrame"]
            )

            if frame_difference > self.state_ttl:
                expired.append(tracker_id)

        for tracker_id in expired:
            del states[tracker_id]

    @staticmethod
    def normalize_detections(detections):
        if detections is None:
            return []

        if isinstance(detections, list):
            return detections

        return [detections]

    def run(self):
        self.bootstrap["frame_index"] += 1

        detections = self.normalize_detections(
            self.tracked_detections
        )

        output_detections = []

        for detection in detections:
            output_detections.append(
                self.process_detection(detection)
            )

        self.cleanup_states()

        self.output_detections = output_detections

        return build_response_lock_class(
            context=self
        )


if __name__ == "__main__":
    Executor(sys.argv[1]).run()