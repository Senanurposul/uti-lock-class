import os
import sys
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from capsules.LockClass.src.models.PackageModel import PackageModel
from capsules.LockClass.src.utils.response import build_response


class LockClass(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        self.tracked_detections = self.request.get_param(
            "InputTrackedDetections"
        )

        self.minimum_votes = self.request.get_param("MinimumVotes")
        self.vote_confidence = self.request.get_param("VoteConfidence")
        self.lead_margin = self.request.get_param("LeadMargin")
        self.switch_after = self.request.get_param("SwitchAfter")
        self.state_ttl = self.request.get_param("StateTTL")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "track_states": {},
            "frame_index": 0
        }

    def create_state(self) -> Dict[str, Any]:
        return {
            "votes": {},
            "locked_class_id": None,
            "locked_class_label": None,
            "is_locked": False,
            "vote_count": 0,
            "last_seen": 0,
            "switch_count": 0,
        }

    def update_state(
        self,
        state: Dict[str, Any],
        detection: Dict[str, Any],
        frame_index: int
    ):
        class_id = detection.get("classId")
        class_label = detection.get("classLabel")
        confidence = detection.get("confidence", 0)

        state["last_seen"] = frame_index

        if confidence < self.vote_confidence:
            return

        if class_id not in state["votes"]:
            state["votes"][class_id] = {
                "count": 0,
                "label": class_label
            }

        state["votes"][class_id]["count"] += 1

        best_class_id = None
        best_count = 0
        second_best_count = 0

        for cid, vote_data in state["votes"].items():
            count = vote_data["count"]

            if count > best_count:
                second_best_count = best_count
                best_count = count
                best_class_id = cid
            elif count > second_best_count:
                second_best_count = count

        state["vote_count"] = best_count

        if best_class_id is None:
            return

        best_label = state["votes"][best_class_id]["label"]

        if not state["is_locked"]:
            if (
                best_count >= self.minimum_votes
                and (best_count - second_best_count) >= self.lead_margin
            ):
                state["locked_class_id"] = best_class_id
                state["locked_class_label"] = best_label
                state["is_locked"] = True

        else:
            if best_class_id != state["locked_class_id"]:
                if best_count > second_best_count:
                    state["switch_count"] += 1

                    if state["switch_count"] >= self.switch_after:
                        state["locked_class_id"] = best_class_id
                        state["locked_class_label"] = best_label
                        state["switch_count"] = 0
            else:
                state["switch_count"] = 0

    def process_detection(
        self,
        detection: Dict[str, Any],
        frame_index: int
    ) -> Dict[str, Any]:

        tracker_id = detection.get("trackerID")

        if tracker_id is None:
            return detection

        states = self.bootstrap["track_states"]

        if tracker_id not in states:
            states[tracker_id] = self.create_state()

        state = states[tracker_id]

        self.update_state(
            state,
            detection,
            frame_index
        )

        output_detection = detection.copy()

        output_detection["voteCount"] = state["vote_count"]
        output_detection["frameIndex"] = frame_index
        output_detection["lockedClassId"] = state["locked_class_id"]
        output_detection["lockedClassLabel"] = state["locked_class_label"]
        output_detection["isLocked"] = state["is_locked"]

        return output_detection

    def cleanup_states(self, frame_index: int):

        states = self.bootstrap["track_states"]

        expired_trackers = []

        for tracker_id, state in states.items():
            if frame_index - state["last_seen"] > self.state_ttl:
                expired_trackers.append(tracker_id)

        for tracker_id in expired_trackers:
            del states[tracker_id]

    def run(self):

        # ---------------------------------------------------------
        # DEBUG
        # ---------------------------------------------------------
        print(
            "LOCK CLASS DEBUG:",
            id(self.bootstrap),
            self.bootstrap.get("frame_index")
        )

        # ---------------------------------------------------------
        # FRAME INDEX
        # ---------------------------------------------------------
        self.bootstrap["frame_index"] += 1
        frame_index = self.bootstrap["frame_index"]

        output_detections = []

        if self.tracked_detections is None:
            self.tracked_detections = []

        for detection in self.tracked_detections:

            processed_detection = self.process_detection(
                detection=detection,
                frame_index=frame_index
            )

            output_detections.append(processed_detection)

        self.cleanup_states(frame_index)

        package_model = build_response(
            context=self,
            output_detections=output_detections
        )

        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()