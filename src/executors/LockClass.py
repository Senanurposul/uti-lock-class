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


# ============================================================
# IoU YARDIMCI FONKSIYONU (Reattachment icin)
# ============================================================

def compute_iou(bbox_a, bbox_b):
    """
    bbox format: {"left": x, "top": y, "width": w, "height": h}
    Reattachment karari icin iki bounding box arasindaki IoU'yu hesaplar.
    """
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax1, ay1 = bbox_a["left"], bbox_a["top"]
    ax2, ay2 = ax1 + bbox_a["width"], ay1 + bbox_a["height"]

    bx1, by1 = bbox_b["left"], bbox_b["top"]
    bx2, by2 = bx1 + bbox_b["width"], by1 + bbox_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, bbox_a["width"]) * max(0.0, bbox_a["height"])
    area_b = max(0.0, bbox_b["width"]) * max(0.0, bbox_b["height"])

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class LockClass(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.tracked_detections = self.request.get_param(
            "InputTrackedDetections"
        )

        # Video/stream ayrimi icin. Girdi saglanmazsa tek bir
        # varsayilan akis (default) altinda calisir.
        self.video_identifier = (
            self.request.get_param("VideoIdentifier")
            or "default"
        )

        self.min_votes = int(self.request.get_param("MinVotes"))
        self.vote_confidence = float(self.request.get_param("VoteConfidence"))
        self.lead_margin = int(self.request.get_param("LeadMargin"))
        self.switch_after = int(self.request.get_param("SwitchAfter"))
        self.state_ttl = int(self.request.get_param("StateTTL"))
        self.reattach_window = int(self.request.get_param("ReattachWindow"))
        self.reattach_iou = float(self.request.get_param("ReattachIoU"))

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # video_identifier -> { "tracks": {...}, "lost": {...}, "frame_index": int }
        return {"videos": {}}

    # --------------------------------------------------------
    # DETECTION ALAN OKUMA (SDK-bagimsiz, dict veya obje kabul eder)
    # --------------------------------------------------------

    @staticmethod
    def get_value(detection, key, default=None):
        if isinstance(detection, dict):
            return detection.get(key, default)
        return getattr(detection, key, default)

    def get_tracker_id(self, detection):
        value = self.get_value(detection, "trackerID")
        if value is None:
            value = self.get_value(detection, "tracker_id")
        return value

    def get_class_id(self, detection):
        value = self.get_value(detection, "classId")
        if value is None:
            value = self.get_value(detection, "class_id")
        return value

    def get_class_label(self, detection):
        value = self.get_value(detection, "classLabel")
        if value is None:
            value = self.get_value(detection, "class_label")
        return value

    def get_confidence(self, detection):
        value = self.get_value(detection, "confidence", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def get_bbox(self, detection):
        value = self.get_value(detection, "boundingBox")
        if value is None:
            value = self.get_value(detection, "bounding_box")
        return value

    # --------------------------------------------------------
    # VIDEO / TRACK STATE
    # --------------------------------------------------------

    def get_video_state(self):
        videos = self.bootstrap["videos"]

        if self.video_identifier not in videos:
            videos[self.video_identifier] = {
                "tracks": {},
                "lost": {},
                "frame_index": 0,
            }

        return videos[self.video_identifier]

    @staticmethod
    def create_state():
        return {
            "votes": {},              # class_id -> count (pre-lock)
            "vote_conf_sum": {},      # class_id -> confidence toplami (pre-lock)
            "lockedClassId": None,
            "lockedClassLabel": None,
            "confSum": 0.0,           # locked class icin running-mean toplami
            "confCount": 0,
            "switchClassId": None,
            "switchCount": 0,
            "switchConfSum": 0.0,
            "lastSeenFrame": 0,
            "lastBbox": None,
        }

    def get_or_create_track(self, video_state, tracker_id):
        tracks = video_state["tracks"]

        if tracker_id in tracks:
            return tracks[tracker_id]

        lost = video_state["lost"]

        # -------- AYNI ID GERI GELDI: gercek reattachment DEGIL --------
        # Tracker (ByteTrack vb.) bazen nesneyi birkac kare boyunca
        # algilayamasa bile kendi ic belleginde ayni tracker_id'yi
        # koruyabilir. Bu durumda IoU tahminine hic gerek yok --
        # ID zaten ayni oldugu icin bu %100 ayni nesnedir. Dogrudan
        # eski state geri yuklenir, REATTACH olarak loglanmaz.
        if tracker_id in lost:
            return lost.pop(tracker_id)["state"]

        # -------- REATTACHMENT DENEMESI (gercekten farkli bir ID) --------
        # Bu tracker_id daha once hic gorulmemis. Yakin zamanda
        # kaybolmus (locked) bir track var mi ve IoU esigini
        # geciyor mu diye kontrol edilir.
        adopted_state = self._try_reattach(video_state, tracker_id)

        if adopted_state is not None:
            tracks[tracker_id] = adopted_state
            return adopted_state

        new_state = self.create_state()
        tracks[tracker_id] = new_state
        return new_state

    def _try_reattach(self, video_state, new_tracker_id):
        current_bbox = self._pending_bbox_by_tracker.get(new_tracker_id)

        if current_bbox is None:
            return None

        lost = video_state["lost"]
        current_frame = video_state["frame_index"]

        best_match_id = None
        best_iou = 0.0

        for lost_id, lost_entry in lost.items():
            frame_gap = current_frame - lost_entry["lostFrame"]

            if frame_gap > self.reattach_window:
                continue

            iou = compute_iou(current_bbox, lost_entry["lastBbox"])

            if iou >= self.reattach_iou and iou > best_iou:
                best_iou = iou
                best_match_id = lost_id

        if best_match_id is None:
            return None

        recovered_state = lost.pop(best_match_id)["state"]

        # ---- GECICI DEBUG LOG (test sonrasi kaldirilacak) ----
        print(
            f"REATTACH DEBUG -> eski_tracker_id: {best_match_id} "
            f"yeni_tracker_id: {new_tracker_id} iou: {best_iou:.3f} "
            f"locked_class: {recovered_state.get('lockedClassLabel')} "
            f"frame: {video_state['frame_index']}",
            flush=True,
        )
        # ---- /GECICI DEBUG LOG ----

        return recovered_state

    # --------------------------------------------------------
    # VOTING / LOCKING / SWITCHING
    # --------------------------------------------------------

    def add_vote(self, state, class_id, confidence):
        if confidence < self.vote_confidence:
            return

        state["votes"][class_id] = state["votes"].get(class_id, 0) + 1
        state["vote_conf_sum"][class_id] = (
            state["vote_conf_sum"].get(class_id, 0.0) + confidence
        )

    def find_winner(self, state):
        votes = state["votes"]

        if not votes:
            return None

        sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        winner_id, winner_votes = sorted_votes[0]
        second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
        margin = winner_votes - second_votes

        if winner_votes >= self.min_votes and margin >= self.lead_margin:
            return winner_id

        return None

    def update_lock(self, state, class_id, class_label, confidence):
        # -------- HENUZ KILITLI DEGIL: cumulative voting --------
        if state["lockedClassId"] is None:
            winner = self.find_winner(state)

            if winner is not None and winner == class_id:
                state["lockedClassId"] = class_id
                state["lockedClassLabel"] = class_label
                state["confSum"] = state["vote_conf_sum"].get(class_id, 0.0)
                state["confCount"] = state["votes"].get(class_id, 0)

            return

        # -------- KILITLI: mevcut class ile eslesiyor --------
        if class_id == state["lockedClassId"]:
            if confidence >= self.vote_confidence:
                state["confSum"] += confidence
                state["confCount"] += 1

            state["lockedClassLabel"] = class_label
            state["switchClassId"] = None
            state["switchCount"] = 0
            state["switchConfSum"] = 0.0
            return

        # -------- KILITLI: farkli class geldi (switch adayi) --------
        if confidence < self.vote_confidence:
            # Roboflow: dusuk confidence switch streak'ini bozar/sayilmaz.
            state["switchClassId"] = None
            state["switchCount"] = 0
            state["switchConfSum"] = 0.0
            return

        if state["switchClassId"] == class_id:
            state["switchCount"] += 1
            state["switchConfSum"] += confidence
        else:
            state["switchClassId"] = class_id
            state["switchCount"] = 1
            state["switchConfSum"] = confidence

        if state["switchCount"] >= self.switch_after:
            state["lockedClassId"] = class_id
            state["lockedClassLabel"] = class_label

            # Roboflow davranisi: yeni class'in oy sayaci sifirdan
            # degil, switch streak'inden beslenir.
            state["confSum"] = state["switchConfSum"]
            state["confCount"] = state["switchCount"]

            state["votes"] = {}
            state["vote_conf_sum"] = {}
            state["switchClassId"] = None
            state["switchCount"] = 0
            state["switchConfSum"] = 0.0

    @staticmethod
    def running_mean_confidence(state):
        if state["confCount"] <= 0:
            return None

        mean = state["confSum"] / state["confCount"]
        return min(mean, 1.0)

    # --------------------------------------------------------
    # FRAME ISLEME
    # --------------------------------------------------------

    def process_detection(self, video_state, detection):
        tracker_id = self.get_tracker_id(detection)

        if tracker_id is None:
            return copy.deepcopy(detection)

        class_id = self.get_class_id(detection)
        class_label = self.get_class_label(detection)
        confidence = self.get_confidence(detection)
        bbox = self.get_bbox(detection)

        if class_id is None:
            return copy.deepcopy(detection)

        state = self.get_or_create_track(video_state, tracker_id)

        state["lastSeenFrame"] = video_state["frame_index"]
        state["lastBbox"] = bbox

        self.add_vote(state, class_id, confidence)
        self.update_lock(state, class_id, class_label, confidence)

        result = copy.deepcopy(detection)

        if isinstance(result, dict):
            is_locked = state["lockedClassId"] is not None

            result["voteCount"] = state["votes"].get(class_id, 0)
            result["frameIndex"] = video_state["frame_index"]
            result["lockedClassId"] = state["lockedClassId"]
            result["lockedClassLabel"] = state["lockedClassLabel"]
            result["isLocked"] = is_locked
            result["classLocked"] = is_locked  # Roboflow: class_locked flag

            if is_locked:
                result["classId"] = state["lockedClassId"]
                result["classLabel"] = state["lockedClassLabel"]
                result["confidence"] = self.running_mean_confidence(state)

        return result

    def mark_disappeared_tracks(self, video_state, seen_tracker_ids):
        tracks = video_state["tracks"]
        lost = video_state["lost"]
        current_frame = video_state["frame_index"]

        # Gercek tracker'lar (ByteTrack vb.) bazen tek bir karede
        # nesneyi algilayamayabilir (motion blur, gecici confidence
        # dususu) ama ayni tracker_id'yi korumaya devam eder. Bu tek
        # karelik bosluk gercek bir ID degisimi degildir, bu yuzden
        # aninda "kayip" sayip gereksiz reattach tetiklemek yerine
        # kucuk bir tolerans (grace period) taniyoruz. Yalnizca bu
        # toleransi asan track'ler gercekten "kayip" sayilip
        # reattachment icin saklanir.
        MISS_GRACE_FRAMES = 1

        disappeared_ids = [
            tracker_id
            for tracker_id, state in tracks.items()
            if tracker_id not in seen_tracker_ids
            and (current_frame - state["lastSeenFrame"]) > MISS_GRACE_FRAMES
        ]

        for tracker_id in disappeared_ids:
            state = tracks.pop(tracker_id)

            # Sadece kilitli track'ler reattach icin saklanir;
            # kilitlenmemis track'lerin devri anlamsizdir.
            if state["lockedClassId"] is not None:
                lost[tracker_id] = {
                    "state": state,
                    "lastBbox": state["lastBbox"],
                    "lostFrame": current_frame,
                }

    def cleanup(self, video_state):
        current_frame = video_state["frame_index"]

        # Uzun sure eslesmemis lost track'leri temizle
        lost = video_state["lost"]
        expired_lost = [
            tid for tid, entry in lost.items()
            if current_frame - entry["lostFrame"] > self.reattach_window
        ]
        for tid in expired_lost:
            del lost[tid]

        # state_ttl asan aktif track'leri temizle (guvenlik agi)
        tracks = video_state["tracks"]
        expired_tracks = [
            tid for tid, state in tracks.items()
            if current_frame - state["lastSeenFrame"] > self.state_ttl
        ]
        for tid in expired_tracks:
            del tracks[tid]

    @staticmethod
    def normalize_detections(detections):
        if detections is None:
            return []
        if isinstance(detections, list):
            return detections
        return [detections]

    def run(self):
        video_state = self.get_video_state()
        video_state["frame_index"] += 1

        detections = self.normalize_detections(self.tracked_detections)

        # Reattachment kontrolunde kullanmak icin bu karedeki
        # tum tracker_id -> bbox eslemesini onceden cikar.
        self._pending_bbox_by_tracker = {}
        for detection in detections:
            tid = self.get_tracker_id(detection)
            if tid is not None:
                self._pending_bbox_by_tracker[tid] = self.get_bbox(detection)

        seen_tracker_ids = set(self._pending_bbox_by_tracker.keys())

        output_detections = []
        for detection in detections:
            output_detections.append(
                self.process_detection(video_state, detection)
            )

        self.mark_disappeared_tracks(video_state, seen_tracker_ids)
        self.cleanup(video_state)

        self.output_detections = output_detections

        return build_response_lock_class(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()