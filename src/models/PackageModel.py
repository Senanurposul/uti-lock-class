from typing import Any, List, Literal, Optional, Union

from pydantic import Field

from sdks.novavision.src.base.model import (
    Package,
    Inputs,
    Outputs,
    Configs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


# ============================================================
# INPUT
# ============================================================

class InputTrackedDetections(Input):
    name: Literal["InputTrackedDetections"] = "InputTrackedDetections"
    value: List
    type: Literal["list"] = "list"

    class Config:
        title = "Tracked Detections"


class VideoIdentifier(Input):
    name: Literal["VideoIdentifier"] = "VideoIdentifier"
    value: Optional[Any] = None
    type: Literal["string"] = "string"

    class Config:
        title = "Video Identifier"
        json_schema_extra = {
            "shortDescription":
                "Optional stream/video identifier used to isolate tracker "
                "state across multiple simultaneous video sources. "
                "Defaults to a single shared stream if left unconnected."
        }


# ============================================================
# OUTPUT
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: list
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


# ============================================================
# CONFIGS
# ============================================================

class MinVotes(Config):
    name: Literal["MinVotes"] = "MinVotes"
    value: int = Field(
        default=3,
        ge=1
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Votes"
        json_schema_extra = {
            "shortDescription":
                "Minimum number of votes required to lock a class."
        }


class VoteConfidence(Config):
    name: Literal["VoteConfidence"] = "VoteConfidence"
    value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Vote Confidence"
        json_schema_extra = {
            "shortDescription":
                "Minimum confidence required for voting."
        }


class LeadMargin(Config):
    name: Literal["LeadMargin"] = "LeadMargin"
    value: int = Field(
        default=1,
        ge=0
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Lead Margin"
        json_schema_extra = {
            "shortDescription":
                "Minimum vote difference required to lock a class."
        }


class SwitchAfter(Config):
    name: Literal["SwitchAfter"] = "SwitchAfter"
    value: int = Field(
        default=3,
        ge=1
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Switch After"
        json_schema_extra = {
            "shortDescription":
                "Consecutive frames required to switch class."
        }


class StateTTL(Config):
    name: Literal["StateTTL"] = "StateTTL"
    value: int = Field(
        default=30,
        ge=1
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "State TTL"
        json_schema_extra = {
            "shortDescription":
                "Frames before an inactive tracker state is removed."
        }


class ReattachWindow(Config):
    name: Literal["ReattachWindow"] = "ReattachWindow"
    value: int = Field(
        default=10,
        ge=0
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Reattach Window"
        json_schema_extra = {
            "shortDescription":
                "Maximum number of frames a lost locked track is kept "
                "in memory for possible reattachment to a new tracker ID."
        }


class ReattachIoU(Config):
    name: Literal["ReattachIoU"] = "ReattachIoU"
    value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Reattach IoU"
        json_schema_extra = {
            "shortDescription":
                "Minimum bounding box IoU required to reattach a lost "
                "locked track to a newly appeared tracker ID."
        }


# ============================================================
# PACKAGE INPUT / OUTPUT
# ============================================================

class LockClassInputs(Inputs):
    InputTrackedDetections: InputTrackedDetections
    VideoIdentifier: Optional[VideoIdentifier] = None


class LockClassConfigs(Configs):
    MinVotes: MinVotes
    VoteConfidence: VoteConfidence
    LeadMargin: LeadMargin
    SwitchAfter: SwitchAfter
    StateTTL: StateTTL
    ReattachWindow: ReattachWindow
    ReattachIoU: ReattachIoU


class LockClassOutputs(Outputs):
    OutputDetections: OutputDetections


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class LockClassRequest(Request):
    inputs: Optional[LockClassInputs] = None
    configs: LockClassConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class LockClassResponse(Response):
    outputs: LockClassOutputs


# ============================================================
# EXECUTOR MODEL
# ============================================================

class LockClassExecutor(Config):
    name: Literal["LockClass"] = "LockClass"

    value: Union[
        LockClassRequest,
        LockClassResponse,
    ]

    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Lock Class"

        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"

    value: LockClassExecutor

    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"

        json_schema_extra = {
            "target": "value"
        }


# ============================================================
# PACKAGE
# ============================================================

class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs

    type: Literal["component"] = "component"

    name: Literal["LockClass"] = "LockClass"