from sdks.novavision.src.helper.package import PackageHelper

from components.LockClass.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    LockClassExecutor,
    LockClassResponse,
    LockClassOutputs,
    OutputDetections,
)


def build_response_lock_class(context):

    output_detections = OutputDetections(
        value=context.output_detections
    )

    outputs = LockClassOutputs(
        OutputDetections=output_detections
    )

    response = LockClassResponse(
        outputs=outputs
    )

    executor = LockClassExecutor(
        value=response
    )

    config_executor = ConfigExecutor(
        value=executor
    )

    package_configs = PackageConfigs(
        executor=config_executor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs,
    )

    return package.build_model(context)