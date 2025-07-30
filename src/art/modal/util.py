import dataclasses
import enum
from typing import Union


class ModalGPUType(str, enum.Enum):
    B200 = "B200"
    H200 = "H200"
    H100 = "H100"
    H100_STRICT = "H100!"
    A100_80G = "A100-80G"
    A100_40G = "A100-40G"
    L40S = "L40S"
    A10G = "A10G"
    L4 = "L4"
    T4 = "T4"


class ModalGPUConfig(str):
    def __new__(cls, value):
        # Handle both "GPU_TYPE" and "GPU_TYPE:count" formats
        if ":" in value:
            gpu_type_str, count_str = value.split(":", 1)
            try:
                count = int(count_str)
                if not (1 <= count <= 8):
                    raise ValueError(f"GPU count must be between 1 and 8, got {count}")
            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(
                        f"Invalid GPU count '{count_str}', must be an integer"
                    )
                raise
            # Validate GPU type
            try:
                ModalGPUType(gpu_type_str)
            except ValueError:
                valid_types = ", ".join([f"'{g.value}'" for g in ModalGPUType])
                raise ValueError(
                    f"Invalid GPU type '{gpu_type_str}'. Must be one of: {valid_types}"
                )
        else:
            # Just GPU type, validate it exists
            try:
                ModalGPUType(value)
            except ValueError:
                valid_types = ", ".join([f"'{g.value}'" for g in ModalGPUType])
                raise ValueError(
                    f"Invalid GPU type '{value}'. Must be one of: {valid_types}"
                )

        return str.__new__(cls, value)

    @property
    def gpu_type(self) -> ModalGPUType:
        """Get the GPU type part."""
        if ":" in self:
            return ModalGPUType(self.split(":", 1)[0])
        return ModalGPUType(self)

    @property
    def gpu_count(self) -> int:
        """Get the GPU count (defaults to 1 if not specified)."""
        if ":" in self:
            return int(self.split(":", 1)[1])
        return 1


# TODO[jvmncs]: support @modal.experimental.clustered
@dataclasses.dataclass
class ModalClusterConfig:
    num_nodes: int
    gpu_config: Union[str, ModalGPUConfig] = ModalGPUConfig(ModalGPUType.H100)

    def __post_init__(self):
        if isinstance(self.gpu_config, str):
            self.gpu_config = ModalGPUConfig(self.gpu_config)

        # @modal.experimental.clustered only supports H100s at the moment
        if self.gpu_config.gpu_type != ModalGPUType.H100 and self.num_nodes != 1:
            raise ValueError(
                f"num_nodes must be 1 when using gpu_type {self.gpu_config.gpu_type}. "
                f"At time of writing, only {ModalGPUType.H100} supports multiple nodes."
            )

    @property
    def gpus_per_node(self) -> int:
        """Get the number of GPUs per node from the GPU config."""
        # Ensure gpu_config is ModalGPUConfig type
        if isinstance(self.gpu_config, str):
            gpu_config = ModalGPUConfig(self.gpu_config)
            return gpu_config.gpu_count
        return self.gpu_config.gpu_count

    @property
    def gpu_type(self) -> ModalGPUType:
        """Get the GPU type from the GPU config."""
        # Ensure gpu_config is ModalGPUConfig type
        if isinstance(self.gpu_config, str):
            gpu_config = ModalGPUConfig(self.gpu_config)
            return gpu_config.gpu_type
        return self.gpu_config.gpu_type

    def gpu_str(self):
        return str(self.gpu_config)
