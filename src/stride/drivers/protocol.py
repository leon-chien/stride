from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class State:
    """Serializable starting point for a simulation trajectory."""

    id: str
    payload_uri: str


@dataclass(frozen=True, slots=True)
class TrajectoryHandle:
    """Opaque handle returned by a SimulationDriver."""

    id: str


@dataclass(frozen=True, slots=True)
class FrameWindow:
    """A non-blocking batch of frames emitted by a running trajectory."""

    trajectory_id: str
    start_frame: int
    frame_count: int
    payload_uri: str


class SimulationDriver(Protocol):
    def list_active(self) -> list[TrajectoryHandle]: ...

    def submit(self, init_state: State, budget_ns: float, priority: float) -> TrajectoryHandle: ...

    def poll(self, handle: TrajectoryHandle) -> FrameWindow | None: ...

    def extend(self, handle: TrajectoryHandle, additional_ns: float) -> None: ...

    def terminate(self, handle: TrajectoryHandle) -> None: ...
