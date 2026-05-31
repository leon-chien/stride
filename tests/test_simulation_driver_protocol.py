from stride.drivers import FrameWindow, State, TrajectoryHandle


def test_driver_payload_types_are_hashable() -> None:
    state = State(id="state-1", payload_uri="memory://state-1")
    handle = TrajectoryHandle(id="traj-1")
    window = FrameWindow(
        trajectory_id=handle.id,
        start_frame=0,
        frame_count=64,
        payload_uri="memory://traj-1/0",
    )

    assert state.id == "state-1"
    assert window.frame_count == 64
    assert len({handle}) == 1
