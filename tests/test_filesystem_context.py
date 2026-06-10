"""Unit tests for FilesystemContextAssembler."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from video_agent.memory import FilesystemContextAssembler
from video_agent.file_memory import ActivityLog


def test_lightweight_includes_activity_log_only():
    """Lightweight mode includes activity log but no delta or sandbox."""
    log = ActivityLog()
    assembler = FilesystemContextAssembler(activity_log=log)
    # Snapshot memory so delta would show if mode allowed it
    assembler.snapshot_memory(1, "## Section A\n" + "x" * 3000)
    result = assembler.assemble(
        mode="lightweight", tool_name="execute_bash",
        tool_input={}, current_memory="small", iteration=2,
    )
    assert "ACTIVITY LOG:" in result
    assert "WORKING MEMORY DELTA" not in result
    assert "SANDBOX FILES" not in result


def test_bootstrap_includes_activity_log():
    """Bootstrap mode includes activity log."""
    log = ActivityLog()
    log.record("execute_bash", {"command": "echo hello"}, "hello", iteration=1)
    assembler = FilesystemContextAssembler(activity_log=log)
    result = assembler.assemble(
        mode="bootstrap", tool_name="execute_bash",
        tool_input={}, current_memory="", iteration=2,
    )
    assert "ACTIVITY LOG:" in result
    assert "echo hello" in result


def test_working_memory_delta_detection():
    """Detects shrinkage in working memory."""
    log = ActivityLog()
    assembler = FilesystemContextAssembler(activity_log=log)

    # Snapshot a large memory
    big_memory = "## Section A\n" + "x" * 3000
    assembler.snapshot_memory(9, big_memory)

    # Assemble with much smaller memory
    small_memory = "## Section A\n" + "x" * 2000
    result = assembler.assemble(
        mode="edit", tool_name="analyze_frames",
        tool_input={}, current_memory=small_memory, iteration=10,
    )
    assert "WORKING MEMORY DELTA" in result
    assert "-" in result  # Negative percentage


def test_section_loss_detection():
    """Detects lost sections and provides recovery hints."""
    log = ActivityLog()
    assembler = FilesystemContextAssembler(activity_log=log)

    # Snapshot memory with multiple sections
    prev_memory = (
        "## Section A\nContent A here\n\n"
        "## Section B\nContent B here\n\n"
        "## Open Questions\n- Question 1\n- Question 2\n\n"
        "## Work Ledger\nSome work\n"
    )
    assembler.snapshot_memory(9, prev_memory)

    # Current memory missing two sections
    curr_memory = "## Section A\nContent A here\n"  # Much shorter too

    result = assembler.assemble(
        mode="edit", tool_name="analyze_frames",
        tool_input={}, current_memory=curr_memory, iteration=10,
    )
    assert "Lost:" in result
    # Should mention at least some of the lost sections
    assert "RECOVERY" in result


def test_no_recovery_when_stable():
    """No recovery hint when memory is stable."""
    log = ActivityLog()
    assembler = FilesystemContextAssembler(activity_log=log)

    memory = "## Section A\nContent A\n\n## Section B\nContent B\n"
    assembler.snapshot_memory(9, memory)

    # Similar size memory with same sections
    result = assembler.assemble(
        mode="edit", tool_name="analyze_frames",
        tool_input={}, current_memory=memory, iteration=10,
    )
    assert "RECOVERY" not in result


def test_ring_buffer_eviction():
    """Ring buffer keeps only last 3 snapshots."""
    log = ActivityLog()
    assembler = FilesystemContextAssembler(activity_log=log)

    for i in range(5):
        assembler.snapshot_memory(i, f"Memory at iter {i}")

    assert len(assembler._memory_history) == 3
    assert assembler._memory_history[0][0] == 2  # Oldest remaining is iter 2


def test_available_frames_in_context():
    """Available frames appear in assembled context."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t120.jpg", "/outputs/frames/frame_t180.jpg"]},
        "Stage with blue curtains and podium.",
        iteration=5,
    )
    assembler = FilesystemContextAssembler(activity_log=log)
    result = assembler.assemble(
        mode="edit", tool_name="analyze_frames",
        tool_input={}, current_memory="test", iteration=6,
    )
    assert "AVAILABLE FRAMES:" in result
    assert "frame_t120.jpg" in result
    assert "inspect_frames" in result  # Instruction to inspect frames


def test_budget_truncation():
    """Output respects max_chars budget."""
    log = ActivityLog()
    # Generate lots of activity
    for i in range(50):
        log.record(
            "analyze_frames",
            {"image_paths": [f"/outputs/frames/frame_t{i * 1000}.jpg"]},
            f"Long description for scene at {i * 1000}s that goes on and on.",
            iteration=i,
        )
    assembler = FilesystemContextAssembler(activity_log=log, max_chars=500)
    result = assembler.assemble(
        mode="edit", tool_name="analyze_frames",
        tool_input={}, current_memory="test", iteration=51,
    )
    assert len(result) <= 500


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
