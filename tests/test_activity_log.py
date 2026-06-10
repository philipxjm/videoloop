"""Unit tests for ActivityLog merge logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from video_agent.file_memory import ActivityLog


def test_single_visual_analysis():
    """Single VLM analysis renders correctly."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t120.jpg", "/outputs/frames/frame_t150.jpg"]},
        "Scene shows a stage with blue curtains and a podium in the center.",
        iteration=4,
    )
    assert len(log.entries) == 1
    rendered = log.render()
    assert "120-150s" in rendered
    assert "2 frames" in rendered
    assert "iter 4" in rendered


def test_consecutive_vlm_analyses_no_merge():
    """Consecutive VLM analyses stay separate (each preserves its own context)."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t0.jpg", "/outputs/frames/frame_t60.jpg",
                         "/outputs/frames/frame_t90.jpg", "/outputs/frames/frame_t120.jpg"]},
        "Opening scene with credits.",
        iteration=4,
    )
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t120.jpg", "/outputs/frames/frame_t180.jpg",
                         "/outputs/frames/frame_t210.jpg", "/outputs/frames/frame_t240.jpg"]},
        "Interview segment begins.",
        iteration=5,
    )
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t240.jpg", "/outputs/frames/frame_t300.jpg",
                         "/outputs/frames/frame_t330.jpg", "/outputs/frames/frame_t360.jpg"]},
        "Trophy presentation ceremony.",
        iteration=6,
    )
    # Each call is its own entry (no merging — preserves per-call analysis)
    assert len(log.entries) == 3
    assert "Opening scene" in log.entries[0]["visual_analysis"]
    assert "Interview" in log.entries[1]["visual_analysis"]
    assert "Trophy" in log.entries[2]["visual_analysis"]


def test_non_adjacent_analyses_dont_merge():
    """VLM analyses on non-adjacent ranges (>60s gap) stay separate."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t0.jpg", "/outputs/frames/frame_t120.jpg"]},
        "Opening scene.",
        iteration=4,
    )
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t600.jpg", "/outputs/frames/frame_t720.jpg"]},
        "Closing scene.",
        iteration=5,
    )
    assert len(log.entries) == 2


def test_frame_extraction_merge():
    """Consecutive frame extractions with same step merge."""
    log = ActivityLog()
    log.record(
        "execute_bash",
        {"command": "for t in $(seq 0 30 900); do ffmpeg -ss $t -i /videos/v.mp4 -frames:v 1 /outputs/frames/frame_t${t}.jpg; done"},
        "Extracted 31 frames.",
        iteration=2,
    )
    log.record(
        "execute_bash",
        {"command": "for t in $(seq 0 30 900); do ffmpeg -ss $t -i /videos/v.mp4 -frames:v 1 /outputs/frames/frame_t${t}.jpg; done"},
        "Extracted 31 frames.",
        iteration=3,
    )
    assert len(log.entries) == 1
    rendered = log.render()
    assert "iter 2-3" in rendered


def test_different_types_dont_merge():
    """Different action types don't merge."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t0.jpg", "/outputs/frames/frame_t60.jpg"]},
        "Scene analysis.",
        iteration=4,
    )
    log.record(
        "execute_bash",
        {"command": "python analyze.py"},
        "Analysis complete.",
        iteration=5,
    )
    assert len(log.entries) == 2


def test_interleaved_types_break_merges():
    """Bash command between VLM analyses prevents merging."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t0.jpg", "/outputs/frames/frame_t120.jpg"]},
        "Opening scene.",
        iteration=4,
    )
    log.record(
        "execute_bash",
        {"command": "python analyze.py"},
        "Analysis complete.",
        iteration=5,
    )
    log.record(
        "analyze_frames",
        {"image_paths": ["/outputs/frames/frame_t120.jpg", "/outputs/frames/frame_t240.jpg"]},
        "Middle scene.",
        iteration=6,
    )
    # The bash in between prevents merging
    assert len(log.entries) == 3


def test_transcript_stays_singular():
    """Transcript entries don't merge."""
    log = ActivityLog()
    log.record(
        "transcribe_audio",
        {},
        '{"segments_count": 516, "duration": 3446, "transcript": "..."}',
        iteration=4,
    )
    log.record(
        "transcribe_audio",
        {},
        '{"segments_count": 516, "duration": 3446, "transcript": "..."}',
        iteration=5,
    )
    assert len(log.entries) == 2


def test_render_compact_under_load():
    """Render stays compact with many diverse actions."""
    log = ActivityLog()
    for i in range(25):
        if i % 4 == 0:
            log.record(
                "analyze_frames",
                {"image_paths": [f"/outputs/frames/frame_t{i * 30}.jpg"]},
                f"Scene at {i * 30}s.",
                iteration=i,
            )
        elif i % 4 == 1:
            log.record("execute_bash", {"command": f"python step{i}.py"}, "ok", iteration=i)
        elif i % 4 == 2:
            log.record("create_file", {"path": f"/testbed/file_{i}.txt"}, "ok", iteration=i)
        else:
            log.record("other_tool", {}, "result", iteration=i)

    rendered = log.render()
    assert len(rendered) < 2000  # Should stay well under budget


def test_submit_answer_ignored():
    """submit_answer and pin_memory are not recorded."""
    log = ActivityLog()
    log.record("submit_answer", {"answer": "A"}, "Submitted", iteration=10)
    log.record("pin_memory", {"content": "test"}, "Pinned", iteration=11)
    assert len(log.entries) == 0
    assert log.render() == "(no activity yet)"


def test_get_available_frames():
    """get_available_frames returns frame paths with VLM summaries."""
    log = ActivityLog()
    log.record(
        "analyze_frames",
        {"image_paths": [
            "/outputs/frames/frame_t120.jpg",
            "/outputs/frames/frame_t150.jpg",
            "/outputs/frames/frame_t180.jpg",
        ]},
        "Stage with blue curtains, podium center, speaker at lectern.",
        iteration=5,
    )
    frames = log.get_available_frames()
    assert len(frames) == 3
    assert frames[0]["path"] == "/outputs/frames/frame_t120.jpg"
    assert frames[0]["timestamp"] == 120.0
    assert "Stage with blue curtains" in frames[0]["visual_analysis"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
