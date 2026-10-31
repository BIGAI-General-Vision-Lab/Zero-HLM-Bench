import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime/scripts/imitation_learning/robomimic'
sys.path.insert(0, str(RUNTIME))
from evaluation_video import EpisodeVideoRecorder, diagnostic_video_required, capture_third_person


def test_every_tenth_manifest_episode_including_last():
    assert [i + 1 for i in range(100) if diagnostic_video_required(i)] == list(range(10, 101, 10))
    assert [i + 1 for i in range(17, 33) if diagnostic_video_required(i)] == [20, 30]
    assert diagnostic_video_required(4, 5)
    with pytest.raises(ValueError):
        diagnostic_video_required(0, 0)


def test_paired_files_decode_with_identical_timebase(tmp_path):
    import cv2
    recorder = EpisodeVideoRecorder(tmp_path / 'episode', fps=50)
    for i in range(8):
        recorder.write(np.full((48, 64, 3), (i * 20 + 10, 20, 30), np.uint8),
                       np.full((60, 80, 3), (30, i * 20 + 10, 20), np.uint8))
    recorder.close()
    recorder.close()
    assert recorder.frames == 8
    assert np.array_equal(cv2.imread(str(recorder.ego.path.with_suffix('.png'))),
                          np.full((48, 64, 3), (30, 20, 10), np.uint8))
    assert np.array_equal(cv2.imread(str(recorder.third_person.path.with_suffix('.png'))),
                          np.full((60, 80, 3), (20, 10, 30), np.uint8))
    for path, width, height, channel in ((recorder.ego.path, 64, 48, 2),
                                         (recorder.third_person.path, 80, 60, 1)):
        stream = cv2.VideoCapture(str(path))
        try:
            assert stream.get(cv2.CAP_PROP_FPS) == pytest.approx(50)
            frames = []
            while True:
                ok, frame = stream.read()
                if not ok:
                    break
                assert frame.shape == (height, width, 3)
                frames.append(float(frame[:, :, channel].mean()))
            assert len(frames) == 8
            assert all(b > a for a, b in zip(frames, frames[1:]))
        finally:
            stream.release()


def test_paired_close_releases_both_on_failure(tmp_path):
    recorder = EpisodeVideoRecorder(tmp_path, fps=50)
    recorder.ego.close = Mock(side_effect=RuntimeError('encoder error'))
    recorder.third_person.close = Mock()
    with pytest.raises(RuntimeError):
        recorder.close()
    recorder.third_person.close.assert_called_once()


def test_third_person_capture_refreshes_sensor_without_physics_step():
    frame = np.ones((4, 6, 4), dtype=np.uint8)
    tensor = Mock()
    tensor.detach.return_value.cpu.return_value.numpy.return_value = frame
    camera = SimpleNamespace(update=Mock(), data=SimpleNamespace(output={'rgb': [None, tensor]}))
    env = SimpleNamespace(scene={'evaluation_camera': camera}, step=Mock())
    captured = capture_third_person(env, 1)
    camera.update.assert_called_once_with(dt=0.0, force_recompute=True)
    env.step.assert_not_called()
    assert captured.shape == (4, 6, 3)
    assert captured.dtype == np.uint8
