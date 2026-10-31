"""Synchronized RGB recordings for sampled evaluation episodes."""
from pathlib import Path

import numpy as np


def diagnostic_video_required(row_index: int, interval: int = 10) -> bool:
    if interval <= 0:
        raise ValueError("Recording interval must be positive")
    return (int(row_index) + 1) % interval == 0


class VideoRecorder:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps)
        self.writer = None
        self.frames = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        import cv2

        frame = np.asarray(frame_rgb, dtype=np.uint8)[..., :3]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h), isColor=True)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.path}")
            preview = self.path.with_suffix(".png")
            if not cv2.imwrite(str(preview), frame[..., ::-1]):
                raise RuntimeError(f"Failed to save video preview: {preview}")
        self.writer.write(frame[..., ::-1])
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class EpisodeVideoRecorder:
    def __init__(self, directory: Path, fps: float):
        self.ego = VideoRecorder(directory / "ego.mp4", fps)
        self.third_person = VideoRecorder(directory / "third_person.mp4", fps)
        self.fps = float(fps)

    @property
    def frames(self) -> int:
        if self.ego.frames != self.third_person.frames:
            raise RuntimeError("Evaluation video streams have different frame counts")
        return self.ego.frames

    def write(self, ego: np.ndarray, third_person: np.ndarray) -> None:
        self.ego.write(ego)
        self.third_person.write(third_person)

    def close(self) -> None:
        try:
            self.ego.close()
        finally:
            self.third_person.close()


def capture_third_person(env, env_id: int) -> np.ndarray:
    camera = env.scene["evaluation_camera"]
    camera.update(dt=0.0, force_recompute=True)
    return np.asarray(camera.data.output["rgb"][env_id].detach().cpu().numpy(), dtype=np.uint8)[..., :3].copy()
