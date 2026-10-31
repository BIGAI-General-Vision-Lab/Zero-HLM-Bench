import imageio
import h5py
import cv2
import numpy as np
from pathlib import Path

def read_rgb_video(path: Path) -> np.ndarray:
    """
    Read RGB video file and return as numpy array in N*H*W*3 format (uint8).

    Args:
        path: Path to video file (e.g., .mp4)

    Returns:
        numpy.ndarray: Video frames in shape [T, H, W, 3] (uint8)
    """
    reader = imageio.get_reader(str(path))
    frames = [frame for frame in reader]
    video_array = np.stack(frames, axis=0)

    # Ensure correct shape and type
    if video_array.ndim != 4:
        raise ValueError(f"Unexpected video shape: {video_array.shape}")
    if video_array.shape[3] != 3:
        raise ValueError(f"Expected 3 channels, got {video_array.shape[3]}")

    return video_array

def read_depth_h5(path: Path) -> np.ndarray:
    """
    Read depth frames from HDF5 file and return as numpy array in N*H*W format (float32 in meters).

    Args:
        path: Path to HDF5 file containing depth frames

    Returns:
        numpy.ndarray: Depth frames in shape [N, H, W] (float32 in meters)
    """
    depth_frames = []
    with h5py.File(path, 'r') as h5:
        # Get frames in order (frame_0000, frame_0001, ...)
        frame_keys = sorted(k for k in h5.keys() if k.startswith('frame_'))
        for key in frame_keys:
            encoded = h5[key][()]
            # Decode PNG
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            depth_frames.append(decoded)

    if not depth_frames:
        raise ValueError("No depth frames found in HDF5 file")

    depth_array = np.stack(depth_frames, axis=0)
    # Convert from uint16 (millimeters) to float32 (meters)
    depth_array = depth_array.astype(np.float32) / 1000.0

    # Replace any invalid values with 0.0
    depth_array[depth_array < 0] = 0.0
    depth_array[depth_array <= 0] = 0.0

    return depth_array

def read_semantic_video(path: Path) -> np.ndarray:
    """
    Read semantic video file and return as numpy array in N*H*W*4 format (RGBA, uint8).

    Args:
        path: Path to semantic video file (e.g., .avi)

    Returns:
        numpy.ndarray: Semantic frames in shape [N, H, W, 4] (RGBA, uint8)
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add alpha channel (255 = fully opaque)
        if frame_rgb.shape[2] == 3:
            alpha = np.full(frame_rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            frame_rgba = np.concatenate([frame_rgb, alpha], axis=2)
        else:
            frame_rgba = frame_rgb

        frames.append(frame_rgba)

    cap.release()

    if not frames:
        raise ValueError("No frames read from video file")

    semantic_array = np.stack(frames, axis=0)

    # Ensure we have RGBA output
    if semantic_array.shape[3] != 4:
        raise ValueError(f"Expected RGBA output, got {semantic_array.shape[3]} channels")

    return semantic_array