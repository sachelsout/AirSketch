"""
src/stream.py

Top-level AirSketch application entry point.

Wires together:
    InferenceLoop  (issue #13) -- webcam capture + ONNX prediction
    StrokeOverlay  (issue #14) -- stroke buffer + cv2 rendering
    VirtualCamera              -- pyvirtualcam output to video-call apps

The annotated frame flows:
    webcam frame
        -> InferenceLoop (MediaPipe + ONNX)
        -> StrokeOverlay.process() via frame_callback
        -> VirtualCamera.send()
        -> Zoom / Google Meet / Microsoft Teams

Usage:
    python src/stream.py

    # With explicit device (Linux)
    python src/stream.py --device /dev/video2

    # Show a local preview window alongside the virtual camera output
    python src/stream.py --preview

    # Headless (virtual camera only, no preview window)
    python src/stream.py --no-preview
"""

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam

from src.inference import InferenceLoop, InferenceResult
from src.overlay import StrokeOverlay, StrokeColor


# -- Virtual camera wrapper ----------------------------------------------------


class VirtualCamera:
    """
    Thin wrapper around pyvirtualcam.Camera that handles OS differences,
    frame format conversion, and timing.

    pyvirtualcam expects frames in RGB format (not OpenCV's default BGR).
    This wrapper converts automatically so callers can pass BGR frames
    directly from OpenCV without thinking about color channels.

    It also implements a frame-repeat strategy: if the pipeline is running
    slower than the target FPS (e.g. a MediaPipe spike took 60 ms on a
    50 ms frame budget), the last good frame is re-sent to keep the virtual
    camera's frame rate stable. Video-call applications drop the connection
    if frame delivery stops for more than ~500 ms.

    Args:
        width:       Frame width in pixels.
        height:      Frame height in pixels.
        fps:         Target frame rate for the virtual camera.
        device:      Device path (Linux: "/dev/video2", macOS: None for auto).
        fmt:         pyvirtualcam pixel format. RGB is the default and works
                     on both macOS and Linux.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        device: str | None = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device

        self._cam: pyvirtualcam.Camera | None = None
        self._last_frame: np.ndarray | None = None

    def open(self) -> "VirtualCamera":
        """Open the virtual camera. Call before sending frames."""
        kwargs = dict(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        if self.device:
            kwargs["device"] = self.device

        try:
            self._cam = pyvirtualcam.Camera(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open virtual camera: {exc}\n"
                f"Check that the driver is installed (see issue #15 docs).\n"
                f"  macOS: OBS Studio must be installed and virtual camera "
                f"activated once from the OBS UI.\n"
                f"  Linux: v4l2loopback must be loaded "
                f"(sudo modprobe v4l2loopback ...)."
            ) from exc

        print(f"Virtual camera opened: {self._cam.device}")
        print(f"  Resolution: {self.width}x{self.height} @ {self.fps} fps")
        return self

    def send(self, bgr_frame: np.ndarray) -> None:
        """
        Send one BGR frame to the virtual camera.

        Converts BGR -> BGR (pyvirtualcam.PixelFormat.BGR handles this natively).
        Resizes the frame if its dimensions do not match the camera's configured
        resolution (guards against resolution changes mid-session).

        Args:
            bgr_frame: (H, W, 3) uint8 BGR frame from OpenCV.
        """
        if self._cam is None:
            raise RuntimeError("VirtualCamera not opened. Call open() first.")

        h, w = bgr_frame.shape[:2]
        if w != self.width or h != self.height:
            bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

        self._last_frame = bgr_frame
        self._cam.send(bgr_frame)

    def repeat_last_frame(self) -> None:
        """
        Re-send the last frame to prevent the virtual camera from stalling.
        Called when the pipeline is behind schedule.
        """
        if self._last_frame is not None and self._cam is not None:
            self._cam.send(self._last_frame)

    def sleep_until_next_frame(self) -> None:
        """Block until the next frame slot opens (rate limiter)."""
        if self._cam is not None:
            self._cam.sleep_until_next_frame()

    def close(self) -> None:
        if self._cam is not None:
            self._cam.__exit__(None, None, None)
            self._cam = None

    def __enter__(self) -> "VirtualCamera":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()


# -- Main stream function ------------------------------------------------------


def run_stream(
    model_path: str | Path = "checkpoints/best_model.optimized.onnx",
    config_path: str | Path = "configs/default.yaml",
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    device: str | None = None,
    show_preview: bool = True,
    log_path: str | Path | None = None,
) -> None:
    """
    Run the full AirSketch stream: capture -> predict -> overlay -> virtual cam.

    Args:
        model_path:   Path to best_model.optimized.onnx.
        config_path:  Path to configs/default.yaml.
        camera_index: OpenCV webcam device index.
        width:        Capture and output width.
        height:       Capture and output height.
        fps:          Target frame rate.
        device:       Virtual camera device path (Linux only).
        show_preview: Open a local OpenCV preview window.
        log_path:     Optional CSV path for per-frame latency log.
    """
    print("AirSketch -- starting stream")
    print(f"  OS:     {platform.system()} {platform.release()}")
    print(f"  Model:  {model_path}")
    print(f"  Camera: index={camera_index}  {width}x{height} @ {fps}fps")
    print(f"  Device: {device or 'auto'}")
    print()

    # -- Build overlay ---------------------------------------------------------
    overlay = StrokeOverlay(
        initial_color=StrokeColor.WHITE,
        thickness=3,
        debounce_frames=4,
        show_hud=True,
        show_fingertip=True,
    )

    # -- Annotated frame store (set inside callback, read in main loop) --------
    annotated: dict[str, np.ndarray | None] = {"frame": None}

    def frame_callback(bgr_frame: np.ndarray, result: InferenceResult) -> None:
        """Called by InferenceLoop after each ONNX inference pass."""
        # Apply stroke overlay in-place
        overlay.process(bgr_frame, result)
        annotated["frame"] = bgr_frame

        # Keyboard controls -- only active when preview window is open
        if show_preview:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                overlay.clear_canvas()
            elif key == ord("n"):
                overlay.cycle_color()
            elif key == ord("1"):
                overlay.set_color(StrokeColor.WHITE)
            elif key == ord("2"):
                overlay.set_color(StrokeColor.RED)
            elif key == ord("3"):
                overlay.set_color(StrokeColor.GREEN)
            elif key == ord("4"):
                overlay.set_color(StrokeColor.BLUE)
            elif key == ord("+") or key == ord("="):
                overlay.set_thickness(overlay._buffer.thickness + 1)
            elif key == ord("-"):
                overlay.set_thickness(overlay._buffer.thickness - 1)

    # -- Build inference loop --------------------------------------------------
    loop = InferenceLoop(
        model_path=model_path,
        camera_index=camera_index,
        width=width,
        height=height,
        target_fps=fps,
        log_path=log_path,
        frame_callback=frame_callback,
    )

    # -- Open virtual camera and run -------------------------------------------
    with VirtualCamera(width=width, height=height, fps=fps, device=device) as vcam:
        print("Streaming to virtual camera. Open your video-call app and")
        print(f"select '{vcam._cam.device}' as your camera source.")
        print("Press Q in the preview window to quit.\n")

        frame_deadline_sec = 1.0 / fps
        last_send_time = time.perf_counter()

        for result in loop.run(show_preview=False):
            now = time.perf_counter()

            if annotated["frame"] is not None:
                vcam.send(annotated["frame"])

                if show_preview:
                    cv2.imshow("AirSketch -- preview (Q to quit)", annotated["frame"])

                last_send_time = now
            else:
                # Pipeline is behind -- repeat last frame to keep vcam alive
                if now - last_send_time > frame_deadline_sec:
                    vcam.repeat_last_frame()
                    last_send_time = now

            vcam.sleep_until_next_frame()

    print("\nStream stopped.")
    print(f"  Total strokes:  {overlay.stroke_count}")
    print(f"  Total points:   {overlay.total_points}")
    print(f"  Canvas clears:  {overlay.clear_count}")


# -- CLI -----------------------------------------------------------------------


def _detect_default_device() -> str | None:
    """Return the default virtual camera device path for the current OS."""
    if platform.system() == "Linux":
        # Find the first v4l2loopback device
        import glob

        devices = sorted(glob.glob("/dev/video*"))
        for d in devices:
            try:
                result = __import__("subprocess").run(
                    ["v4l2-ctl", "--device", d, "--info"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if (
                    "v4l2loopback" in result.stdout.lower()
                    or "airsketch" in result.stdout.lower()
                ):
                    return d
            except Exception:
                continue
        # Fall back to /dev/video2 if auto-detect fails
        return "/dev/video2"
    # macOS: pyvirtualcam auto-detects OBS Virtual Camera
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AirSketch -- virtual camera stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="checkpoints/best_model.optimized.onnx")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--device",
        default=None,
        help="Virtual camera device (Linux: /dev/video2). " "Auto-detected if omitted.",
    )
    p.add_argument("--preview", dest="preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument(
        "--log-path", default=None, help="CSV path for per-frame latency log."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or _detect_default_device()
    run_stream(
        model_path=args.model,
        config_path=args.config,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        device=device,
        show_preview=args.preview,
        log_path=args.log_path,
    )
