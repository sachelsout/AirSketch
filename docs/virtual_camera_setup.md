# Virtual Camera Setup

AirSketch routes its annotated video feed through a virtual camera driver
so it appears as a camera source in Zoom, Google Meet, Microsoft Teams,
Discord, and other video-call applications.

## Windows (tested)

1. Download and install OBS Studio from https://obsproject.com/download
2. Open OBS → find **Start Virtual Camera** in the Controls panel (bottom right) → click it once
3. Close OBS (the driver persists)
4. Install pyvirtualcam:
   ```bash
   pip install pyvirtualcam==0.11.1
   ```
5. Run AirSketch:
   ```bash
   python -m src.stream --no-preview
   ```

## macOS (untested — should work)

1. Install OBS Studio: `brew install --cask obs`
2. Open OBS → click **Start Virtual Camera** once → close OBS
3. Grant camera permission: System Settings → Privacy & Security → Camera
4. `pip install pyvirtualcam==0.11.1`
5. `python -m src.stream`

## Linux Ubuntu 22.04/24.04 (untested)

1. `sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils`
2. `sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="AirSketch Virtual Camera" exclusive_caps=1`
3. `sudo usermod -aG video $USER` (log out and back in)
4. `pip install pyvirtualcam==0.11.1`
5. `python -m src.stream --device /dev/video2`

## Selecting AirSketch in your video-call app

Start `src/stream.py` **before** opening your video-call application.
Then select **OBS Virtual Camera** as your camera source in:

- **Zoom**: Settings → Video → Camera
- **Google Meet**: Settings → Video
- **Microsoft Teams**: Settings → Devices → Camera
- **Discord**: Settings → Voice & Video → Camera

## Notes

- Minimizing the terminal window on Windows may cause lag — keep it visible or run with `start python -m src.stream --no-preview`
- The virtual camera driver runs independently of OBS after the one-time activation step
- Drawing requires index finger pointing gesture (other fingers curled)
- Press **C** to clear canvas, **N** to cycle colors, **Q** to quit
