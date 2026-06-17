# HandForge Tracker — System & Camera Tuning Guide

* Last Modified: 2026-05-27

## §1 Purpose & Scope

This document captures *recipe-style operational guidance* for tuning the HandForge tracker's camera and capture stack — choices that affect runtime behaviour but are not derivable from reading the code alone (USB-bandwidth interactions, OS-specific backend quirks, hand-tracking-specific stabilisation).

**In scope**
- USB bandwidth & FOURCC codec selection (§2).
- OS-specific capture backends (§3).
- Auto-focus / auto-exposure disabling rationale (§4).

**Out of scope**
- Measured performance values / KPIs → see [`PERFORMANCE.md`](PERFORMANCE.md).
- Full reference list of every configurable setting → see [`config.yaml`](config.yaml) and [`config.py`](src/hand_tracker/config.py). This document covers a *subset of settings that benefit from explicit guidance*, not a complete catalogue.

## §2 USB Bandwidth & FOURCC Codec

### Why
A webcam delivers frames over USB. The USB cable has a fixed bandwidth budget (USB 2.0 ≈ 35 MB/s effective; USB 3.0 ≈ 400 MB/s). High resolution combined with high FPS in an uncompressed format can exceed that budget — at which point the driver may silently cap the frame rate (e.g., 1080p YUYV @ 60 FPS dropping to 5–15 FPS without an explicit error).

The codec the camera streams in is identified by a **FOURCC** (Four Character Code) — e.g., `"MJPG"`, `"YUYV"`, `"H264"`. Compressed codecs save bandwidth; uncompressed codecs do not.

### Project defaults (code)
- `camera.fourcc: "MJPG"` — default in `CameraConfig` (`config.py`).
- FOURCC is applied *before* resolution and FPS in the property sequence (see `_open_device` in `capture.py`); this ordering is enforced because some drivers silently reject a high resolution / FPS combination if the codec was already negotiated as uncompressed.

### Recommended
- **`MJPG`** — Motion JPEG, per-frame compressed. Tested as this project's baseline. Recommended for any USB 2.0 / USB 3.0 webcam at typical resolutions.

### Known trap
- **`YUYV`** — raw YUV 4:2:2, uncompressed. At high resolution or high FPS, USB bandwidth is exhausted and the driver silently caps FPS. Symptoms: requested 60 FPS, actual FPS is 5–15, no error logged. See the design comment in `_open_device` (`capture.py`) for the rationale.

### Other codecs (e.g., `H264`)
- Not tested in this project. Driver / hardware support varies (some webcams have hardware H.264 encoders, others do not). If you want to use one, measure first — see [`PERFORMANCE.md`](PERFORMANCE.md) for the runtime stats command.

### Caveats
- The OpenCV official documentation does not explicitly describe the FOURCC-first ordering or the USB-bandwidth silent-fallback behaviour. The rationale recorded here is taken from the design comment inside `_open_device` (`capture.py`).

## §3 OS-Specific Capture Backends

OpenCV reaches the camera through an OS-specific API layer called the *capture backend*. The backend is independent of the FOURCC codec (§2): different backends can negotiate different codec sets with the same camera.

### Available backends (defined as a Literal type on `CameraConfig.backend` in `config.py`)
- `AUTO` (default) — OpenCV picks the OS standard backend. This is the project's verified configuration.
- `ANY` — OpenCV picks the first available backend.
- `DSHOW` (Windows) — DirectShow.
- `MSMF` (Windows) — Media Foundation.
- `V4L2` (Linux) — Video for Linux 2.
- `AVFOUNDATION` (macOS) — AVFoundation.

### Verified
- **Windows + `AUTO`** — this project's development environment. Normal operation.

### Not verified in this project
- Windows: `DSHOW` vs `MSMF` direct comparison. External material claims DSHOW has faster initialisation, but this project has no measurement of its own.
- Linux (`V4L2`) — no test environment.
- macOS (`AVFOUNDATION`) — no test environment.

If `AUTO` produces problems on your environment, try a specific backend from the list above as a starting point for diagnosis.

## §4 Disabling Auto-Focus & Auto-Exposure

### Why (hand-tracking-specific)
A webcam normally adjusts focus and exposure continuously based on what it sees, optimising for visual quality. For hand tracking, these adjustments are jitter sources:
- **Auto-focus**: hand moves closer or farther → lens re-focuses → frames are briefly blurry → landmark positions wobble.
- **Auto-exposure**: hand enters / leaves frame → brightness adjusts → per-frame input changes → landmark positions wobble.

Hand tracking values *frame-to-frame consistency* over visual quality, so both are disabled by default.

### Project defaults (code)
- `camera.disable_auto_focus: true` — default on `CameraConfig` (`config.py`).
- `camera.disable_auto_exposure: true` — default on `CameraConfig` (`config.py`).

### What it does
- Auto-focus → `cv2.CAP_PROP_AUTOFOCUS = 0` (applied in `_open_device`, `capture.py`).
- Auto-exposure → `cv2.CAP_PROP_AUTO_EXPOSURE = 0.25` (applied in `_open_device`, `capture.py`).
  - The `0.25` value commonly corresponds to manual mode in DirectShow drivers; see the design comment in `_open_device` (`capture.py`).

### Caveats
- The `0.25` value is **not documented in the OpenCV official reference** for `CAP_PROP_AUTO_EXPOSURE`. Its origin is in OpenCV's **V4L2 (Linux) backend**: when range normalisation is enabled (`OPENCV_VIDEOIO_V4L_RANGE_NORMALIZED`), V4L2's exposure mode enum (`0 = Auto`, `1 = Manual`, `2 = Shutter Priority`, `3 = Aperture Priority`) is mapped onto `[0, 1)` via the hardcoded `Range(0, 4)`, yielding `(1 − 0) / 4 = 0.25` for Manual mode. See the `normalizePropRange` logic in OpenCV (`modules/videoio/src/cap_v4l.cpp`).
- This project runs on Windows (DirectShow / Media Foundation), where the V4L2 normalisation does **not** apply. The `0.25` value in `_open_device` therefore originates from V4L2 convention rather than a documented Windows-backend behaviour, and works on the project's setup through driver passthrough.
- The hardcoded `0.25` is intentionally scoped to this project's Windows-only deployment. Cross-platform deployment would require backend-aware dispatch.
- If brightness still varies after enabling the disable flag, your specific camera / driver may need a different value; consult its documentation.
- Linux / macOS driver behaviour for these properties is not verified in this project (see §3).

## §5 References

**Project documents**
- [`PERFORMANCE.md`](PERFORMANCE.md) — measured performance values and KPIs (separate concern from this document).
- [`config.yaml`](config.yaml) and [`src/hand_tracker/config.py`](src/hand_tracker/config.py) — full reference for every configurable setting (this document covers only a subset that benefits from explicit guidance).

**Code design comments (this document's primary authoritative source)**
- FOURCC ordering & USB bandwidth: design comment inside `_open_device` (`capture.py`).
- Auto-exposure `0.25` semantics: design comment inside `_open_device` (`capture.py`).

**External notes**
- OpenCV official documentation does not standardise value semantics for properties such as `CAP_PROP_FOURCC` ordering or `CAP_PROP_AUTO_EXPOSURE` values; backend-specific behaviour is the responsibility of the underlying drivers.
