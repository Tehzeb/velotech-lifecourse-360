"""
VeloTech LifeCourse-360 - Optional OpenCV smartphone bio-signal & pediatric visual pre-screener.

WELLNESS MOTIVATION ONLY. Nothing here diagnoses anything. The output nudges asymptomatic people
to attend a screening visit and is never an input to the clinical risk score.

Modules
-------
1. Synthetic facial video: realistic green-channel pulse injected into a rendered face (noise,
   motion jitter, skin-tone options) so the pipeline always works offline at $0.
2. Contactless rPPG extractor: Haar face / forehead ROI -> mean green trace -> detrend ->
   FFT band-pass 0.7-3.0 Hz -> heart-rate peak, SNR, pulse-interval variability proxy, stress index.
3. Webcam still analysis (`st.camera_input`): a single photo cannot carry a pulse, so the still is
   used for REAL face detection / lighting / sharpness checks, and a simulated pulse is injected into
   the real pixels so the full pipeline can be shown. Real rPPG requires video; this is stated in the UI.
4. Pediatric visual-milestone checker: age-banded parent checklist + animated gaze-tracking stimulus.
5. Technology-neutral wearable / IoT adapter (simulated), "hardware welcome".
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd

DISCLAIMER = ("Wellness indicator only. Not a diagnosis, not used in the clinical risk score. "
              "Final clinical decisions rest with licensed physicians.")
SKIN_TONES_BGR = {"light": (150, 175, 225), "medium": (110, 150, 205), "deep": (60, 95, 140)}


# --------------------------------------------------------------------------------------
# 1. SYNTHETIC FACIAL VIDEO
# --------------------------------------------------------------------------------------
def _base_face(size: int, tone: str) -> tuple[np.ndarray, np.ndarray]:
    """Clean studio-style portrait (soft backdrop, shoulders, neck, shaded head, hair).

    Returns (float32 BGR image, boolean mask of the head region that carries the pulse)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size // 2
    vignette = 1 - 0.20 * (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.78) ** 2)
    grad = 214 - 46 * (yy / size)
    img = np.stack([(grad - 3) * vignette, grad * vignette, (grad + 3) * vignette], axis=-1).astype(np.float32)

    def blend(img_: np.ndarray, mask_u8: np.ndarray, colour) -> np.ndarray:
        a = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0), 0.9)[..., None]
        return img_ * (1 - a) + np.asarray(colour, np.float32) * a

    skin = np.array(SKIN_TONES_BGR[tone], np.float32)
    shoulders = np.zeros((size, size), np.uint8)
    cv2.ellipse(shoulders, (cx, int(size * 1.04)), (int(size * 0.47), int(size * 0.31)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    img = blend(img, shoulders, (112, 98, 88))                       # muted slate clothing
    neck = np.zeros((size, size), np.uint8)
    cv2.rectangle(neck, (cx - int(size * 0.08), int(size * 0.62)), (cx + int(size * 0.08), int(size * 0.86)), 255, -1, cv2.LINE_AA)
    img = blend(img, neck, skin * 0.80)
    head = np.zeros((size, size), np.uint8)
    axes = (int(size * 0.30), int(size * 0.40))
    cv2.ellipse(head, (cx, size // 2), axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    r = np.sqrt(((xx - (cx - 0.05 * size)) / (0.30 * size)) ** 2 + ((yy - size * 0.44) / (0.40 * size)) ** 2)
    shade = np.clip(1.10 - 0.26 * r, 0.82, 1.10)[..., None]
    head_rgb = skin[None, None, :] * shade
    a_head = cv2.GaussianBlur(head.astype(np.float32) / 255.0, (0, 0), 0.9)[..., None]
    img = img * (1 - a_head) + head_rgb * a_head
    ears = np.zeros((size, size), np.uint8)
    for sx in (-1, 1):
        cv2.ellipse(ears, (cx + sx * int(size * 0.30), int(size * 0.50)), (int(size * 0.028), int(size * 0.06)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    img = blend(img, ears, skin * 0.9)
    hair = np.zeros((size, size), np.uint8)
    cv2.ellipse(hair, (cx, int(size * 0.49)), (int(size * 0.318), int(size * 0.418)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(hair, (cx, int(size * 0.41)), (int(size * 0.262), int(size * 0.215)), 0, 0, 360, 0, -1, cv2.LINE_AA)  # forehead / hairline
    hair[int(size * 0.34):, :] = 0
    img = blend(img, hair, (38, 34, 32))
    return img.astype(np.float32), head.astype(bool)


def pulse_waveform(t: np.ndarray, hr_bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Quasi-periodic pulse with a dicrotic harmonic, respiratory modulation and mild rate variability."""
    f = hr_bpm / 60.0
    ph = rng.uniform(0, 6.28)
    # slow (0.1 Hz) and respiratory (0.25 Hz) modulation of the instantaneous rate, a few percent each
    warp = t - 0.02 * np.cos(2 * np.pi * 0.10 * t + ph) / (2 * np.pi * 0.10) - 0.01 * np.cos(2 * np.pi * 0.25 * t) / (2 * np.pi * 0.25)
    phase = 2 * np.pi * f * warp
    resp = 1 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    return resp * (np.sin(phase) + 0.30 * np.sin(2 * phase + 0.6))


def generate_session(hr_bpm: float = 72.0, seconds: float = 10.0, fps: int = 30, tone: str = "medium",
                     motion_px: float = 1.5, noise_sd: float = 3.0, amplitude: float = 1.6,
                     size: int = 192, seed: int = 11) -> list[np.ndarray]:
    """Render a synthetic face video with an embedded green-channel pulse."""
    rng = np.random.default_rng(seed)
    base, mask = _base_face(size, tone)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    pulse = amplitude * pulse_waveform(t, hr_bpm, rng)
    frames = []
    for i in range(n):
        f = base.copy()
        f[..., 1][mask] += pulse[i]  # green channel carries the haemoglobin absorption change
        f += rng.normal(0, noise_sd, f.shape)
        dx, dy = rng.normal(0, motion_px, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


# --------------------------------------------------------------------------------------
# 2. rPPG EXTRACTION
# --------------------------------------------------------------------------------------
_CASCADE = None          # None = not tried yet, False = unavailable, else a cv2.CascadeClassifier
FACE_METHOD = "none"     # which detector produced the last result: "haar", "skin-colour" or "none"


def _cascade():
    """Haar face cascade, if this OpenCV build has one (OpenCV 5 removed CascadeClassifier)."""
    global _CASCADE
    if _CASCADE is None:
        try:
            if not hasattr(cv2, "CascadeClassifier"):
                raise AttributeError("CascadeClassifier not available in this OpenCV build")
            clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            _CASCADE = clf if not clf.empty() else False
        except Exception:
            _CASCADE = False
    return _CASCADE or None


def _skin_face_box(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Model-free fallback: largest skin-coloured blob (YCrCb thresholds), trimmed to a face-like aspect."""
    try:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.03 * bgr.shape[0] * bgr.shape[1]:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return int(x), int(y), int(w), int(min(h, 1.4 * w))
    except Exception:
        return None


def detect_face_ex(bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, str]:
    """Return (face box or None, method). Uses Haar when available, otherwise the skin-colour fallback."""
    cas = _cascade()
    if cas is not None:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(40, bgr.shape[1] // 8),) * 2)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return (int(x), int(y), int(w), int(h)), "haar"
        except Exception:
            pass
    box = _skin_face_box(bgr)
    return (box, "skin-colour") if box is not None else (None, "none")


def detect_face(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    return detect_face_ex(bgr)[0]


def forehead_roi(shape: tuple[int, int], face: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    """Forehead region (x, y, w, h). Falls back to the central upper region when no face is detected."""
    H, W = shape[:2]
    if face is not None:
        x, y, w, h = face
        return int(x + 0.28 * w), int(y + 0.10 * h), int(0.44 * w), int(0.18 * h)
    return int(0.35 * W), int(0.26 * H), int(0.30 * W), int(0.12 * H)


def _bandpass_fft(sig: np.ndarray, fps: float, lo: float = 0.7, hi: float = 3.0) -> np.ndarray:
    n = len(sig)
    spec = np.fft.rfft(sig * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1 / fps)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n)


def extract_rppg(frames: list[np.ndarray], fps: float = 30.0, simulated_pulse: bool = False) -> dict:
    """Contactless heart-rate and vascular-stress proxy from a sequence of BGR frames."""
    face, method = detect_face_ex(frames[0])
    x, y, w, h = forehead_roi(frames[0].shape, face)
    g = np.array([float(f[y:y + h, x:x + w, 1].mean()) for f in frames])
    # detrend with a ~1 s moving average (reflect-padded), then z-score
    win = max(3, int(fps))
    padded = np.pad(g, win // 2 + 1, mode="reflect")
    trend = np.convolve(padded, np.ones(win) / win, mode="same")[win // 2 + 1: win // 2 + 1 + len(g)]
    det = g - trend
    det = (det - det.mean()) / (det.std() + 1e-9)
    filt = _bandpass_fft(det, fps)
    # high-resolution spectrum via zero padding
    n_fft = 8 * len(filt)
    spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / fps)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    pk = np.argmax(np.where(band, spec, 0))
    f_peak = float(freqs[pk])
    hr = f_peak * 60.0
    near = band & (np.abs(freqs - f_peak) <= 0.12)
    far = band & ~near
    snr_db = float(10 * np.log10((spec[near].sum() + 1e-12) / (spec[far].mean() * max(near.sum(), 1) + 1e-12)))
    quality = "Good" if snr_db >= 6 else ("Fair" if snr_db >= 2 else "Poor")
    # pulse intervals from filtered-signal peaks (parabolic sub-frame refinement + outlier rejection)
    # beats are detected on a signal band-passed tightly around the detected rate, so the dicrotic
    # harmonic of the pulse cannot create a second peak per beat
    narrow = _bandpass_fft(det, fps, max(0.7, f_peak - 0.35), min(3.0, f_peak + 0.35))
    min_gap = int(0.6 * fps / max(f_peak, 0.7) * 1.0)
    peaks = [i for i in range(1, len(narrow) - 1) if narrow[i] > narrow[i - 1] and narrow[i] >= narrow[i + 1] and narrow[i] > 0.15 * narrow.std()]
    keep: list[int] = []
    for p in peaks:
        if not keep or p - keep[-1] >= min_gap:
            keep.append(p)
        elif narrow[p] > narrow[keep[-1]]:
            keep[-1] = p
    times = []
    for i in keep:
        d = narrow[i - 1] - 2 * narrow[i] + narrow[i + 1]
        off = 0.5 * (narrow[i - 1] - narrow[i + 1]) / d if abs(d) > 1e-9 else 0.0
        times.append((i + float(np.clip(off, -0.5, 0.5))) / fps)
    ibi = np.diff(times) * 1000.0 if len(times) > 2 else np.array([])
    if len(ibi) >= 4:
        med = float(np.median(ibi))
        good = (ibi > 0.7 * med) & (ibi < 1.3 * med)
        pairs = [(ibi[i], ibi[i + 1]) for i in range(len(ibi) - 1) if good[i] and good[i + 1]]
        rmssd = float(np.sqrt(np.mean([(b_ - a_) ** 2 for a_, b_ in pairs]))) if len(pairs) >= 3 else float("nan")
    else:
        rmssd = float("nan")
    stress = 50 * np.clip((hr - 65) / 35, 0, 1) + (50 * np.clip((50 - rmssd) / 45, 0, 1) if not np.isnan(rmssd) else 25)
    return {
        "hr_bpm": round(hr, 1), "snr_db": round(snr_db, 1), "quality": quality, "rmssd_ms": None if np.isnan(rmssd) else round(rmssd, 1),
        "stress_index": int(round(float(stress))), "face_detected": face is not None, "face_method": method, "roi": (x, y, w, h),
        "n_frames": len(frames), "fps": fps, "t": np.arange(len(g)) / fps, "raw": g, "filtered": filt,
        "freqs_hz": freqs[band], "power": spec[band] / (spec[band].max() + 1e-12), "peaks": keep,
        "simulated_pulse": simulated_pulse, "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------------------
# 3. WEBCAM STILL HANDLING
# --------------------------------------------------------------------------------------
def decode_image(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def analyze_still_frame(bgr: np.ndarray) -> dict:
    """REAL image checks on the webcam photo: face found, lighting, sharpness, ROI colour."""
    face = detect_face(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    advice = []
    if face is None:
        advice.append("No face found: centre your face, look at the camera and remove strong backlight.")
    if brightness < 70:
        advice.append("Too dark: face a window or lamp.")
    elif brightness > 200:
        advice.append("Too bright: move away from direct light.")
    if sharp < 60:
        advice.append("Image is soft: hold the phone steady.")
    x, y, w, h = forehead_roi(bgr.shape, face)
    annotated = bgr.copy()
    if face is not None:
        fx, fy, fw, fh = face
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (92, 122, 31), 2)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (235, 99, 37), 2)
    return {"face_found": face is not None, "brightness": round(brightness, 1), "sharpness": round(sharp, 1),
            "roi_green_mean": round(float(bgr[y:y + h, x:x + w, 1].mean()), 1), "advice": advice or ["Capture quality looks good."],
            "annotated": annotated, "face": face}


def frames_from_still(bgr: np.ndarray, hr_bpm: float = 74.0, seconds: float = 10.0, fps: int = 30, seed: int = 5) -> list[np.ndarray]:
    """Inject a SIMULATED pulse into the real still's face pixels (a photo cannot contain a pulse)."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(bgr, (240, int(240 * bgr.shape[0] / bgr.shape[1]))).astype(np.float32)
    face = detect_face(img.astype(np.uint8))
    mask = np.zeros(img.shape[:2], bool)
    if face is not None:
        x, y, w, h = face
        mask[y:y + h, x:x + w] = True
    else:
        cv2.ellipse(mask.view(np.uint8), (img.shape[1] // 2, img.shape[0] // 2), (img.shape[1] // 4, img.shape[0] // 3), 0, 0, 360, 1, -1)
        mask = mask.astype(bool)
    n = int(seconds * fps)
    pulse = 1.6 * pulse_waveform(np.arange(n) / fps, hr_bpm, rng)
    out = []
    for i in range(n):
        f = img.copy()
        f[..., 1][mask] += pulse[i]
        f += rng.normal(0, 2.5, f.shape)
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


def to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


def annotate_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = roi
    cv2.rectangle(out, (x, y), (x + w, y + h), (235, 99, 37), 2)
    return out


# --------------------------------------------------------------------------------------
# 4. PEDIATRIC VISUAL-MILESTONE CHECK
# --------------------------------------------------------------------------------------
MILESTONES: dict[int, list[str]] = {
    12: ["Looks toward you when you say their name", "Waves or points to show something", "Babbles with changing sounds", "Looks where you point",
         "Pulls up to stand or cruises along furniture", "Plays simple games like peek-a-boo"],
    18: ["Points to show you things they find interesting", "Says several single words", "Walks without help", "Copies simple tasks such as sweeping",
         "Looks at your face when you talk", "Plays simple pretend, like feeding a doll"],
    24: ["Puts two words together", "Follows simple two-step instructions", "Kicks a ball", "Points to body parts when asked",
         "Plays alongside other children", "Shows interest in other children"],
    36: ["Speaks in short sentences that others mostly understand", "Climbs well", "Plays make-believe", "Turns book pages one at a time",
         "Shows affection without prompting", "Copies a circle"],
    48: ["Tells a simple story", "Hops on one foot", "Names some colours", "Draws a person with two to four parts",
         "Takes turns in games", "Understands same and different"],
}
VISUAL_ITEMS = ["Eyes follow the moving dot smoothly from left to right", "Both eyes look straight, with no obvious turn",
                "Shows interest in the dot for about five seconds"]


def milestone_bracket(age_months: float) -> int:
    keys = sorted(MILESTONES)
    return max([k for k in keys if k <= max(age_months, keys[0])], default=keys[0])


def milestone_items(age_months: float) -> list[dict]:
    b = milestone_bracket(age_months)
    items = [{"id": f"m{b}-{i}", "text": t, "type": "milestone"} for i, t in enumerate(MILESTONES[b])]
    items += [{"id": f"v-{i}", "text": t, "type": "visual"} for i, t in enumerate(VISUAL_ITEMS)]
    return items


def score_milestones(answers: dict[str, bool], items: list[dict]) -> dict:
    yes = sum(bool(answers.get(i["id"], False)) for i in items)
    frac = yes / max(1, len(items))
    if frac >= 0.85:
        status, tone, msg = "On track", "ok", "Lovely progress. Keep the Well-Child Visit booked so growth and development stay on track."
    elif frac >= 0.60:
        status, tone, msg = "Worth mentioning at the visit", "info", "A few things to talk through. The Well-Child Visit is the right place to ask, and it is quick."
    else:
        status, tone, msg = "Book the visit soon", "warn", "Several items are not yet seen. Please book the Well-Child Visit this month so the team can take a closer look together."
    return {"yes": yes, "total": len(items), "fraction": frac, "status": status, "tone": tone, "message": msg, "disclaimer": DISCLAIMER}


def render_tracking_stimulus(t: float, width: int = 560, height: int = 180, dark: bool = False) -> bytes:
    """One frame of a left-right smooth-pursuit stimulus (parent plays it by moving the slider)."""
    bg, track, dot, core = ((24, 29, 34), (58, 66, 75), (160, 184, 212), (24, 29, 34)) if dark else ((234, 228, 216), (192, 205, 213), (84, 96, 115), (255, 255, 255))
    img = np.full((height, width, 3), bg, np.uint8)
    x = int(40 + (width - 80) * (0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))))
    cv2.line(img, (40, height // 2), (width - 40, height // 2), track, 2, cv2.LINE_AA)
    cv2.circle(img, (x, height // 2), 24, dot, -1, cv2.LINE_AA)
    cv2.circle(img, (x, height // 2), 10, core, -1, cv2.LINE_AA)
    return to_png_bytes(img)


# --------------------------------------------------------------------------------------
# 5. TECHNOLOGY-NEUTRAL WEARABLE / IoT ADAPTER
# --------------------------------------------------------------------------------------
class SensorAdapter(ABC):
    """Implement `read_window` for any device (BLE heart-rate strap, smartwatch API, clinic kiosk...)."""
    name = "generic"

    @abstractmethod
    def read_window(self, minutes: int = 30) -> pd.DataFrame:
        """Return columns: minute, hr_bpm, rmssd_ms, spo2_pct, skin_temp_c, steps."""


class SimulatedWearable(SensorAdapter):
    name = "Simulated smartwatch (demo)"

    def __init__(self, person: dict, seed: int = 21):
        self.person = person
        tail = str(person["id"]).split("-")[-1]
        offset = int(tail) if tail.isdigit() else sum(ord(ch) for ch in tail) % 1000  # custom ids such as "P-CUSTOM" are not numeric
        self.rng = np.random.default_rng(seed + offset)

    def read_window(self, minutes: int = 30) -> pd.DataFrame:
        f = self.person["features"]
        heat = float(f.get("heat_exposure") or 0.1)
        base_hr = 68 + 0.1 * (self.person["age_years"] - 40) + 10 * heat + (6 if f.get("smoker") else 0)
        m = np.arange(minutes)
        hr = base_hr + 4 * np.sin(m / 6) + self.rng.normal(0, 2, minutes)
        rmssd = np.clip(48 - 14 * heat - 0.2 * max(self.person["age_years"] - 30, 0) + self.rng.normal(0, 4, minutes), 8, 90)
        return pd.DataFrame({
            "minute": m, "hr_bpm": hr.round(1), "rmssd_ms": rmssd.round(1),
            "spo2_pct": np.clip(97.5 - 1.2 * heat + self.rng.normal(0, 0.4, minutes), 90, 100).round(1),
            "skin_temp_c": (33.2 + 1.8 * heat + self.rng.normal(0, 0.15, minutes)).round(2),
            "steps": np.clip(self.rng.poisson(35 - 15 * heat, minutes), 0, None),
        })


def wearable_summary(df: pd.DataFrame) -> dict:
    s = {"mean_hr": round(float(df["hr_bpm"].mean()), 1), "mean_rmssd": round(float(df["rmssd_ms"].mean()), 1),
         "min_spo2": float(df["spo2_pct"].min()), "mean_skin_temp": round(float(df["skin_temp_c"].mean()), 2), "steps": int(df["steps"].sum())}
    s["flag"] = "Elevated heat strain pattern: prioritise hydration and the heat-stress kidney panel." if s["mean_skin_temp"] > 34.3 and s["mean_rmssd"] < 38 else "No unusual pattern in this window."
    return s


# --------------------------------------------------------------------------------------
# 6. MOTIVATIONAL NUDGES (multilingual, wellness framing)
# --------------------------------------------------------------------------------------
NUDGES = {
    "English": {"Good": "Your pulse check looks steady. A quick screening visit now helps keep it that way.",
                "Fair": "Your pulse check was a little noisy. A short clinic visit can give you a clear picture.",
                "Poor": "We could not get a clear reading, which is common. A clinic visit gives a reliable check."},
    "Arabic": {"Good": "قياس النبض لديك يبدو مستقراً. زيارة فحص قصيرة الآن تساعدك على الحفاظ على ذلك.",
               "Fair": "كانت قراءة النبض غير واضحة قليلاً. زيارة قصيرة للعيادة تعطيك صورة أوضح.",
               "Poor": "لم نتمكن من الحصول على قراءة واضحة، وهذا أمر شائع. زيارة العيادة تعطي فحصاً موثوقاً."},
    "Urdu": {"Good": "آپ کی نبض کی جانچ مستحکم لگ رہی ہے۔ ابھی ایک مختصر اسکریننگ وزٹ اسے برقرار رکھنے میں مدد دے گا۔",
             "Fair": "نبض کی ریڈنگ تھوڑی غیر واضح تھی۔ کلینک کا مختصر وزٹ آپ کو واضح تصویر دے گا۔",
             "Poor": "ہم واضح ریڈنگ نہیں لے سکے، جو عام بات ہے۔ کلینک کا وزٹ قابلِ اعتماد جانچ دیتا ہے۔"},
}


def motivation_message(quality: str, language: str = "English") -> str:
    lang = "Arabic" if language.startswith("Arabic") else language
    return NUDGES.get(lang, NUDGES["English"]).get(quality, NUDGES["English"]["Fair"])


if __name__ == "__main__":  # python -m src.opencv_biosignal
    for hr in (58, 72, 96, 120):
        r = extract_rppg(generate_session(hr_bpm=hr), 30)
        print(f"true {hr:>3} bpm -> estimated {r['hr_bpm']:>5} bpm | SNR {r['snr_db']:>5} dB | {r['quality']} | RMSSD {r['rmssd_ms']} | stress {r['stress_index']}")
    print(wearable_summary(SimulatedWearable({"id": "P-004", "age_years": 38, "features": {"heat_exposure": 0.95, "smoker": 1}}).read_window()))
