from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2, numpy as np, requests, tempfile, os

app = FastAPI(title="TVCM Ball Tracking API")

# Lazy-load YOLO model (faster startup)
yolo_model = None
def get_model():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # small, fast default model
    return yolo_model


class VideoRequest(BaseModel):
    video_url: str


def download_video(url: str) -> str:
    """Download a video from URL into a temp file."""
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Cannot download video.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name


@app.get("/")
def root():
    return {"ok": True, "msg": "TVCM Ball Tracking API running!"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze_video(request: VideoRequest):
    """Analyze a cricket video and estimate basic ball trajectory."""
    path = download_video(request.video_url)
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        model = get_model()
        traj = []

        for _ in range(min(frame_count, 300)):  # limit for demo
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame, conf=0.4, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes
                confs = b.conf.cpu().numpy()
                i = int(confs.argmax())
                x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i][:4]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                traj.append([cx, cy])

        cap.release()

        # Naive velocity estimate
        speed_kph = 0.0
        if len(traj) > 2:
            diffs = [
                ((traj[i + 1][0] - traj[i][0]) ** 2 +
                 (traj[i + 1][1] - traj[i][1]) ** 2) ** 0.5
                for i in range(len(traj) - 1)
            ]
            px_per_s = (sum(diffs) / max(1, len(diffs))) * fps
            speed_kph = round(px_per_s * 0.02, 2)

        return {
            "frames_analyzed": frame_count,
            "fps": fps,
            "estimated_speed_kph": speed_kph,
            "points_sampled": len(traj),
            "trajectory_points": traj[:50],
        }
    finally:
        try:
            os.remove(path)
        except:
            pass
