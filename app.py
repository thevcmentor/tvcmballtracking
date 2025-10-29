from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2, numpy as np, requests, tempfile, os, shutil

app = FastAPI(title="TVCM Ball Tracking API")

# Lazy-load YOLO model (faster startup)
yolo_model = None
def get_model():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # small, fast model
    return yolo_model


class VideoRequest(BaseModel):
    video_url: str


def download_video(url: str) -> str:
    """Download a video from URL into a temp file."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    try:
        r = requests.get(url, stream=True, timeout=60, headers=headers)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download video: {e}")

    ctype = r.headers.get("Content-Type", "").lower()
    if "video" not in ctype and "octet-stream" not in ctype:
        raise HTTPException(status_code=400, detail=f"URL did not return a video file (Content-Type: {ctype})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name


def analyze_local_file(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count == 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Video file has 0 frames or invalid format")

    model = get_model()
    traj = []

    for _ in range(min(frame_count, 300)):
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

    speed_kph = 0.0
    if len(traj) > 2:
        diffs = [
            ((traj[i + 1][0] - traj[i][0]) ** 2 + (traj[i + 1][1] - traj[i][1]) ** 2) ** 0.5
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


@app.get("/")
def root():
    return {"ok": True, "msg": "TVCM Ball Tracking API running!"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze_video(request: VideoRequest):
    """Analyze a cricket video via URL."""
    path = download_video(request.video_url)
    try:
        return analyze_local_file(path)
    finally:
        try:
            os.remove(path)
        except:
            pass


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Analyze a locally uploaded video file."""
    if not file.content_type.startswith("video/"):
        return JSONResponse(status_code=400, content={"error": "File must be a video."})

    tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = analyze_local_file(tmp_path)
        return {"ok": True, "source": "upload", **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
