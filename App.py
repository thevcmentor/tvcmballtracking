from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2, numpy as np, requests, tempfile, os
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

app = FastAPI(title="TVCM Ball Tracking API")

# Load YOLO model (you can replace this with your own fine-tuned model later)
model = YOLO("yolov8n.pt")

class VideoRequest(BaseModel):
    video_url: str

def download_video(url):
    """Download video from a public URL to a temporary local file."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Cannot download video.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

@app.post("/analyze")
def analyze_video(request: VideoRequest):
    """Analyze cricket video and return trajectory and speed data."""
    video_path = download_video(request.video_url)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trajectory = []

    # Set up a simple Kalman filter for ball tracking
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])
    kf.P *= 1000
    kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])

    for i in range(frame_count):
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            kf.predict()
            kf.update(np.array([cx, cy]))
            trajectory.append((int(kf.x[0]), int(kf.x[1])))

    cap.release()
    os.remove(video_path)

    # Rough speed estimation (just a placeholder)
    if len(trajectory) > 2:
        dist_px = np.mean([np.hypot(trajectory[i+1][0]-trajectory[i][0],
                                    trajectory[i+1][1]-trajectory[i][1])
                           for i in range(len(trajectory)-1)])
        speed_px_per_s = dist_px * fps
        speed_kph = round(speed_px_per_s * 0.02, 2)  # rough scale
    else:
        speed_kph = 0

    return {
        "frames_analyzed": frame_count,
        "fps": fps,
        "estimated_speed_kph": speed_kph,
        "trajectory_points": trajectory[:20]
    }

@app.get("/")
def root():
    return {"message": "TVCM Ball Tracking API running!"}
