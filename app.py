# Made by | Discord Bearman.entp |
import cv2
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# 1. FastAPI 설정
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. 전역 상태 관리
class AIState:
    def __init__(self):
        self.model_name = "yolo11n.pt"  # 가장 가벼운 최신 Nano 모델
        self.model = YOLO(self.model_name)
        self.detections = []
        self.is_running = True

state = AIState()

# 3. API 엔드포인트
@app.get("/status")
async def get_status():
    return {"detected": state.detections, "model": state.model_name}

@app.get("/control/model")
async def change_model(version: str):
    models = {"v8": "yolov8n.pt", "v10": "yolov10n.pt", "v11": "yolo11n.pt"}
    target = models.get(version, "yolo11n.pt")
    if state.model_name != target:
        state.model = YOLO(target)
        state.model_name = target
    return {"status": "success", "model": state.model_name}

# 4. 서버 스레드 함수
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

# 5. 메인 실행 (OpenCV 창은 메인 스레드에서 실행)
if __name__ == "__main__":
    # FastAPI 서버를 백그라운드에서 실행
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 카메라 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 찾을 수 없습니다.")
        exit()

    print("🚀 YOLO 모니터 창이 활성화되었습니다.")
    print("💡 종료하려면 카메라 창에서 'ESC' 키를 누르세요.")

    while state.is_running:
        success, frame = cap.read()
        if not success: continue

        # YOLO 추론
        results = state.model.predict(frame, verbose=False, imgsz=320)
        annotated_frame = results[0].plot()

        # 현재 감지 결과 업데이트
        state.detections = [results[0].names[int(c)] for c in results[0].boxes.cls]

        # OpenCV 창 띄우기 (메인 스레드)
        cv2.imshow("YOLO NANO MONITOR", annotated_frame)
        
        # 창을 맨 앞으로 유지하기 위한 설정 (옵션)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()