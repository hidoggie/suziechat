# main.py (최종 진단용)

import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio

from google.cloud import speech

# --- 1. 설정 ---
# Render Secret Files에 등록된 STT 인증서 경로
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
SAMPLE_RATE = 48000 # 브라우저 기본값

# --- 2. 앱 및 STT 클라이언트 초기화 ---
app = FastAPI()
# 앱이 시작될 때 STT 클라이언트를 한 번만 생성합니다.
try:
    stt_client = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
    print("✅ STT 클라이언트 초기화 성공!")
except Exception as e:
    stt_client = None
    print(f"💥 FATAL: STT 클라이언트 초기화 실패! 원인: {e}")

# --- 3. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    if stt_client is None:
        await websocket.send_text("[서버 오류] STT 클라이언트가 초기화되지 않았습니다.")
        await websocket.close()
        return

    try:
        while True:
            # 클라이언트로부터 오디오 데이터를 받습니다.
            audio_bytes = await websocket.receive_bytes()
            print(f"🎧 오디오 수신됨 (사이즈: {len(audio_bytes)} bytes)")

            try:
                # STT 처리
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                    sample_rate_hertz=SAMPLE_RATE,
                    language_code="ko-KR"
                )
                audio = speech.RecognitionAudio(content=audio_bytes)
                
                # 비동기 환경에서 동기 함수를 안전하게 실행
                response = await asyncio.to_thread(stt_client.recognize, config=config, audio=audio)
                
                user_text = response.results[0].alternatives[0].transcript if response.results else "[인식된 내용 없음]"
                print(f"🗣️ STT 결과: {user_text}")

                # 인식된 텍스트를 클라이언트로 다시 보냄
                await websocket.send_text(user_text)

            except Exception as e:
                print(f"💥 STT 처리 중 오류: {e}")
                await websocket.send_text(f"[STT 오류] {e}")

    except WebSocketDisconnect:
        print(f"🔌 클라이언트 연결 끊어짐: {client_id}")
    except Exception as e:
        print(f"💥 웹소켓 오류: {e}")
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")