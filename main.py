# main.py (최종 진단 및 안정화 버전)

import asyncio
import io
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import httpx
from selectolax.parser import HTMLParser

import google.generativeai as genai
from google.cloud import speech, texttospeech

# --- 1. 설정 ---
TARGET_WEBSITE_URL = "https://blog.google/technology/ai/google-gemini-ai/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-f8e24b57b208.json"

# --- 2. 전역 변수 및 상태 관리 ---
app = FastAPI()
WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT = None, None, None, None
app_lock = asyncio.Lock()
# ✨✨✨ 초기화 실패 여부를 기억할 플래그 추가 ✨✨✨
APP_INITIALIZATION_FAILED = False

# --- 3. 헬퍼 함수들 ---
async def scrape_website_text_async(url: str):
    # (이전 답변과 동일)
    pass

# ✨✨✨ 강력한 예외 처리가 추가된 초기화 함수 ✨✨✨
async def initialize_app():
    global WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT, APP_INITIALIZATION_FAILED
    async with app_lock:
        if WEBSITE_CONTEXT is None and not APP_INITIALIZATION_FAILED:
            print("✨ 첫 사용자 접속. 앱 초기화를 시작합니다...")
            try:
                # 1. 웹사이트 스크래핑
                WEBSITE_CONTEXT = await scrape_website_text_async(TARGET_WEBSITE_URL)
                if not WEBSITE_CONTEXT:
                    raise Exception("웹사이트 스크래핑 결과가 비어있습니다.")

                # 2. STT 클라이언트 초기화
                print("STT 클라이언트 초기화 중...")
                STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
                print("✅ STT 클라이언트 초기화 성공!")

                # 3. TTS 클라이언트 초기화
                print("TTS 클라이언트 초기화 중...")
                TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
                print("✅ TTS 클라이언트 초기화 성공!")
                
                # 4. Gemini API 설정 및 모델 초기화
                print("Gemini 모델 초기화 중...")
                if not GEMINI_API_KEY:
                    raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
                genai.configure(api_key=GEMINI_API_KEY)
                
                system_instruction = f""" ... """ # (이전과 동일한 시스템 명령어)
                generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
                MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
                # 모델이 실제로 존재하는지 간단한 테스트
                if MODEL is None: raise Exception("Gemini 모델 객체 생성에 실패했습니다.")
                print("✅ Gemini 모델 초기화 성공!")

                print("🎉 모든 초기화 완료. 챗봇 서비스 준비 완료.")

            except Exception as e:
                # ✨ 초기화 과정 중 어디서든 오류가 나면 여기로 들어옴
                APP_INITIALIZATION_FAILED = True
                print("="*60)
                print(f"💥 FATAL: 앱 초기화 중 심각한 오류 발생! 💥")
                print(f"오류 원인: {e}")
                print("="*60)
                # 오류의 원인이 될 수 있는 설정들을 다시 확인해주세요. (아래 체크리스트 참고)

# --- 4. FastAPI 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    # (이전 답변과 동일)
    pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 앱 초기화가 실패했거나, 아직 진행 중인 경우 처리
    if MODEL is None or STT_CLIENT is None or TTS_CLIENT is None:
        if not app_lock.locked():
            await initialize_app() # 첫 접속 시 초기화 시도
        else:
            await app_lock.wait() # 다른 요청이 초기화 중이면 대기

    # 초기화 최종 실패 시, 클라이언트에게 알리고 연결 종료
    if APP_INITIALIZATION_FAILED:
        error_msg = "서버 초기화에 실패했습니다. 관리자에게 문의하세요."
        await websocket.send_json({"type": "ai_text", "data": error_msg})
        await websocket.close()
        return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    try:
        # ... (이하 while 루프와 그 안의 로직은 이전 답변과 동일하게 유지)
        pass 
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")

# (기존 코드에서 사용했던 transcribe_audio_stream, synthesize_speech, get_root 함수 등은
# 이제 websocket_endpoint 안에서 직접 client 객체를 사용하므로 필요에 따라 정리하거나 유지할 수 있습니다.
# 위 예시에서는 websocket_endpoint 안에서 직접 API를 호출하는 방식으로 간소화했습니다.)