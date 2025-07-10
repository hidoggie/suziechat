# main.py (최종 완성본)

import asyncio
import io
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import httpx
from selectolax.parser import HTMLParser
import google.generativeai as genai
from google.cloud import speech, texttospeech

# --- 1. 설정 ---
TARGET_WEBSITE_URL = "https://ardocent.com/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-e445e48514e2.json"
SAMPLE_RATE = 48000 # 브라우저 MediaRecorder의 기본 샘플링 레이트와 일치

# --- 2. 전역 변수 및 상태 관리 ---
app = FastAPI()
WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT = None, None, None, None
app_lock = asyncio.Lock()
APP_INITIALIZATION_FAILED = False
# ✨✨✨ 실패 시의 상세 오류 메시지를 저장할 전역 변수 추가 ✨✨✨
INITIALIZATION_ERROR_MESSAGE = "알 수 없는 초기화 오류"


# --- 3. 헬퍼 함수 ---
async def scrape_website_text_async(url: str) -> str:
    print(f"💻 '{url}'에서 정보 로딩을 시작합니다...")
    try:
        async with httpx.AsyncClient() as client:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = await client.get(url, headers=headers, timeout=20.0)
            response.raise_for_status()
        html = HTMLParser(response.text)
        article_node = html.css_first('article')
        text = article_node.text(separator='\n', strip=True) if article_node else html.body.text(separator='\n', strip=True)
        print("✅ 정보 로딩 완료!")
        return text
    except Exception as e:
        print(f"💥 웹 스크래핑 중 오류 발생: {e}"); return None

async def initialize_app():
    global WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT, APP_INITIALIZATION_FAILED, INITIALIZATION_ERROR_MESSAGE
    
    async with app_lock:
        if WEBSITE_CONTEXT is not None or APP_INITIALIZATION_FAILED:
            return

        print("✨ 첫 사용자 접속. 앱 리소스 초기화를 시작합니다...")
        try:
            # 각 단계별로 명확하게 실행
            print("1/4: 웹사이트 컨텐츠 로딩...")
            WEBSITE_CONTEXT = await scrape_website_text_async(TARGET_WEBSITE_URL)
            if not WEBSITE_CONTEXT: raise Exception("웹사이트 스크래핑 결과가 비어있습니다.")
            print("✅ 웹사이트 로딩 성공")

            print("2/4: STT 클라이언트 초기화...")
            STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
            print("✅ STT 클라이언트 성공")

            print("3/4: TTS 클라이언트 초기화...")
            TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
            print("✅ TTS 클라이언트 성공")
            
            print("4/4: Gemini 모델 초기화...")
            if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            genai.configure(api_key=GEMINI_API_KEY)
            
            system_instruction = f""" ... """ # (시스템 명령어 내용은 이전과 동일)
            generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
            MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
            print("✅ Gemini 모델 성공")

            print("🎉 모든 리소스 초기화 완료. 챗봇이 준비되었습니다.")

        except Exception as e:
            # ✨ 오류 발생 시, 실패 플래그와 함께 상세 메시지를 전역 변수에 저장
            APP_INITIALIZATION_FAILED = True
            error_type = type(e).__name__
            INITIALIZATION_ERROR_MESSAGE = f"[{error_type}] {e}"
            print("="*60)
            print(f"💥 FATAL: 앱 초기화 중 심각한 오류 발생! 💥")
            print(f"상세 원인: {INITIALIZATION_ERROR_MESSAGE}")
            print("="*60)


# --- 4. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if MODEL is None:
        await initialize_app()
    
    # 초기화 최종 실패 시, 상세 오류 메시지를 클라이언트에게 전송
    if APP_INITIALIZATION_FAILED:
        error_msg = f"서버 초기화 실패: {INITIALIZATION_ERROR_MESSAGE}"
        await websocket.send_json({"type": "ai_text", "data": error_msg})
        await websocket.close()
        return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    # ... (이하 while 루프와 그 안의 로직은 이전 답변과 동일하게 유지)
    try:
        while True:
            # ...
            pass
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")
    try:
        while True:
            try:
                audio_bytes = await websocket.receive_bytes()
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                    sample_rate_hertz=SAMPLE_RATE, # 브라우저의 MediaRecorder가 주로 사용하는 샘플링 레이트
                    language_code="ko-KR"
                )
                audio = speech.RecognitionAudio(content=audio_bytes)
                stt_response = STT_CLIENT.recognize(config=config, audio=audio)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""

                if user_text:
                    await websocket.send_json({"type": "user_text", "data": user_text})
                    if "이제 그만" in user_text.strip():
                        ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                    else:
                        gemini_response = await MODEL.generate_content_async(user_text)
                        ai_text = gemini_response.text
                    
                    await websocket.send_json({"type": "ai_text", "data": ai_text})
                    
                    input_text = texttospeech.SynthesisInput(text=ai_text)
                    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                    tts_response = TTS_CLIENT.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
                    if tts_response.audio_content:
                        await websocket.send_bytes(tts_response.audio_content)
                    
                    if "이제 그만" in user_text.strip():
                        await asyncio.sleep(1); break
            except WebSocketDisconnect:
                print(f"🔌 클라이언트 연결 끊어짐: {client_id}"); break
            except Exception as e:
                print(f"💥 처리 중 오류 ({client_id}): {e}")
                await websocket.send_json({"type": "ai_text", "data": "요청 처리 중 오류가 발생했습니다."})
    except Exception as e:
        print(f"💥 웹소켓의 최상위 오류: {e}")
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")