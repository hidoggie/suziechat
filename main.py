# main.py (0716 10시 40분 완성본)

import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import google.generativeai as genai
from google.cloud import speech, texttospeech

# --- 1. 설정 ---
KNOWLEDGE_FILE_PATH = "knowledge.txt"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-e445e48514e2.json"
SAMPLE_RATE = 48000 # 브라우저 MediaRecorder 기본값

# --- 2. 앱 및 전역 클라이언트 초기화 ---
app = FastAPI()
MODEL, STT_CLIENT, TTS_CLIENT, KNOWLEDGE_CONTEXT = None, None, None, None
INITIALIZATION_ERROR = None

@app.on_event("startup")
def startup_event():
    """서버가 시작될 때 모든 API 클라이언트와 컨텍스트를 미리 로딩합니다."""
    global MODEL, STT_CLIENT, TTS_CLIENT, KNOWLEDGE_CONTEXT, INITIALIZATION_ERROR
    try:
        print("✨ 앱 리소스 초기화를 시작합니다...")
        
        # 파일 로딩
        file_path = Path(__file__).resolve().parent / KNOWLEDGE_FILE_PATH
        if not file_path.exists(): raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")
        with open(file_path, "r", encoding="utf-8") as f: KNOWLEDGE_CONTEXT = f.read()
        
        # 클라이언트 초기화
        STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        
        # Gemini 모델 초기화
        if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=GEMINI_API_KEY)
        system_instruction = f"""당신은 다음 '제공된 내용'에 대해서만 답변하는 전문 Q&A 어시스턴트입니다. 모든 답변은 간결하게, 핵심만 요약해서 생성해야 합니다. '제공된 내용'에 정보가 없으면, "제가 가진 정보 내에서는 답변하기 어렵습니다."라고 솔직하게 답변해야 합니다. --- 제공된 내용 --- {KNOWLEDGE_CONTEXT}"""
        generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
        MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
        
        print("🎉 모든 리소스 초기화 완료. 챗봇이 준비되었습니다.")
    except Exception as e:
        INITIALIZATION_ERROR = f"[{type(e).__name__}] {e}"
        print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR}")

# --- 3. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
@app.get("/", response_class=FileResponse)
async def read_index(): return FileResponse(BASE_DIR / "static" / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if INITIALIZATION_ERROR:
        await websocket.send_json({"type": "ai_text", "data": f"서버 초기화 실패: {INITIALIZATION_ERROR}"})
        await websocket.close(); return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")
    try:
        while True:
            try:
                audio_bytes = await websocket.receive_bytes()
                
                # STT 호출 (안전한 비동기 방식)
                config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hertz=SAMPLE_RATE, language_code="ko-KR")
                audio = speech.RecognitionAudio(content=audio_bytes)
                stt_response = await asyncio.to_thread(STT_CLIENT.recognize, config=config, audio=audio)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""

                if user_text:
                    await websocket.send_json({"type": "user_text", "data": user_text})
                    
                    if "이제 그만" in user_text.strip():
                        ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                    else:
                        gemini_response = await MODEL.generate_content_async(user_text)
                        ai_text = gemini_response.text
                    
                    await websocket.send_json({"type": "ai_text", "data": ai_text})
                    
                    # ✨✨✨ TTS 호출을 안전한 비동기 방식으로 변경 (핵심 수정) ✨✨✨
                    input_text = texttospeech.SynthesisInput(text=ai_text)
                    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                    tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, input=input_text, voice=voice, audio_config=audio_config)
                    
                    if tts_response.audio_content: await websocket.send_bytes(tts_response.audio_content)
                    if "이제 그만" in user_text.strip(): await asyncio.sleep(1); break

            except WebSocketDisconnect: print(f"🔌 클라이언트 연결 끊어짐: {client_id}"); break
            except Exception as e:
                print(f"💥 처리 중 오류 ({client_id}): {e}")
                await websocket.send_json({"type": "ai_text", "data": "요청 처리 중 오류가 발생했습니다."})
    except Exception as e: print(f"💥 웹소켓의 최상위 오류: {e}")
    finally: print(f"🏁 웹소켓 세션 종료: {client_id}")