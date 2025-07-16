# main.py (최종 완성 및 오류 수정 버전)

import asyncio
import io
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
SAMPLE_RATE = 16000

# --- 2. 전역 변수 및 앱 초기화 ---
app = FastAPI()
KNOWLEDGE_CONTEXT, MODEL = None, None
app_lock = asyncio.Lock()
APP_INITIALIZATION_FAILED = False
INITIALIZATION_ERROR_MESSAGE = "알 수 없는 초기화 오류"

# --- 3. 앱 초기화 함수 (첫 접속 시 실행) ---
async def initialize_app():
    global KNOWLEDGE_CONTEXT, MODEL, APP_INITIALIZATION_FAILED, INITIALIZATION_ERROR_MESSAGE
    
    async with app_lock:
        if KNOWLEDGE_CONTEXT is not None or APP_INITIALIZATION_FAILED: return
        print("✨ 첫 사용자 접속. 핵심 리소스 초기화를 시작합니다...")
        try:
            # 파일 로딩
            file_path = BASE_DIR / KNOWLEDGE_FILE_PATH
            if not file_path.exists(): raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")
            with open(file_path, "r", encoding="utf-8") as f: KNOWLEDGE_CONTEXT = f.read()
        
          # Gemini 모델 초기화
            if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            genai.configure(api_key=GEMINI_API_KEY)

            system_instruction = f"""
            당신은 다음 '제공된 내용'에 대해서만 답변하는 전문 Q&A 어시스턴트입니다. 
            당신의 임무는 사용자의 질문에 대해, 오직 아래 제공된 '제공된 내용' 안에서만 정보를 찾아 답변하는 것입니다.
            당신의 내부 지식이나 다른 정보를 절대 사용해서는 안 됩니다.
        
            ✨✨✨ [중요 규칙] ✨✨✨
            1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
            2. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.
            3. 친절하고 명확한 한국어 말투를 사용해주세요.

            '제공된 내용'에 정보가 없으면, "제가 가진 정보 내에서는 답변하기 어렵습니다."라고 솔직하게 답변해야 합니다. 
            --- 제공된 내용 --- 
            {KNOWLEDGE_CONTEXT}"""

            generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
            MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
            print("✅ Gemini 모델 초기화 성공")
            
        except Exception as e:
            APP_INITIALIZATION_FAILED = True
            INITIALIZATION_ERROR_MESSAGE = f"[{type(e).__name__}] {e}"
            print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR_MESSAGE}")

# --- 4. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
@app.get("/", response_class=FileResponse)
async def read_index(): return FileResponse(BASE_DIR / "static" / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

 # 핵심 리소스(모델, 컨텍스트)가 준비되었는지 확인    
    if MODEL is None: await initialize_app()
    if APP_INITIALIZATION_FAILED:
        await websocket.send_json({"type": "ai_text", "data": f"서버 초기화 실패: {INITIALIZATION_ERROR_MESSAGE}"})
        await websocket.close(); return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    # ✨✨✨ 웹소켓 연결마다 새로운 STT, TTS 클라이언트를 생성 ✨✨✨
    try:
        stt_client = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        tts_client = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        print(f"✅ STT/TTS 클라이언트 생성 완료 ({client_id})")
    except Exception as e:
        print(f"💥 클라이언트 초기화 오류 ({client_id}): {e}")
        await websocket.send_json({"type": "ai_text", "data": "인증 클라이언트 생성에 실패했습니다."})
        await websocket.close(); return
    

    try:
        while True:
            try:
                 # 오디오 수신
                audio_bytes = await websocket.receive_bytes()
                
                # ✨✨✨ STT 설정 수정 (핵심 해결책) ✨✨✨
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # 브라우저 MediaRecorder 포맷
                    sample_rate_hertz=48000, # 대부분 브라우저의 기본 샘플링 레이트
                    language_code="ko-KR"
                )
                audio = speech.RecognitionAudio(content=audio_bytes)
                stt_response = stt_client.recognize(config=config, audio=audio)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""

                if user_text:
                    await websocket.send_json({"type": "user_text", "data": user_text})


                # Gemini 처리 및 종료 명령어 확인
                   
                    if "이제 그만" in user_text.strip():
                        ai_text = "도슨트봇을 종료합니다. 이용해주셔서 감사합니다."
                    else:
                        gemini_response = await MODEL.generate_content_async(user_text)
                        ai_text = gemini_response.text
                    
                    await websocket.send_json({"type": "ai_text", "data": ai_text})


                 # TTS 처리
                   
                    input_text = texttospeech.SynthesisInput(text=ai_text)
                    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                    tts_response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
                    if tts_response.audio_content: await websocket.send_bytes(tts_response.audio_content)

                    if "이제 그만" in user_text.strip(): 
                        await asyncio.sleep(1); break
                    
            except WebSocketDisconnect: 
                print(f"🔌 클라이언트 연결 끊어짐: {client_id}"); break
            except Exception as e:
                print(f"💥 처리 중 오류 ({client_id}): {e}")
                await websocket.send_json({"type": "ai_text", "data": "요청 처리 중 오류가 발생했습니다."})
    
    finally: print(f"🏁 웹소켓 세션 종료: {client_id}")