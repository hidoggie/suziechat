# main.py (최종 안정화 및 지연 로딩 버전)

import asyncio
import io
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import httpx # requests 대신 사용
from selectolax.parser import HTMLParser # BeautifulSoup 대신 사용

import google.generativeai as genai
from google.cloud import speech, texttospeech

# --- 1. 설정 ---
TARGET_WEBSITE_URL = "https://blog.google/technology/ai/google-gemini-ai/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Render 환경변수 사용 권장
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-72cee39fbfb5.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-f8e24b57b208.json"

MAX_OUTPUT_TOKENS = 1500

# --- 2. 전역 변수 및 상태 관리 ---
app = FastAPI()
# 초기화 전에는 None으로 설정
WEBSITE_CONTEXT = None
MODEL = None
STT_CLIENT = None
TTS_CLIENT = None
# 여러 사용자가 동시에 초기화를 시도하는 것을 방지하기 위한 Lock
app_lock = asyncio.Lock()

# --- 3. 헬퍼 함수들 ---
async def scrape_website_text_async(url: str) -> str:
    """httpx와 selectolax를 사용하여 비동기적으로 웹사이트 텍스트를 추출합니다."""
    print(f"💻 '{url}'에서 정보 로딩을 시작합니다...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()

        html = HTMLParser(response.text)
        
        # article 태그가 있다면 그 안의 텍스트만 가져오고, 없다면 body 전체 텍스트를 가져옴
        article_node = html.css_first('article')
        if article_node:
            text = article_node.text(separator='\n', strip=True)
        else:
            text = html.body.text(separator='\n', strip=True)
            
        print("✅ 정보 로딩 완료!")
        return text
    except Exception as e:
        print(f"💥 웹 스크래핑 중 오류 발생: {e}")
        return None

async def initialize_app():
    """첫 사용자 접속 시, 무거운 초기화 작업을 수행하는 함수"""
    global WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT
    
    # Lock을 사용하여 한 번에 하나의 초기화만 진행되도록 보장
    async with app_lock:
        # 다른 요청이 이미 초기화를 완료했는지 다시 한번 확인
        if WEBSITE_CONTEXT is None:
            print("✨ 첫 사용자 접속. 앱 초기화를 시작합니다...")
            
            WEBSITE_CONTEXT = await scrape_website_text_async(TARGET_WEBSITE_URL)
            if not WEBSITE_CONTEXT:
                WEBSITE_CONTEXT = "오류: 웹사이트 컨텐츠 로딩에 실패했습니다."
            
            genai.configure(api_key=GEMINI_API_KEY)
            system_instruction = f"""
            당신은 다음 '웹사이트 내용'에 대해서만 답변하는 전문 Q&A 어시스턴트입니다.
            [중요 규칙]
            1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
            2. '웹사이트 내용'에 정보가 없으면, "제가 참고하고 있는 정보 내에서는 답변을 찾기 어렵습니다."라고 솔직하게 답변해야 합니다.
            --- 웹사이트 내용 ---
            {WEBSITE_CONTEXT}
            """
            generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
            MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
            STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
            TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
            print("✅ 모든 초기화 완료. 챗봇 서비스 준비 완료.")

# --- 4. FastAPI 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html 파일을 찾을 수 없습니다.</h1>"

# ✨✨✨ 웹소켓 엔드포인트 수정 (세분화된 오류 처리 및 로깅 강화) ✨✨✨
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    # --- 각 API 클라이언트를 여기서 한 번만 초기화 ---
    stt_client, tts_client = None, None
    try:
        stt_client = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        tts_client = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        print(f"✅ API 클라이언트 초기화 성공 ({client_id})")
    except Exception as e:
        print(f"💥 FATAL: 인증 파일(Secret Files) 로딩 실패! ({client_id}): {e}")
        print("Render.com 대시보드의 'Secret Files' 경로와 내용이 올바른지 확인하세요.")
        await websocket.send_json({"type": "ai_text", "data": "서버 인증 설정에 문제가 발생했습니다. 관리자에게 문의하세요."})
        await websocket.close()
        return

    # 앱 초기화가 완료되었는지 최종 확인
    if MODEL is None:
        print(f"💥 FATAL: Gemini 모델이 초기화되지 않았음! ({client_id})")
        await websocket.send_json({"type": "ai_text", "data": "AI 모델 로딩에 실패했습니다. 서버 로그를 확인해주세요."})
        await websocket.close()
        return

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            user_text = ""
            ai_text = ""
            
            # --- 1. STT 처리 블록 ---
            try:
                config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code="ko-KR")
                audio = speech.RecognitionAudio(content=audio_bytes)
                response = stt_client.recognize(config=config, audio=audio)
                user_text = response.results[0].alternatives[0].transcript if response.results else ""
            except Exception as e:
                print(f"💥 STT API 오류 ({client_id}): {e}")
                await websocket.send_json({"type": "ai_text", "data": "음성 인식 중 오류가 발생했습니다."})
                continue # 다음 요청을 위해 현재 턴을 건너뜀

            if user_text:
                print(f"👤 사용자 ({client_id}): {user_text}")
                await websocket.send_json({"type": "user_text", "data": user_text})

                # --- 2. Gemini 처리 블록 ---
                try:
                    if "이제 그만" in user_text.strip():
                        ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                        print(f"🤖 종료 메시지 전송 ({client_id})")
                    else:
                        print(f"🤖 Gemini에게 답변 요청 ({client_id})...")
                        response = await MODEL.generate_content_async(user_text)
                        ai_text = response.text
                except Exception as e:
                    print(f"💥 Gemini API 오류 ({client_id}): {e}")
                    ai_text = "AI 답변 생성 중 오류가 발생했습니다."
                
                print(f"🤖 Gemini 답변 ({client_id}): {ai_text}")
                await websocket.send_json({"type": "ai_text", "data": ai_text})

                # --- 3. TTS 처리 블록 ---
                try:
                    input_text = texttospeech.SynthesisInput(text=ai_text)
                    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                    tts_response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
                    if tts_response.audio_content:
                        await websocket.send_bytes(tts_response.audio_content)
                except Exception as e:
                    print(f"💥 TTS API 오류 ({client_id}): {e}")
                    # TTS만 실패한 경우, 텍스트 응답은 이미 전송되었으므로 별도 처리는 생략
                
                if "이제 그만" in user_text.strip():
                    await asyncio.sleep(1)
                    break # 종료

    except WebSocketDisconnect:
        print(f"🔌 클라이언트 연결 끊어짐: {client_id}")
    except Exception as e:
        print(f"💥 웹소켓의 심각한 오류 ({client_id}): {e}")
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")