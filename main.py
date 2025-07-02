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
# Render 환경 변수와 Secret Files를 사용합니다.
TARGET_WEBSITE_URL = "https://blog.google/technology/ai/google-gemini-ai/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-f8e24b57b208.json"

# --- 2. 전역 변수 및 상태 관리 ---
app = FastAPI()

# 앱 리소스를 담을 전역 변수들. 처음에는 비어있습니다.
WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT = None, None, None, None
# 여러 요청이 동시에 초기화를 시도하는 것을 방지하는 Lock
app_lock = asyncio.Lock()
# 초기화 실패 시 상태를 기억하는 플래그
APP_INITIALIZATION_FAILED = False

# --- 3. 헬퍼 함수 ---
async def scrape_website_text_async(url: str) -> str:
    """httpx와 selectolax를 사용하여 비동기적으로 웹사이트 텍스트를 추출합니다."""
    print(f"💻 '{url}'에서 정보 로딩을 시작합니다...")
    try:
        async with httpx.AsyncClient() as client:
            # User-Agent를 설정하여 봇 차단을 우회 시도
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = await client.get(url, headers=headers, timeout=20.0)
            response.raise_for_status()

        html = HTMLParser(response.text)
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

# --- 4. 서버 시작 이벤트 대신, 첫 접속 시 초기화를 수행하는 함수 ---
async def initialize_app():
    """첫 사용자 접속 시, 무거운 초기화 작업을 수행합니다."""
    global WEBSITE_CONTEXT, MODEL, STT_CLIENT, TTS_CLIENT, APP_INITIALIZATION_FAILED
    
    # Lock을 사용하여 여러 요청이 동시에 들어와도 초기화는 한 번만 실행되도록 보장
    async with app_lock:
        # 이중 확인: Lock을 기다리는 동안 다른 요청이 이미 초기화를 완료했을 수 있음
        if WEBSITE_CONTEXT is not None or APP_INITIALIZATION_FAILED:
            return

        print("✨ 첫 사용자 접속. 앱 리소스 초기화를 시작합니다...")
        try:
            # 1. 웹사이트 스크래핑
            WEBSITE_CONTEXT = await scrape_website_text_async(TARGET_WEBSITE_URL)
            if not WEBSITE_CONTEXT:
                raise Exception("웹사이트 스크래핑 결과가 비어있습니다.")

            # 2. STT/TTS 클라이언트 초기화
            print("STT/TTS 클라이언트 초기화 중...")
            STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
            TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
            print("✅ STT/TTS 클라이언트 초기화 성공!")

            # 3. Gemini API 설정 및 모델 초기화
            print("Gemini 모델 초기화 중...")
            if not GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
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
            print("✅ Gemini 모델 초기화 성공!")

            print("🎉 모든 리소스 초기화 완료. 챗봇이 준비되었습니다.")

        except Exception as e:
            APP_INITIALIZATION_FAILED = True
            print("="*60)
            print(f"💥 FATAL: 앱 초기화 중 심각한 오류 발생! 💥")
            print(f"오류 원인: {e}")
            print("="*60)

# --- 5. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 앱 리소스가 준비되었는지 확인하고, 안 되었다면 초기화 시도
    if MODEL is None:
        await initialize_app()
    
    # 초기화 최종 실패 시, 클라이언트에게 알리고 연결 종료
    if APP_INITIALIZATION_FAILED:
        await websocket.send_json({"type": "ai_text", "data": "서버 초기화에 실패했습니다. 관리자에게 문의하세요."})
        await websocket.close(); return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    try:
        while True:
            # 하나의 대화 턴(Turn)
            try:
                # 1. 오디오 수신 및 STT
                audio_bytes = await websocket.receive_bytes()
                config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code="ko-KR")
                audio = speech.RecognitionAudio(content=audio_bytes)
                stt_response = STT_CLIENT.recognize(config=config, audio=audio)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""

                if user_text:
                    print(f"👤 사용자 ({client_id}): {user_text}")
                    await websocket.send_json({"type": "user_text", "data": user_text})
                    
                    # 2. 종료 명령어 또는 Gemini 응답 생성
                    if "이제 그만" in user_text.strip():
                        ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                        print(f"🤖 종료 메시지 전송 ({client_id})")
                    else:
                        print(f"🤖 Gemini에게 답변 요청 ({client_id})...")
                        gemini_response = await MODEL.generate_content_async(user_text)
                        ai_text = gemini_response.text
                    
                    print(f"🤖 Gemini 답변 ({client_id}): {ai_text}")
                    await websocket.send_json({"type": "ai_text", "data": ai_text})

                    # 3. TTS로 음성 합성 및 전송
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
        print(f"💥 웹소켓 연결 오류: {e}")
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")