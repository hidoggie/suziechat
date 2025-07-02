# main.py
from fastapi.responses import HTMLResponse

import asyncio
import io
import os
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import google.generativeai as genai
from google.cloud import speech, texttospeech
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import requests
import time
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- 1. 최종 설정: 시작 전 이 부분을 자신의 환경에 맞게 수정하세요! ---

# ✨ 1-1. 정보 출처가 될 웹사이트 URL ✨
TARGET_WEBSITE_URL = "https://ardocent.com/"

# ✨ 1-2. 3가지 인증 정보 설정 ✨
GEMINI_API_KEY = "AIzaSyADiixw8TnJllQQq1G2UC9sCEsWalYc2zE" # 여기에 본인의 Gemini API 키를 입력하세요
STT_CREDENTIALS_PATH = "https://guide3.ivyro.net/docentchat/voice-chat-462608-72cee39fbfb5.json"
TTS_CREDENTIALS_PATH = "https://guide3.ivyro.net/docentchat/voice-chat-462608-f8e24b57b208.json"

# ✨ 1-3. Gemini 답변 길이 및 스타일 제어 설정 ✨
MAX_OUTPUT_TOKENS = 1500 # 물리적인 최대 답변 토큰 수 (약 2000자 이내)

# --- 2. 전역 변수 및 객체 ---
app = FastAPI()
WEBSITE_CONTEXT = "아직 웹사이트 정보가 로딩되지 않았습니다." # 전역 컨텍스트 저장 변수
model = None # Gemini 모델 객체 (전역)

# --- 3. 헬퍼 함수들 ---
# ✨✨✨ Selenium을 사용하지 않는 새로운 스크래핑 함수 ✨✨✨
def scrape_website_text(url: str) -> str:
    """requests와 BeautifulSoup을 사용하여 웹사이트 텍스트를 추출합니다."""
    print(f"💻 '{url}'에서 정보 로딩을 시작합니다...")
    try:
        # 구글봇인 것처럼 User-Agent 헤더를 설정하여 차단을 우회
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 200 OK가 아니면 오류 발생

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 불필요한 태그 제거 (선택 사항)
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()

        text = soup.get_text(separator='\n', strip=True)
        print("✅ 정보 로딩 완료!")
        return text
    except requests.RequestException as e:
        print(f"웹사이트를 불러오는 데 실패했습니다: {e}")
        return None



def scrape_with_selenium(url: str) -> str:
    # (이전 답변과 동일한 Selenium 스크래핑 함수)
    print(f"💻 '{url}'에서 정보 로딩을 시작합니다... (최대 10-15초 소요)")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless'); options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url); time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        print("✅ 정보 로딩 완료!")
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"Selenium 스크래핑 중 오류 발생: {e}"); return None
    finally:
        if 'driver' in locals(): driver.quit()

# STT, TTS 함수 (이전 답변과 동일)
def transcribe_audio_stream(audio_bytes: bytes, client: speech.SpeechClient) -> str:
    """WAV 형식의 오디오 바이트를 텍스트로 변환합니다."""
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        # WAV(LINEAR16) 포맷을 명시하고, 샘플링 레이트를 반드시 지정해야 합니다.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000, # 프론트엔드와 일치시킨 값
            language_code="ko-KR",
        )
        response = client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript if response.results else ""
    except Exception as e:
        print(f"STT API 처리 중 오류 발생: {e}")
        return "" # 오류 발생 시 빈 문자열 반환

def synthesize_speech(text, client):
    # (이전과 동일, 단 클라이언트를 인자로 받도록 수정)
    try:
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e: print(f"TTS 오류: {e}"); return b""


# --- 4. 서버 시작 시 실행되는 로직 ---
# ✨✨✨ 루트 경로("/")에 대한 GET 요청 처리 함수 추가 ✨✨✨
@app.get("/", response_class=HTMLResponse)
async def get_root():
    """웹사이트 접속 시 index.html 파일을 반환합니다."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>HTML 파일을 찾을 수 없습니다.</h1><p>static/index.html 경로를 확인해주세요.</p>"

@app.on_event("startup")
async def startup_event():
    """서버가 시작될 때 웹사이트 컨텐츠를 로드하고 Gemini 모델을 초기화합니다."""
    global WEBSITE_CONTEXT, model
    
    # 1. 웹사이트 컨텐츠 스크래핑
#    WEBSITE_CONTEXT = scrape_with_selenium(TARGET_WEBSITE_URL)
    WEBSITE_CONTEXT = scrape_website_text(TARGET_WEBSITE_URL)
    if not WEBSITE_CONTEXT:
        WEBSITE_CONTEXT = "오류: 웹사이트 컨텐츠 로딩에 실패했습니다."
        print(WEBSITE_CONTEXT)

    # 2. Gemini API 설정
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 3. 시스템 명령어 및 생성 설정 정의
    system_instruction = f"""
    당신은 다음 '웹사이트 내용'에 대해서만 답변하는 전문 Q&A 어시스턴트입니다.
    당신의 임무는 사용자의 질문에 대해, 오직 아래 제공된 '웹사이트 내용' 안에서만 정보를 찾아 답변하는 것입니다.
    당신의 내부 지식이나 다른 정보를 절대 사용해서는 안 됩니다.

    [중요 규칙]
    1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
    2. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.
    3. 친절하고 명확한 한국어 말투를 사용해주세요.

    만약 '웹사이트 내용'에 질문에 대한 정보가 없다면, "제가 참고하고 있는 정보 내에서는 답변하기 어렵습니다."라고 솔직하게 답변해야 합니다.

    --- 웹사이트 내용 ---
    {WEBSITE_CONTEXT}
    """
    
    generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
    
    # 4. 전역 모델 객체 초기화
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction=system_instruction,
        generation_config=generation_config
    )
    print("✅ Gemini 모델이 성공적으로 초기화되었습니다.")

# main.py

# --- 5. 웹소켓 엔드포인트 (종료 명령어 감지 기능 추가) ---
# ✨✨✨ 웹소켓 엔드포인트 수정 (강력한 예외 처리 추가) ✨✨✨
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stt_client = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
    tts_client = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
    
    print("클라이언트가 연결되었습니다.")
    try:
        while True:
            # 하나의 대화 턴 전체를 try...except로 감싸서, 실패하더라도 다음 요청을 받을 수 있도록 함
            try:
                audio_bytes = await websocket.receive_bytes()
                
                # 1. STT 처리
                user_text = transcribe_audio_stream(audio_bytes, stt_client)

                if user_text:
                    print(f"👤 사용자: {user_text}")
                    await websocket.send_json({"type": "user_text", "data": user_text})

                    # 2. 종료 명령어 확인
                    if "이제 그만" in user_text.strip():
                        ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                        print(f"🤖 종료 메시지: {ai_text}")
                        await websocket.send_json({"type": "ai_text", "data": ai_text})
                        ai_audio_bytes = synthesize_speech(ai_text, tts_client)
                        if ai_audio_bytes: await websocket.send_bytes(ai_audio_bytes)
                        await asyncio.sleep(1)
                        break # 루프 탈출 및 연결 종료

                    # 3. Gemini 처리
                    print("🤖 Gemini가 답변 생성 중...")
                    response = model.generate_content(user_text)
                    ai_text = response.text
                    
                    print(f"🤖 Gemini: {ai_text}")
                    await websocket.send_json({"type": "ai_text", "data": ai_text})

                    # 4. TTS 처리 및 전송
                    ai_audio_bytes = synthesize_speech(ai_text, tts_client)
                    if ai_audio_bytes:
                        await websocket.send_bytes(ai_audio_bytes)

            except WebSocketDisconnect:
                # 클라이언트가 브라우저를 닫는 등 정상적인 종료
                print("클라이언트가 연결을 닫았습니다.")
                break
            except Exception as e:
                # STT, Gemini, TTS 등 처리 과정에서 발생하는 모든 예외를 처리
                print(f"처리 중 심각한 오류 발생: {e}")
                # 사용자에게 오류 상황을 알려주는 것이 좋음
                error_message = "죄송합니다, 요청을 처리하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                await websocket.send_json({"type": "ai_text", "data": error_message})
                ai_audio_bytes = synthesize_speech(error_message, tts_client)
                if ai_audio_bytes: await websocket.send_bytes(ai_audio_bytes)


    except Exception as e:
        # 웹소켓 연결 자체의 문제
        print(f"웹소켓 연결 오류: {e}")
    finally:
        print("웹소켓 연결을 종료합니다.")