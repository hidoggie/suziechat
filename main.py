# main.py (PDF 및 이미지 처리 0718 오후 12시 28분 )

import asyncio
import os
import fitz  # PyMuPDF 라이브러리
import re
import shutil

from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image # Pillow 라이브러리 import
import io # 바이트 데이터를 메모리에서 다루기 위한 import

import google.generativeai as genai
from google.cloud import speech, texttospeech

from fastapi import Body

# --- 1. 설정 ---
KNOWLEDGE_PDF_PATH = "knowledge.pdf"
IMAGES_DIR = Path(__file__).resolve().parent / "static" / "images"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-e445e48514e2.json"
SAMPLE_RATE = 48000 # 브라우저 MediaRecorder 기본값

# --- 2. 앱 및 전역 클라이언트 초기화 ---
app = FastAPI()
MODEL, STT_CLIENT, TTS_CLIENT, KNOWLEDGE_CONTEXT = None, None, None, None
INITIALIZATION_ERROR = None

PDF_CONTENT = [] 

@app.on_event("startup")
def startup_event():
    """서버가 시작될 때 모든 API 클라이언트와 컨텍스트를 미리 로딩합니다."""
    global MODEL, STT_CLIENT, TTS_CLIENT, PDF_CONTENT, INITIALIZATION_ERROR
    try:
        print("✨  앱 리소스 초기화 (PDF 처리 포함)를 시작합니다...")

        if IMAGES_DIR.exists():
            print(f"📁 기존 이미지 폴더({IMAGES_DIR})를 삭제합니다...")
            shutil.rmtree(IMAGES_DIR)
        print(f"📁 새로운 이미지 폴더({IMAGES_DIR})를 생성합니다...")
        IMAGES_DIR.mkdir()

        pdf_path = Path(__file__).resolve().parent / KNOWLEDGE_PDF_PATH
        if not pdf_path.exists(): raise FileNotFoundError(f"{pdf_path} 파일을 찾을 수 없습니다.")
        
        doc = fitz.open(pdf_path)
#        extracted_content = []
        for page_num, page in enumerate(doc):
            page_data = {"page": page_num, "text": page.get_text()}
            image_files = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

            # --- ✨ Pillow를 사용한 이미지 변환 로직 (핵심 수정) ✨ ---
                try:
                    # 1. 추출한 이미지 바이트를 Pillow 이미지 객체로 변환
                    image_obj = Image.open(io.BytesIO(image_bytes))
                    # 2. 파일 이름을 PNG로 고정
                    image_filename = f"page_{page_num}_img_{img_index}.png"
     #               save_path = IMAGES_DIR / image_filename
                    # 3. PNG 포맷으로 파일 저장 (이 과정에서 변환이 일어남)
                    image_obj.save(IMAGES_DIR / image_filename, "PNG")
                    image_files.append(image_filename)
                except Exception as img_e:
                    print(f"경고: 이미지 처리 실패 (페이지 {page_num}): {img_e}")

                # --- ✨ 여기까지 수정 ---


                page_data["images"] = image_files
                PDF_CONTENT.append(page_data)
                print(f"✅ PDF 처리 완료: {len(doc)} 페이지, {sum(len(p['images']) for p in PDF_CONTENT)}개 이미지 추출")


        # 2. 클라이언트 및 모델 초기화 (시스템 명령어 수정)
        STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        genai.configure(api_key=GEMINI_API_KEY)
        
        system_instruction = f"""
            당신은 전문 Q&A 어시스턴트입니다. 
            사용자의 질문에 대해, 제공되는 컨텍스트(텍스트와 이미지)만을 사용하여 간결하고 정확하게 답변해야 합니다.
            만약 답변이 제공된 특정 이미지와 관련이 있다면, 반드시 답변의 맨 끝에 다음 형식의 태그를 추가해야 합니다: [IMAGE: 이미지파일이름.png]
            당신의 내부 지식이나 다른 정보를 절대 사용해서는 안 됩니다.
        
            ✨✨✨ [중요 규칙] ✨✨✨
            1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
            2. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.
            3. 친절하고 명확한 한국어 말투를 사용해주세요.

           """

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


# ✨✨✨ AR 도슨트 음성 안내를 위한 TTS API 엔드포인트 추가 ✨✨✨
@app.post("/api/tts")
async def text_to_speech_api(text_to_speak: str = Body(..., embed=True)):
    """주어진 텍스트를 음성(MP3) 데이터로 변환하여 반환합니다."""
    if not TTS_CLIENT:
        return {"error": "TTS client not initialized"}
    
    try:
        input_text = texttospeech.SynthesisInput(text=text_to_speak)
        voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        tts_request = texttospeech.SynthesizeSpeechRequest(input=input_text, voice=voice, audio_config=audio_config)
        
        tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
        
        # Base64로 인코딩하여 JSON으로 반환
        import base64
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')
        return {"audio": audio_base64}
        
    except Exception as e:
        print(f"💥 TTS API 엔드포인트 오류: {e}")
        return {"error": str(e)}

# ✨✨✨ PDF 내용을 프론트엔드로 전달하기 위한 API 추가 ✨✨✨
@app.get("/api/pdf-content")
async def get_pdf_content():
    """추출된 PDF의 텍스트와 이미지 정보를 JSON으로 반환합니다."""
    if PDF_CONTENT is None:
        return {"error": "PDF content not loaded"}
    return PDF_CONTENT




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if INITIALIZATION_ERROR:
        await websocket.send_json({"type": "error", "data": f"서버 초기화 실패: {INITIALIZATION_ERROR}"})
        await websocket.close(); return
            
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")
    try:
        while True:
            raw_data = await websocket.receive_json()
            message_type = raw_data.get("type")
            user_input = raw_data.get("data")
            user_text = ""

            if message_type == "audio":
                import base64
                audio_bytes = base64.b64decode(user_input)
                config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hertz=SAMPLE_RATE, language_code="ko-KR")
                audio = speech.RecognitionAudio(content=audio_bytes)
                stt_response = await asyncio.to_thread(STT_CLIENT.recognize, request={"config": config, "audio": audio})
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""
                if user_text: await websocket.send_json({"type": "user_text", "data": user_text})
            elif message_type == "text":
                user_text = user_input

            if user_text:
                if "이제 그만" in user_text.strip(): 
                    ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                    await websocket.send_json({"type": "ai_text", "data": ai_text})
                    break

                keywords = user_text.split()
                relevant_pages = [
                    page for page in PDF_CONTENT 
                    if any(keyword.lower() in page["text"].lower() for keyword in keywords if len(keyword) > 1)
                ]
                if not relevant_pages: # 관련된 페이지가 없으면 전체 텍스트를 컨텍스트로 사용
                    relevant_pages = PDF_CONTENT

                # --- ✨ 2. Gemini에게 전달할 프롬프트 재구성 (텍스트 + '파일명' + 이미지) ✨ ---
                prompt_parts = []
                prompt_parts.append("아래 제공된 텍스트와 이미지를 참고하여 다음 질문에 답변해줘. 이미지에 대해 언급할 때는, 내가 알려준 '이미지 파일명'을 사용하여 `[IMAGE: 파일명]` 태그를 붙여야 해.")
                
                for page in relevant_pages[:3]: # 너무 많은 컨텍스트를 보내지 않도록 3페이지만 참고
                    prompt_parts.append(f"\n--- 참고 텍스트 (페이지 {page['page']+1}) ---\n{page['text']}")
                    if page["images"]:
                        prompt_parts.append("\n--- 이 페이지의 참고 이미지 ---")
                        for img_file in page["images"]:
                            img_path = IMAGES_DIR / img_file
                            if img_path.exists():
                                try:
                                    prompt_parts.append(Image.open(img_path))
                                    # ✨✨✨ 바로 이 부분의 명령어를 수정합니다 ✨✨✨
                                    prompt_parts.append(f"참고: 위 이미지의 파일명은 '{img_file}'입니다. 답변 본문에는 이 파일명을 절대 직접 언급하지 말고, 이미지를 참고했을 경우에만 답변 마지막에 `[IMAGE: {img_file}]` 태그만 추가해주세요.")

                                except Exception as img_e:
                                    print(f"경고: 이미지 파일을 여는 데 실패했습니다 {img_path}: {img_e}")
                
                prompt_parts.append(f"\n--- 질문 ---\n{user_text}")

                # --- ✨ 3. 멀티모달 프롬프트로 Gemini 호출 ✨ ---
                gemini_response = await MODEL.generate_content_async(prompt_parts)
                ai_text_raw = gemini_response.text

                # --- ✨ 4. 답변에서 이미지 태그 파싱 및 분리 전송 ✨ ---
                image_tag_match = re.search(r"\[IMAGE:\s*(page_\d+_img_\d+\.\w+)\]", ai_text_raw)
                ai_text = re.sub(r"\[IMAGE:\s*(.*?)\]", "", ai_text_raw).strip()

                await websocket.send_json({"type": "ai_text", "data": ai_text})

                if image_tag_match:
                    image_filename = image_tag_match.group(1).strip()
                    image_url = f"/static/images/{image_filename}"
                    print(f"🖼️ 클라이언트에게 이미지 표시 요청: {image_url}")
                    await websocket.send_json({"type": "ai_image", "url": image_url})

             
                tts_request = texttospeech.SynthesizeSpeechRequest(input=texttospeech.SynthesisInput(text=ai_text), voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A"), audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3))
                tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
                if tts_response.audio_content: await websocket.send_bytes(tts_response.audio_content)
                if "이제 그만" in user_text.strip(): await asyncio.sleep(1); break

    except WebSocketDisconnect: print(f"🔌 클라이언트 연결 끊어짐: {client_id}")
    except Exception as e: print(f"💥 처리 중 오류 ({client_id}): {e}")
    finally: print(f"🏁 웹소켓 세션 종료: {client_id}")
