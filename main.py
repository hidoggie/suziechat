# main.py (0812 오후 12시 08분 최종 완성 및 통합 버전)

import asyncio
import os
import re
import json
#--- import time ---
import shutil
import base64
import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import fitz  # PyMuPDF
import numpy as np # 벡터 계산을 위해 추가

import google.generativeai as genai
from google.cloud import speech, texttospeech


# --- 1. 설정 ---
KNOWLEDGE_PDF_PATH = "knowledge.pdf"
IMAGES_DIR = Path(__file__).resolve().parent / "static" / "images"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-e445e48514e2.json"
SAMPLE_RATE = 48000
EMBEDDING_MODEL = "models/text-embedding-004"

# --- 2. 전역 변수 및 앱 초기화 ---
app = FastAPI()
PDF_CONTENT = []
MODEL, STT_CLIENT, TTS_CLIENT = None, None, None
INITIALIZATION_ERROR = None

# --- 3. Lifespan을 이용한 안정적인 앱 초기화 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, STT_CLIENT, TTS_CLIENT, PDF_CONTENT, INITIALIZATION_ERROR, EMBEDDING_MODEL, KNOWLEDGE_CONTEXT
    print("✨ 앱 리소스 초기화를 시작합니다...")

    try:
        if IMAGES_DIR.exists(): shutil.rmtree(IMAGES_DIR)
        # PDF 처리 로직
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        pdf_path = Path(__file__).resolve().parent / KNOWLEDGE_PDF_PATH
        if not pdf_path.exists(): raise FileNotFoundError(f"{pdf_path} 파일을 찾을 수 없습니다.")
        
        doc = fitz.open(pdf_path)
        content_list = []

        for page_num, page in enumerate(doc):
            page = doc.load_page(page_num) # 페이지를 번호로 명시하여 로드
            
            page_text = page.get_text("text")
            page_images = page.get_images(full=True)

            page_data = {"page": page_num, "text": page_text, "images": []}
           
            
            for img_index, img in enumerate(page_images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image_obj = Image.open(io.BytesIO(image_bytes))
                    image_filename = f"page_{page_num}_img_{img_index}.png"
                    image_obj.save(IMAGES_DIR / image_filename, "PNG")
                    page_data["images"].append(image_filename)
                except Exception as img_e:
                    print(f"경고: 이미지 처리 실패 (페이지 {page_num}): {img_e}")

            content_list.append(page_data)

        PDF_CONTENT = content_list
        KNOWLEDGE_CONTEXT = "\n\n".join([page['text'] for page in content_list]) # 전체 텍스트 컨텍스트 생성

        print(f"✅ PDF 처리 완료: {len(doc)} 페이지, {sum(len(p['images']) for p in PDF_CONTENT)}개 이미지 추출")

        # PDF 텍스트에 대한 임베딩 벡터 생성
        texts_to_embed = [page['text'] for page in PDF_CONTENT if page['text'].strip()]
        if texts_to_embed:
            embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=texts_to_embed, task_type="RETRIEVAL_DOCUMENT")
            text_index = 0
            for i, page_data in enumerate(PDF_CONTENT):
                if page_data['text'].strip():
                    page_data['embedding'] = embedding_response['embedding'][text_index]
                    text_index += 1
            print(f"✅ {len(texts_to_embed)}개 텍스트에 대한 임베딩 생성 완료.")


        # API 클라이언트 초기화
        STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=GEMINI_API_KEY)
        
        system_instruction = f"""
            당신은 전문 도슨트입니다. 
            사용자의 질문에 대해, 반드시 아래 제공된 '지식 베이스' 내용만을 근거로 해야 합니다.
            당신의 일반 지식을 사용해서는 안 됩니다. '지식 베이스'에 내용이 없다면 "제가 가진 정보로는 답변하기 어렵습니다."라고 솔직하게 말해야 합니다.
                        
            ✨✨✨ [중요 규칙] ✨✨✨
                1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
                2. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.
                3. 만약 사용자가 "어떻게 생겼어", "보여줘", "그림", "사진" 등 시각적인 정보를 명시적으로 요구하고, 해당 질문과 가장 관련 있는 이미지가 있다면 답변 끝에 `[SHOW_IMAGE:이미지파일명]` 태그를 붙여주세요.
                4. 일반적인 질문에는 절대 `[SHOW_IMAGE]` 태그를 붙이면 안 됩니다.

         --- 지식 베이스 ---
        {KNOWLEDGE_CONTEXT}
               
        """
        
        generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
        MODEL = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction, generation_config=generation_config)
        
        print("🎉 모든 리소스 초기화 완료. 챗봇이 준비되었습니다.")

    except Exception as e:
        INITIALIZATION_ERROR = f"[{type(e).__name__}] {e}"
        print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR}")

    yield # 애플리케이션 실행
    
    print("👋 서버를 종료합니다.")

app = FastAPI(lifespan=lifespan)

# --- 4. 헬퍼 함수 ---
def find_best_page_by_vector(query_text: str):
    """주어진 텍스트와 가장 유사한 PDF 페이지를 시맨틱 검색으로 찾습니다."""
    if not query_text or not any('embedding' in p for p in PDF_CONTENT): return None
    
    query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query_text, task_type="RETRIEVAL_QUERY")['embedding']
    
    pdf_embeddings = np.array([page['embedding'] for page in PDF_CONTENT if 'embedding' in page])
    query_vector = np.array(query_embedding)
    
    dot_products = np.dot(pdf_embeddings, query_vector)
    norms = np.linalg.norm(pdf_embeddings, axis=1) * np.linalg.norm(query_vector)
    similarity_scores = dot_products / norms
    
    best_match_index = np.argmax(similarity_scores)
    return PDF_CONTENT[best_match_index] if similarity_scores[best_match_index] > 0.6 else None


# --- 4. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=FileResponse)
async def read_index(): return FileResponse(BASE_DIR / "static" / "index.html")

# ✨✨✨ /ar 경로 추가 ✨✨✨
@app.get("/ar", response_class=FileResponse)
async def read_ar_page():
    return FileResponse(BASE_DIR / "static" / "ar.html")

@app.get("/api/pdf-content")
async def get_pdf_content():
    if INITIALIZATION_ERROR or not PDF_CONTENT:
        # 초기화 실패 또는 컨텐츠가 없는 경우 명확한 오류 반환
        return JSONResponse(
            status_code=500, 
            content={"error": "PDF content not loaded or initialization failed."}
        )
    
    # ✨ 딕셔너리가 아닌, 원본 배열 구조를 그대로 반환하도록 수정
    return JSONResponse(content=PDF_CONTENT)

@app.post("/api/tts")
async def text_to_speech_api(payload: dict = Body(...)):
    # ✨ 1. 음성 안내 버튼 오류 수정을 위한 최종 TTS API 코드
    text_to_speak = payload.get("text_to_speak")
    if not text_to_speak:
        raise HTTPException(status_code=400, detail="text_to_speak 필드가 필요합니다.")
    if not TTS_CLIENT:
        raise HTTPException(status_code=500, detail="TTS client not initialized")
    try:
        tts_request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=text_to_speak),
            voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A"),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')
        return {"audio": audio_base64}
    except Exception as e:
        print(f"💥 TTS API 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

# ✨✨✨ AR 이미지 인식을 위한 새로운 AI 쿼리 엔드포인트 추가 ✨✨✨
@app.post("/api/ar-query")
async def ar_query(image_name: str = Body(..., embed=True)):
    """
    인식된 이미지 이름을 받아, 해당 이미지의 컨텍스트를 바탕으로
    Gemini가 동적으로 생성한 설명과 TTS 오디오를 반환합니다.
    """
    if MODEL is None:
        return {"error": "Model not initialized"}

    # PDF 컨텐츠에서 이미지 이름으로 해당 페이지의 텍스트 찾기
    context_text = ""
    for page in PDF_CONTENT:
        if image_name in page["images"]:
            context_text = page["text"]
            break
    
    if not context_text:
        return {"error": "Context not found for this image"}

    try:
        # Gemini에게 동적 설명을 생성하도록 요청하는 프롬프트
        prompt = f"""
        당신은 전문 박물관 도슨트입니다.
        아래는 한 전시물에 대한 기본 정보입니다. 이 정보를 바탕으로, 실제 관람객에게 설명하듯이 생생하고 흥미로운 해설을 1~2 문장으로 간결하게 생성해주세요.

        --- 기본 정보 ---
        {context_text}
        """
        
        gemini_response = await MODEL.generate_content_async(prompt)
        ai_text = gemini_response.text.strip()

        # 생성된 설명으로 TTS 오디오 생성
        tts_request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=ai_text),
            voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A"),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
        
        import base64
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')

        return {
            "text": ai_text,
            "audio": audio_base64
        }

    except Exception as e:
        print(f"💥 AR 쿼리 처리 오류: {e}")
        return {"error": str(e)}    

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
            message_type = raw_data.get("type"); 
            user_input = raw_data.get("data")
            user_text = ""

            if message_type == "audio":
                import base64
                audio_bytes = base64.b64decode(user_input)
                stt_request = speech.RecognizeRequest(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hertz=SAMPLE_RATE, language_code="ko-KR"), audio=speech.RecognitionAudio(content=audio_bytes))
                stt_response = await asyncio.to_thread(STT_CLIENT.recognize, request=stt_request)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""
                if user_text: await websocket.send_json({"type": "user_text", "data": user_text})
            
            elif message_type == "text":
                user_text = user_input

            if user_text:
                print(f"👤 사용자 ({client_id}): {user_text}")

                if "이제 그만" in user_text.strip():
                    ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                    await websocket.send_json({"type": "ai_text", "data": ai_text})
                    # ... (TTS 및 종료 로직)
                    break
                
                 # --- ✨ 2. 백엔드가 직접 이미지 표시 의도 파악 (핵심 수정) ---
                image_keywords = ["보여줘", "사진", "그림", "이미지", "생김새", "모습"]
                show_image_intent = any(keyword in user_text for keyword in image_keywords)

                keywords = re.findall(r'[\w가-힣]{2,}', user_text)
                best_match = {"score": 0, "page_data": None}
                for page in PDF_CONTENT:
                    score = sum(1 for keyword in keywords if keyword.lower() in page["text"].lower())
                    if score > best_match["score"]:
                        best_match["score"] = score
                        best_match["page_data"] = page
                
                context_text = "\n\n".join([p['text'] for p in PDF_CONTENT]) # 기본은 전체 컨텍스트
 
                context_images = []

                if best_match["page_data"]:
                    context_text = best_match["page_data"]["text"]
                    context_images = best_match["page_data"]["images"]
                    print(f"✅ 컨텍스트 찾음: 페이지 {best_match['page_data']['page']} (점수: {best_match['score']})")
 

                # --- 3. AI에게 이미지 표시 여부를 스스로 결정하도록 지시 ---
                # 컨텍스트 이미지 목록은 이제 필요 없으므로 프롬프트에서 제외합니다.
            
                prompt = f"""아래 컨텍스트를 바탕으로 사용자의 질문에 간결하게 답변해줘.
                
                --- 컨텍스트 ---
                {context_text}
                
                --- 질문 ---
                {user_text}
                
                               
                """
                
                gemini_response = await MODEL.generate_content_async(prompt)
                ai_text = gemini_response.text.strip()
                await websocket.send_json({"type": "ai_text", "data": ai_text})

                 # --- 5. 의도(Intent)에 따라 이미지 표시 ---
                if show_image_intent and best_match["page_data"] and best_match["page_data"]["images"]:
                    # 검색된 가장 정확한 페이지의 첫 번째 이미지를 전송
                    image_filename = best_match["page_data"]["images"][0]
                    image_url = f"/static/images/{image_filename}"
                    print(f"🖼️ 클라이언트에게 이미지 표시 요청: {image_url}")
                    await websocket.send_json({"type": "ai_image", "data": {"url": image_url}})

              
                tts_request = texttospeech.SynthesizeSpeechRequest(input=texttospeech.SynthesisInput(text=ai_text), voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Wavenet-A"), audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3))
                tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
                if tts_response.audio_content: await websocket.send_bytes(tts_response.audio_content)

    except WebSocketDisconnect: print(f"🔌 클라이언트 연결 끊어짐: {client_id}")
    except Exception as e: print(f"💥 처리 중 오류 ({client_id}): {e}")
    finally: print(f"🏁 웹소켓 세션 종료: {client_id}")



    # ✨✨✨ 4. 새로운 이미지 인식 API 엔드포인트 추가 ✨✨✨
@app.post("/api/recognize-image")
async def recognize_image(payload: dict = Body(...)):
    user_image_b64 = payload.get("image")
    if not user_image_b64:
        raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")

    try:
        user_image_bytes = base64.b64decode(user_image_b64.split(',')[1])
        user_image = Image.open(io.BytesIO(user_image_bytes))

        # 1. AI에게 이미지 묘사 요청 (빠른 작업)
        describe_prompt = ["이 이미지에 보이는 사물이나 장소의 이름을 포함하여 간결하게 묘사해줘.", user_image]
        describe_response = await MODEL.generate_content_async(describe_prompt)
        image_description = describe_response.text.strip()
        
        # 2. 묘사를 바탕으로 가장 관련있는 페이지를 시맨틱 검색
        best_page = find_best_page_by_vector(image_description)
        if not best_page:
            return {"status": "no_match", "description": "죄송합니다, 관련된 정보를 찾을 수 없습니다."}
        
        context_text = best_page["text"]
        
        # 3. 찾은 텍스트를 바탕으로 최종 해설 생성 (불필요한 태그 규칙 제거)
        summarize_prompt = f"""당신은 전문 도슨트입니다. 아래 텍스트를 바탕으로, 실제 관람객에게 설명하듯이 생생하고 흥미로운 해설을 생성해주세요. 인사말 없이 바로 설명으로 시작해야 합니다. --- 원본 텍스트 --- {context_text}"""
        summarize_response = await MODEL.generate_content_async(summarize_prompt)
        final_description = summarize_response.text.strip()
        
        return {"status": "success", "description": final_description}

    
    except Exception as e:
        print(f"💥 이미지 인식/요약 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))