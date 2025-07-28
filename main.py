# main.py (0722 오후 4시 32분 최종 완성 및 통합 버전)

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

import google.generativeai as genai
from google.cloud import speech, texttospeech

import numpy as np # 벡터 계산을 위해 추가


# --- 1. 설정 ---
KNOWLEDGE_PDF_PATH = "knowledge.pdf"
IMAGES_DIR = Path(__file__).resolve().parent / "static" / "images"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_OUTPUT_TOKENS = 1500
STT_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-412b0459f610.json"
TTS_CREDENTIALS_PATH = "/etc/secrets/voice-chat-462608-e445e48514e2.json"
SAMPLE_RATE = 48000

# --- 2. 전역 변수 및 앱 초기화 ---
PDF_CONTENT = []
MODEL, STT_CLIENT, TTS_CLIENT = None, None, None
INITIALIZATION_ERROR = None

EMBEDDING_MODEL = "models/text-embedding-004"

# --- 3. Lifespan을 이용한 안정적인 앱 초기화 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, STT_CLIENT, TTS_CLIENT, PDF_CONTENT, INITIALIZATION_ERROR,  EMBEDDING_MODEL
    print("✨ 앱 리소스 초기화를 시작합니다...")

    try:
      #  if IMAGES_DIR.exists(): shutil.rmtree(IMAGES_DIR)
        # PDF 처리 로직
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ 이미지 저장 폴더 확인/생성 완료: {IMAGES_DIR}")

        pdf_path = Path(__file__).resolve().parent / KNOWLEDGE_PDF_PATH
        if not pdf_path.exists(): raise FileNotFoundError(f"{pdf_path} 파일을 찾을 수 없습니다.")
        
        doc = fitz.open(pdf_path)
        content_list = []

        for page_num in range(len(doc)):
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
        print(f"✅ PDF 처리 완료: {len(doc)} 페이지, {sum(len(p['images']) for p in PDF_CONTENT)}개 이미지 추출")

        # ✨✨✨ 디버깅을 위한 파일 저장 코드 추가 ✨✨✨
        try:
            debug_path = Path(__file__).resolve().parent / "debug_content.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(PDF_CONTENT, f, ensure_ascii=False, indent=2)
            print(f"✅ 디버깅 파일 저장 완료: {debug_path}")
        except Exception as e:
            print(f"💥 디버깅 파일 저장 실패: {e}")


        # ✨✨✨ PDF 텍스트에 대한 임베딩 벡터 생성 (핵심 추가) ✨✨✨
        print("🤖 PDF 컨텐츠에 대한 임베딩 벡터를 생성합니다...")
        texts_to_embed = [page['text'] for page in PDF_CONTENT]
        # 배치(batch)로 처리하여 효율성 증대
        embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=texts_to_embed, task_type="RETRIEVAL_DOCUMENT")
        
        # 각 페이지 데이터에 임베딩 결과 추가
        for i, page_data in enumerate(PDF_CONTENT):
            page_data['embedding'] = embedding_response['embedding'][i]
        print(f"✅ {len(PDF_CONTENT)}개 페이지에 대한 임베딩 생성 완료.")


        # API 클라이언트 초기화
        STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)
        if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=GEMINI_API_KEY)
        
        system_instruction = f"""
            당신은 전문 Q&A 어시스턴트입니다. 
            사용자의 질문에 대해, 제공되는 컨텍스트(텍스트와 이미지)만을 사용하여 간결하고 정확하게 답변해야 합니다. 
            만약 답변이 제공된 특정 이미지와 관련이 있다면, 반드시 답변의 맨 끝에 다음 형식의 태그를 추가해야 합니다: [IMAGE: 이미지파일이름.png]
            
            ✨✨✨ [중요 규칙] ✨✨✨
                1. 모든 답변은 반드시 2000자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
                2. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.
                
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
            message_type = raw_data.get("type"); user_input = raw_data.get("data")
            user_text = ""

            if message_type == "audio":
                audio_bytes = base64.b64decode(user_input)
                stt_request = speech.RecognizeRequest(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hertz=SAMPLE_RATE, language_code="ko-KR"), audio=speech.RecognitionAudio(content=audio_bytes))
                stt_response = await asyncio.to_thread(STT_CLIENT.recognize, request=stt_request)
                user_text = stt_response.results[0].alternatives[0].transcript if stt_response.results else ""
                if user_text: await websocket.send_json({"type": "user_text", "data": user_text})
            
            elif message_type == "text":
                user_text = user_input

            if user_text:
                if "이제 그만" in user_text.strip():
                    ai_text = "챗봇을 종료합니다. 이용해주셔서 감사합니다."
                    await websocket.send_json({"type": "ai_text", "data": ai_text})
                    # ... (TTS 및 종료 로직)
                    break
                
                keywords = user_text.split()
                relevant_pages = [p for p in PDF_CONTENT if any(kw.lower() in p["text"].lower() for kw in keywords if len(kw) > 1)] or PDF_CONTENT
                
                prompt_parts = [ "아래 텍스트와 이미지를 참고하여 질문에 답해줘. 이미지에 대해 언급할 때는, 내가 알려준 '이미지 파일명'을 사용하여 `[IMAGE: 파일명]` 태그를 답변 맨 끝에 붙여야 해. 답변 본문에는 파일명을 절대 언급하지 마." ]
                for page in relevant_pages[:3]:
                    prompt_parts.append(f"\n--- 참고 텍스트 (페이지 {page['page']+1}) ---\n{page['text']}")
                    if page["images"]:
                        for img_file in page["images"]:
                            img_path = IMAGES_DIR / img_file
                            if img_path.exists():
                                try:
                                    prompt_parts.append(f"참고 이미지 파일명: {img_file}")
                                    prompt_parts.append(Image.open(img_path))
                                except Exception as img_e: print(f"경고: 이미지 열기 실패 {img_path}: {img_e}")
                
                prompt_parts.append(f"\n--- 질문 ---\n{user_text}")

                gemini_response = await MODEL.generate_content_async(prompt_parts)
                ai_text_raw = gemini_response.text
                
                image_tag_match = re.search(r"\[IMAGE:\s*(page_\d+_img_\d+\.\w+)\]", ai_text_raw)
                ai_text = re.sub(r"\[IMAGE:\s*(.*?)\]", "", ai_text_raw).strip()
                await websocket.send_json({"type": "ai_text", "data": ai_text})

                if image_tag_match:
                    image_filename = image_tag_match.group(1).strip()
                    image_url = f"/static/images/{image_filename}"
                    await websocket.send_json({"type": "ai_image", "url": image_url})

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

        # --- 1단계: AI에게 사용자 이미지에 대한 '키워드' 묘사 요청 ---
        describe_prompt = ["이 이미지에 무엇이 보이는지 상세하게 묘사해줘.", user_image]
        
        print("🤖 Gemini에게 이미지 묘사 요청...")
        describe_response = await MODEL.generate_content_async(describe_prompt)
        query_text = describe_response.text.strip()
        print(f"✅ 이미지 묘사 생성: '{query_text[:30]}...'")

        query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query_text, task_type="RETRIEVAL_QUERY")['embedding']

        # --- 3단계: 이미지 묘사 벡터와 가장 유사한 PDF 페이지 벡터 검색 ---
        pdf_embeddings = np.array([page['embedding'] for page in PDF_CONTENT])
        query_vector = np.array(query_embedding)
        
        # 코사인 유사도 계산
        dot_products = np.dot(pdf_embeddings, query_vector)
        norms = np.linalg.norm(pdf_embeddings, axis=1) * np.linalg.norm(query_vector)
        similarity_scores = dot_products / norms
        
        best_match_index = np.argmax(similarity_scores)
        best_match_score = similarity_scores[best_match_index]
        
        print(f"✅ 가장 유사한 페이지 찾음: 페이지 {PDF_CONTENT[best_match_index]['page']} (유사도: {best_match_score:.2f})")
        
        # 특정 점수 이하이면 일치하는 결과가 없는 것으로 간주
        if best_match_score < 0.55:
             return {"status": "no_match", "description": "죄송합니다, 이 이미지와 관련된 정보를 찾을 수 없습니다."}

        context_text = PDF_CONTENT[best_match_index]['text']
        
        # --- 4단계: 검색된 텍스트의 최종 요약 요청 (이전과 동일) ---
        summarize_prompt = f"""
        당신은 전문 박물관 도슨트입니다. 
        아래 텍스트는 하나의 특정 전시물에 대한 정보입니다. 
        이 텍스트의 핵심 내용만을 바탕으로, 실제 관람객에게 설명하듯이 생생하고 흥미로운 해설을 생성해주세요. 
        "안녕하세요" 와 같은 인사말은 사용하지 말고, "이것은..." 과 같이 바로 설명으로 시작해야 합니다.
        --- 원본 텍스트 ---
        {context_text}
        """
        print(f"🤖 Gemini에게 텍스트 요약 요청...")
        summarize_response = await MODEL.generate_content_async(summarize_prompt)
        final_description = summarize_response.text.strip()
        
        return {"status": "success", "description": final_description}
    
    except Exception as e:
        print(f"💥 이미지 인식/요약 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
