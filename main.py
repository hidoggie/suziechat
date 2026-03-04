# main.py

import asyncio
import os
import re
import json
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
import numpy as np

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
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL_NAME = "gemini-2.5-flash"

# --- 2. 전역 변수 ---
PDF_CONTENT = []
MODEL, STT_CLIENT, TTS_CLIENT = None, None, None
INITIALIZATION_ERROR = None
KNOWLEDGE_CONTEXT = ""

# ──────────────────────────────────────────────────────────────
# 언어 설정 테이블
# key   : 프론트엔드가 보내는 lang 코드 (ISO 639-1)
# value : (BCP-47 language_code, Google TTS voice_name,
#          STT language_code, 도슨트 프롬프트용 언어명)
# ──────────────────────────────────────────────────────────────
LANG_CONFIG = {
    "ko": ("ko-KR", "ko-KR-Wavenet-A", "ko-KR",  "한국어"),
    "en": ("en-US", "en-US-Wavenet-D", "en-US",  "English"),
    "ja": ("ja-JP", "ja-JP-Wavenet-B", "ja-JP",  "日本語"),
    "zh": ("cmn-CN", "cmn-CN-Wavenet-A", "cmn-Hans-CN", "中文"),
    "fr": ("fr-FR", "fr-FR-Wavenet-C", "fr-FR",  "Français"),
    "de": ("de-DE", "de-DE-Wavenet-D", "de-DE",  "Deutsch"),
    "es": ("es-ES", "es-ES-Wavenet-B", "es-ES",  "Español"),
}
DEFAULT_LANG = "ko"


def get_lang_config(lang_code: str) -> tuple:
    """
    lang_code 에 해당하는 (tts_lang, tts_voice, stt_lang, lang_name) 을 반환.
    없으면 기본값(ko) 사용.
    """
    return LANG_CONFIG.get(lang_code, LANG_CONFIG[DEFAULT_LANG])


def get_voice_params_from_code(lang_code: str) -> tuple[str, str]:
    """TTS 에 필요한 (language_code, voice_name) 만 반환."""
    cfg = get_lang_config(lang_code)
    return cfg[0], cfg[1]  # tts_lang, tts_voice


# ──────────────────────────────────────────────────────────────
# 도슨트 프롬프트 빌더 (언어별)
# ──────────────────────────────────────────────────────────────
def build_docent_prompt(lang_code: str, context_text: str, user_text: str) -> str:
    """언어별 도슨트 해설 생성 프롬프트를 반환합니다."""
    _, _, _, lang_name = get_lang_config(lang_code)

    return f"""
You are a professional museum docent.
The visitor has asked a question about an exhibit.
You MUST answer in {lang_name} regardless of the language of the context below.

[Important Rules]
1. Answer ONLY in {lang_name}.
2. Base your answer solely on the provided context. Do not use outside knowledge.
3. If the context does not contain relevant information, say so politely in {lang_name}.
4. Keep the answer within 300 characters, using 3-4 natural spoken sentences.
5. Do NOT use special symbols (*, -, #, •). Use natural conjunctions instead.
6. Separate every 1-2 sentences with a blank line (double newline) for readability.
7. Do NOT start with "This is..." or "이것은...". Use the subject's name directly.
8. Speak naturally, as if explaining out loud to a visitor.

--- Context ---
{context_text}

--- Question ---
{user_text}
"""


def build_ar_summarize_prompt(lang_code: str, context_text: str) -> str:
    """AR 이미지 인식 후 해설 생성 프롬프트 (언어별)."""
    _, _, _, lang_name = get_lang_config(lang_code)

    return f"""
You are a professional museum docent.
Below is information about a specific exhibit. Summarize the key points in {lang_name} 
as if you are explaining it to a visitor in person.

[Important Rules]
1. Answer ONLY in {lang_name}.
2. If the original text is just a short caption (e.g. "Figure 10 - Outdoor Gallery"), 
   create one natural introductory sentence rather than saying "no information available."
3. Do NOT invent details not present in the source text.
4. Do NOT use special symbols (*, -, #, •).
5. Separate every 1-2 sentences with a blank line for readability.
6. Do NOT open with a greeting. Start directly with the explanation.

--- Source Text ---
{context_text}
"""


# --- 3. Lifespan 초기화 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, STT_CLIENT, TTS_CLIENT, PDF_CONTENT, INITIALIZATION_ERROR, KNOWLEDGE_CONTEXT
    print("✨ 앱 리소스 초기화를 시작합니다...")

    try:
        if IMAGES_DIR.exists():
            shutil.rmtree(IMAGES_DIR)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(__file__).resolve().parent / KNOWLEDGE_PDF_PATH
        if not pdf_path.exists():
            print(f"경고: {pdf_path} 파일을 찾을 수 없습니다.")

        doc = fitz.open(pdf_path)
        print(f"📄 PDF 열기 성공. 총 페이지 수: {len(doc)}")
        content_list = []

        for page_num, page in enumerate(doc):
            print(f"Processing Page {page_num + 1}/{len(doc)}...")
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            image_files = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image_filename = f"page_{page_num}_img_{img_index}.png"
                    with open(IMAGES_DIR / image_filename, "wb") as f:
                        f.write(image_bytes)
                    image_files.append(image_filename)
                except Exception as img_e:
                    print(f"경고: 이미지 처리 실패 (페이지 {page_num}): {img_e}")

            content_list.append({"page": page_num + 1, "text": page_text, "images": image_files})
            await asyncio.sleep(0.01)

        PDF_CONTENT = content_list
        KNOWLEDGE_CONTEXT = "\n\n".join([p['text'] for p in content_list])
        print(f"✅ PDF 처리 완료: {len(doc)} 페이지")

        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=GEMINI_API_KEY)

        # 임베딩 생성
        texts_to_embed = [p['text'] for p in PDF_CONTENT if p['text'].strip()]
        if texts_to_embed:
            embedding_response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts_to_embed,
                task_type="retrieval_document",
                output_dimensionality=768
            )
            embeddings_list = embedding_response.get('embeddings') or embedding_response.get('embedding')
            if not embeddings_list:
                raise KeyError(f"임베딩 데이터를 찾을 수 없습니다. 응답 구조: {list(embedding_response.keys())}")

            text_index = 0
            for page_data in PDF_CONTENT:
                if page_data['text'].strip():
                    page_data['embedding'] = embeddings_list[text_index]
                    text_index += 1
            print(f"✅ {len(texts_to_embed)}개 임베딩 생성 완료.")

        STT_CLIENT = speech.SpeechClient.from_service_account_file(STT_CREDENTIALS_PATH)
        TTS_CLIENT = texttospeech.TextToSpeechClient.from_service_account_file(TTS_CREDENTIALS_PATH)

        # 챗봇 모델 (한국어 기본 — WebSocket 채팅용)
        system_instruction = f"""
            당신은 전문 도슨트입니다.
            사용자의 질문에 대해, 반드시 아래 제공된 '지식 베이스' 내용만을 근거로 해야 합니다.
            당신의 일반 지식을 사용해서는 안 됩니다. '지식 베이스'에 내용이 없다면 "제가 가진 정보로는 답변하기 어렵습니다."라고 솔직하게 말해야 합니다.

            ✨✨✨ [중요 규칙] ✨✨✨
                1. 사용자가 질문한 언어를 감지하여, 반드시 '그 언어'로 답변하세요.
                   (예: 영어 질문 -> 영어 답변, 일본어 질문 -> 일본어 답변)
                2. 모든 답변은 반드시 300자 이내로, 핵심 내용만 간결하게 요약해서 생성해야 합니다.
                3. 답변이 길어질 경우, 가장 중요한 정보부터 순서대로, 최대 3~4개의 문장으로 정리해주세요.

               🗣️ [가독성 및 TTS 최적화 규칙]
                4. 1~2문장이 끝날 때마다 반드시 실제 줄바꿈(엔터)을 두 번 적용하여 문단을 나누세요.
                5. 별표(*), 대시(-), 글머리 기호(•), 샵(#) 등의 특수기호는 절대 사용하지 마세요.
                6. 여러 항목 나열 시 "첫째,", "둘째,", "또한," 같은 자연스러운 접속사를 사용하세요.
                7. 안내원이 직접 말로 설명해 주듯 부드럽고 자연스러운 구어체 문장으로 작성하세요.
                8. "이것은 ~입니다" 식의 지시어 대신 대상의 이름을 주어로 직접 사용하세요.

            [올바른 출력 예시 시작]
            명부시왕은 지장보살을 모시고 저승을 다스리는 열 명의 왕입니다. 화려하게 채색된 옷을 입고 의자에 앉아 있는 모습으로 표현되었습니다.

            우리의 사후세계관에 따르면, 사람이 죽으면 49일째 되는 날까지 일곱 번의 심판을 받습니다.

            그 후에도 죄가 남아 있다면 세 번에 걸쳐 추가 심판을 받게 됩니다.
            [올바른 출력 예시 끝]

         --- 지식 베이스 ---
        {KNOWLEDGE_CONTEXT}
        """

        generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKENS)
        MODEL = genai.GenerativeModel(
            CHAT_MODEL_NAME,
            system_instruction=system_instruction,
            generation_config=generation_config
        )

        print("🎉 모든 리소스 초기화 완료. 챗봇이 준비되었습니다.")

    except Exception as e:
        INITIALIZATION_ERROR = f"[{type(e).__name__}] {e}"
        print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR}")

    yield

    print("👋 서버를 종료합니다.")


app = FastAPI(lifespan=lifespan)

# --- 4. 헬퍼 함수 ---
def find_best_page_by_vector(query_text: str):
    if not query_text or not any('embedding' in p for p in PDF_CONTENT):
        return None

    res = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query_text,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    query_vector = np.array(res['embedding'])
    pdf_vectors = np.array([p['embedding'] for p in PDF_CONTENT if 'embedding' in p])

    dot_products = np.dot(pdf_vectors, query_vector)
    norms = np.linalg.norm(pdf_vectors, axis=1) * np.linalg.norm(query_vector)
    similarity_scores = dot_products / (norms + 1e-9)

    best_idx = np.argmax(similarity_scores)
    max_score = similarity_scores[best_idx]
    print(f"🔍 벡터 검색 완료. 최고 점수: {max_score:.4f}")
    return PDF_CONTENT[best_idx] if max_score > 0.4 else None


# --- 5. FastAPI 엔드포인트 ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/ar", response_class=FileResponse)
async def read_ar_page():
    return FileResponse(BASE_DIR / "static" / "ar.html")


@app.get("/api/pdf-content")
async def get_pdf_content():
    if INITIALIZATION_ERROR or not PDF_CONTENT:
        return JSONResponse(status_code=500, content={"error": "PDF content not loaded."})
    return JSONResponse(content=PDF_CONTENT)


# ──────────────────────────────────────────────────────────────
# TTS API  ← lang 파라미터 추가
# ──────────────────────────────────────────────────────────────
@app.post("/api/tts")
async def text_to_speech_api(payload: dict = Body(...)):
    """
    payload:
      - text_to_speak : 읽을 텍스트 (필수)
      - lang          : ISO 639-1 언어 코드 (선택, 없으면 텍스트 자동감지)
    """
    text_to_speak = payload.get("text_to_speak")
    lang_code     = payload.get("lang", "").strip().lower()  # 프론트에서 전달

    if not text_to_speak:
        raise HTTPException(status_code=400, detail="text_to_speak 필드가 필요합니다.")
    if not TTS_CLIENT:
        raise HTTPException(status_code=500, detail="TTS client not initialized")

    # lang 파라미터가 있으면 우선 사용, 없으면 텍스트 자동감지
    if lang_code and lang_code in LANG_CONFIG:
        tts_lang, tts_voice = get_voice_params_from_code(lang_code)
        print(f"🌐 TTS 언어 (프론트 지정): {lang_code} → {tts_lang} / {tts_voice}")
    else:
        # 폴백: langdetect 자동감지
        try:
            from langdetect import detect
            detected = detect(text_to_speak)
        except Exception:
            detected = DEFAULT_LANG
        tts_lang, tts_voice = get_voice_params_from_code(detected)
        print(f"🌐 TTS 언어 (자동감지): {detected} → {tts_lang} / {tts_voice}")

    try:
        tts_request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=text_to_speak),
            voice=texttospeech.VoiceSelectionParams(
                language_code=tts_lang,
                name=tts_voice
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            ),
        )
        tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')
        return {"audio": audio_base64}

    except Exception as e:
        print(f"💥 TTS API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# 이미지 인식 API  ← lang 파라미터 추가
# ──────────────────────────────────────────────────────────────
@app.post("/api/recognize-image")
async def recognize_image(payload: dict = Body(...)):
    """
    payload:
      - image : base64 이미지 데이터 (필수)
      - lang  : ISO 639-1 언어 코드 (선택, 기본 'ko')
    """
    user_image_b64 = payload.get("image")
    lang_code      = payload.get("lang", DEFAULT_LANG).strip().lower()

    if not user_image_b64:
        raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")
    if lang_code not in LANG_CONFIG:
        lang_code = DEFAULT_LANG

    _, _, _, lang_name = get_lang_config(lang_code)
    print(f"🌐 이미지 인식 요청 언어: {lang_code} ({lang_name})")

    try:
        user_image_bytes = base64.b64decode(user_image_b64.split(',')[1])
        user_image = Image.open(io.BytesIO(user_image_bytes))

        # 1단계: 키워드 추출 (언어 무관 — 항상 영어/원문)
        extract_keywords_prompt = [
            "You are an expert in OCR and object identification. "
            "Analyze the image and identify any legible text (signs, titles) or unique objects. "
            "Respond ONLY with a comma-separated list of keywords. No sentences. "
            "Example: 대웅전, 청룡, 다포 양식",
            user_image
        ]
        print("🤖 키워드 추출 요청...")
        kw_response = await MODEL.generate_content_async(extract_keywords_prompt)
        keywords = [kw.strip() for kw in kw_response.text.strip().split(',') if kw.strip()]

        if not keywords:
            no_match_msg = {
                "ko": "죄송합니다, 사진에서 특징을 인식할 수 없습니다. 더 선명하게 촬영해보세요.",
                "en": "Sorry, I couldn't identify features in the photo. Please try a clearer shot.",
                "ja": "申し訳ありませんが、写真から特徴を認識できませんでした。もっと鮮明に撮影してください。",
                "zh": "抱歉，无法识别照片中的特征，请尝试拍摄更清晰的照片。",
                "fr": "Désolé, je n'ai pas pu identifier de caractéristiques. Essayez une photo plus nette.",
                "de": "Entschuldigung, es konnten keine Merkmale erkannt werden. Bitte fotografieren Sie klarer.",
                "es": "Lo siento, no pude identificar características. Por favor tome una foto más nítida.",
            }
            return {"status": "no_match", "description": no_match_msg.get(lang_code, no_match_msg["ko"])}

        print(f"✅ 추출된 키워드: {keywords}")

        # 2단계: PDF에서 가장 일치하는 페이지 검색
        best_match = {"score": 0, "page_data": None}
        for page in PDF_CONTENT:
            score = sum(1 for kw in keywords if kw.lower() in page["text"].lower())
            if score > best_match["score"]:
                best_match["score"] = score
                best_match["page_data"] = page

        if best_match["score"] == 0 or best_match["page_data"] is None:
            no_info_msg = {
                "ko": "죄송합니다, 이 이미지와 일치하는 정보를 찾을 수 없습니다.",
                "en": "Sorry, no matching information was found for this image.",
                "ja": "申し訳ありませんが、この画像に一致する情報が見つかりませんでした。",
                "zh": "抱歉，找不到与此图片匹配的信息。",
                "fr": "Désolé, aucune information correspondante n'a été trouvée pour cette image.",
                "de": "Entschuldigung, es wurden keine passenden Informationen für dieses Bild gefunden.",
                "es": "Lo siento, no se encontró información coincidente para esta imagen.",
            }
            return {"status": "no_match", "description": no_info_msg.get(lang_code, no_info_msg["ko"])}

        context_text = best_match["page_data"]["text"]
        matched_page = best_match["page_data"]["page"]
        print(f"✅ 매칭 페이지: {matched_page} (키워드 점수: {best_match['score']})")

        # 3단계: 선택된 언어로 해설 생성
        summarize_prompt = build_ar_summarize_prompt(lang_code, context_text)
        print(f"🤖 페이지 {matched_page} 해설 생성 요청 ({lang_name})...")
        summarize_response = await MODEL.generate_content_async(summarize_prompt)
        final_description = summarize_response.text.strip()

        print("✅ 해설 생성 완료.")
        return {"status": "success", "description": final_description}

    except Exception as e:
        print(f"💥 이미지 인식/요약 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# AR 쿼리 API  ← lang 파라미터 추가
# ──────────────────────────────────────────────────────────────
@app.post("/api/ar-query")
async def ar_query(payload: dict = Body(...)):
    """
    payload:
      - image_name : PDF에서 추출된 이미지 파일명 (필수)
      - lang       : ISO 639-1 언어 코드 (선택, 기본 'ko')
    """
    image_name = payload.get("image_name")
    lang_code  = payload.get("lang", DEFAULT_LANG).strip().lower()

    if not image_name:
        raise HTTPException(status_code=400, detail="image_name이 필요합니다.")
    if MODEL is None:
        return {"error": "Model not initialized"}
    if lang_code not in LANG_CONFIG:
        lang_code = DEFAULT_LANG

    _, _, _, lang_name = get_lang_config(lang_code)

    context_text = ""
    for page in PDF_CONTENT:
        if image_name in page["images"]:
            context_text = page["text"]
            break

    if not context_text:
        return {"error": "Context not found for this image"}

    try:
        summarize_prompt = build_ar_summarize_prompt(lang_code, context_text)
        gemini_response = await MODEL.generate_content_async(summarize_prompt)
        ai_text = gemini_response.text.strip()

        tts_lang, tts_voice = get_voice_params_from_code(lang_code)
        tts_request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=ai_text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=tts_lang,
                name=tts_voice
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            ),
        )
        tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')

        return {"text": ai_text, "audio": audio_base64}

    except Exception as e:
        print(f"💥 AR 쿼리 처리 오류: {e}")
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
# WebSocket  (챗봇 — 언어는 질문 텍스트로 자동감지, 기존 방식 유지)
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket 연결 성공")

    if INITIALIZATION_ERROR:
        await websocket.send_json({"type": "error", "data": f"서버 초기화 실패: {INITIALIZATION_ERROR}"})
        await websocket.close()
        return

    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"✅ 클라이언트 연결됨: {client_id}")

    try:
        while True:
            print(f"⏳ ({client_id}) 메시지 수신 대기 중...")
            raw_data = await websocket.receive_json()
            message_type = raw_data.get("type")
            user_input   = raw_data.get("data")
            user_text    = ""

            if message_type == "audio":
                print(f"🎤 ({client_id}) 오디오 처리 시작")
                audio_bytes = base64.b64decode(user_input)
                stt_request = speech.RecognizeRequest(
                    config=speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                        sample_rate_hertz=SAMPLE_RATE,
                        language_code="ko-KR",
                        alternative_language_codes=["en-US", "ja-JP", "cmn-Hans-CN"]
                    ),
                    audio=speech.RecognitionAudio(content=audio_bytes)
                )
                stt_response = await asyncio.to_thread(STT_CLIENT.recognize, request=stt_request)
                user_text = (
                    stt_response.results[0].alternatives[0].transcript
                    if stt_response.results else ""
                )
                if user_text:
                    print(f"🗣️ ({client_id}) STT 결과: {user_text}")
                    await websocket.send_json({"type": "user_text", "data": user_text})

            elif message_type == "text":
                user_text = user_input
                print(f"⌨️ ({client_id}) 텍스트 입력: {user_text}")

            if not user_text:
                continue

            print(f"👤 사용자 ({client_id}): {user_text}")

            if "이제 그만" in user_text.strip():
                await websocket.send_json({"type": "ai_text", "data": "챗봇을 종료합니다. 이용해주셔서 감사합니다."})
                break

            # 이미지 표시 의도 판단
            image_keywords = ["보여줘", "사진", "그림", "이미지", "생김새", "모습"]
            show_image_intent = any(kw in user_text for kw in image_keywords)

            # 키워드 기반 페이지 검색
            keywords = re.findall(r'[\w가-힣]{2,}', user_text)
            best_match = {"score": 0, "page_data": None}
            for page in PDF_CONTENT:
                score = sum(1 for kw in keywords if kw.lower() in page["text"].lower())
                if score > best_match["score"]:
                    best_match["score"] = score
                    best_match["page_data"] = page

            context_text = KNOWLEDGE_CONTEXT  # 기본: 전체 컨텍스트
            if best_match["page_data"]:
                context_text = best_match["page_data"]["text"]
                print(f"✅ ({client_id}) 컨텍스트: 페이지 {best_match['page_data']['page']}")
            else:
                print(f"⚠️ ({client_id}) 매칭 실패. 전체 컨텍스트 사용.")

            # Gemini 응답 생성
            # WebSocket 채팅은 MODEL의 system_instruction(언어 자동감지)을 그대로 활용
            prompt = f"""
            당신은 전문 박물관 도슨트입니다.

            [중요 규칙]
            1. "제공된 정보에는...", "그림 3은..." 과 같이 상황 설명이나 이미지 번호를 언급하지 마세요.
            2. 인사말 없이 바로 설명으로 시작하세요.
            3. 원본 텍스트에 없는 내용은 지어내지 마세요.

            --- 컨텍스트 ---
            {context_text}

            --- 질문 ---
            {user_text}
            """
            print(f"🤖 ({client_id}) Gemini 요청 전송...")
            gemini_response = await MODEL.generate_content_async(prompt)
            ai_text = gemini_response.text.strip()
            print(f"🤖 ({client_id}) Gemini 응답: {ai_text[:50]}...")

            await websocket.send_json({"type": "ai_text", "data": ai_text})

            # 이미지 전송
            if show_image_intent and best_match["page_data"] and best_match["page_data"]["images"]:
                image_url = f"/static/images/{best_match['page_data']['images'][0]}"
                print(f"🖼️ 이미지 전송: {image_url}")
                await websocket.send_json({"type": "ai_image", "data": {"url": image_url}})

            # TTS — 응답 텍스트 언어 자동감지 (챗봇은 질문 언어를 따라가므로)
            try:
                from langdetect import detect
                detected_lang = detect(ai_text)
            except Exception:
                detected_lang = DEFAULT_LANG

            tts_lang, tts_voice = get_voice_params_from_code(detected_lang)
            print(f"🔊 ({client_id}) TTS 언어: {detected_lang} → {tts_lang}")

            tts_request = texttospeech.SynthesizeSpeechRequest(
                input=texttospeech.SynthesisInput(text=ai_text),
                voice=texttospeech.VoiceSelectionParams(
                    language_code=tts_lang,
                    name=tts_voice
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
            )
            tts_response = await asyncio.to_thread(TTS_CLIENT.synthesize_speech, request=tts_request)
            if tts_response.audio_content:
                print(f"🔊 ({client_id}) TTS 오디오 전송")
                await websocket.send_bytes(tts_response.audio_content)
            else:
                print(f"❌ ({client_id}) TTS 오디오 생성 실패")

    except WebSocketDisconnect:
        print(f"🔌 클라이언트 연결 끊어짐: {client_id}")
    except Exception as e:
        print(f"💥 처리 중 오류 ({client_id}): {e}")
    finally:
        print(f"🏁 웹소켓 세션 종료: {client_id}")
