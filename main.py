# main.py — AI 행사 안내봇 (챗봇 + 간판 스캔)
#
# 주요 변경점
#   - 4개 국어(ko/en/ja/zh): 언어는 프론트가 선택해서 보내고, 서버는 그대로 따름 (langdetect 제거)
#   - 지식 파일을 '섹션(⚫ 제목)' 단위로 파싱 / PDF 또는 Markdown(.md, .txt) 지원
#   - 지식이 작으면 전체를 컨텍스트로 사용, 크면 임베딩 검색(top-k) 자동 전환
#   - 간판 스캔: 섹션 목록을 Gemini에 주고 '어느 항목인지' 분류 → 해설 생성
#   - WebSocket → REST(/api/chat) 전환: 모바일 안정성 ↑, Node.js 메인에서 호출하기 쉬움
#   - google-generativeai(지원 종료) → google-genai SDK, 모든 외부 호출 비동기화
#   - 서버는 즉시 기동하고 초기화는 백그라운드에서 진행 (/api/config 의 ready 로 확인)
#   - IP당 요청 제한, 입력 길이 제한, TTS 캐시, CORS 설정

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import shutil
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pymupdf as fitz  # PyMuPDF (구 'import fitz'는 폐기 예정)
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.cloud import texttospeech


# ══════════════════════════════════════════════════════════════
# 1. 설정 (모두 환경변수로 덮어쓸 수 있음)
# ══════════════════════════════════════════════════════════════
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PDF_IMAGES_DIR = STATIC_DIR / "images"   # PDF에서 추출한 이미지 (기동 시 재생성)
KB_IMAGES_DIR = STATIC_DIR / "kb"        # 지식 파일의 '이미지:' 줄로 지정하는 이미지 (지도 등)
EMBED_CACHE_PATH = BASE_DIR / ".embedding_cache.json"

# 지식 파일: .pdf / .md / .txt  (기존 KNOWLEDGE_PDF_PATH 이름도 호환)
KNOWLEDGE_PATH = BASE_DIR / os.environ.get(
    "KNOWLEDGE_PATH", os.environ.get("KNOWLEDGE_PDF_PATH", "knowledge.txt")
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# 예비 모델: 기본 모델이 하루 한도(RPD)를 다 쓰거나 분당 한도가 꽉 찼을 때 대신 답함. 빈 값이면 끔.
#   Tier 1 기준 gemini-2.5-flash-lite = 분당 4,000건 · 400만 토큰 · 하루 무제한
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
GEMINI_TIMEOUT = _env_int("GEMINI_TIMEOUT", 40)
# 2.5 계열 thinking 토큰: 0이면 끔(응답 빠름). 빈 값/auto 면 모델 기본값 사용
_tb = os.environ.get("GEMINI_THINKING_BUDGET", "0").strip().lower()
THINKING_BUDGET = None if _tb in ("", "auto", "none") else int(_tb)

TTS_CREDENTIALS_PATH = os.environ.get(
    "TTS_CREDENTIALS_PATH", "/etc/secrets/voice-chat-462608-e445e48514e2.json"
)
TTS_SPEAKING_RATE = _env_float("TTS_SPEAKING_RATE", 1.0)

EVENT_NAME = os.environ.get("EVENT_NAME", "행사")
EVENT_TZ = os.environ.get("EVENT_TZ", "Asia/Seoul")

FULL_CONTEXT_MAX_CHARS = _env_int("FULL_CONTEXT_MAX_CHARS", 40000)  # 이하면 지식 전체를 컨텍스트로
TOP_K = _env_int("RAG_TOP_K", 4)
SIM_THRESHOLD = _env_float("RAG_SIM_THRESHOLD", 0.55)  # 새 지식으로 테스트하며 조정
# 자동 절약 모드: 최근 1분 토큰 사용량이 예산의 이 비율을 넘으면, 그동안만 '관련 부분만 보내기(top-k)'로 자동 전환
#   평소엔 지식 전체를 보내 풍부하게 답하고, 몰릴 때만 질문당 토큰을 줄여 처리량을 늘림. 0 이면 끔.
AUTO_RAG_RATIO = _env_float("AUTO_RAG_RATIO", 0.6)

# 환경변수가 없으면 stechart 를 기본 허용 (다른 도메인에서 /api 호출 시 필요)
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "https://stechart.stampquest.kr").split(",") if o.strip()]
# 같은 통신사 공용 IP(CGNAT)를 여러 관람객이 함께 쓰는 경우가 많아 넉넉하게 (축제 현장 대비)
RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 90)
MAX_QUESTION_CHARS = 300
# Gemini 분당 입력 토큰 한도(유료 Tier 1: gemini-2.5-flash 1,000,000) 보다 조금 낮게 → 넘을 것 같으면 잠깐 기다렸다 보냄
TPM_BUDGET = _env_int("GEMINI_TPM_BUDGET", 900000)
TPM_MAX_WAIT = _env_int("GEMINI_TPM_MAX_WAIT", 8)         # 최대 대기(초) — 그래도 자리가 없으면 Gemini 를 부르지 않고 "잠시 후 다시" 안내(503 busy)
ANSWER_CACHE_SEC = _env_int("ANSWER_CACHE_SEC", 180)      # 같은 질문(같은 언어) 답변 재사용 시간(초), 0 이면 끔
MAX_TTS_CHARS = 1200          # Google TTS 5000바이트 제한 대비 (한·중·일 1자 ≈ 3바이트)
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# ──────────────────────────────────────────────────────────────
# 언어 설정 (프론트 lang 코드 → 프롬프트 언어명 / TTS)
# 목소리는 TTS_VOICE_KO 같은 환경변수로 바꿀 수 있음
# ──────────────────────────────────────────────────────────────
LANG_CONFIG = {
    "ko": {"name": "Korean (한국어)", "tts_lang": "ko-KR", "voice": "ko-KR-Wavenet-A"},
    "en": {"name": "English", "tts_lang": "en-US", "voice": "en-US-Wavenet-D"},
    "ja": {"name": "Japanese (日本語)", "tts_lang": "ja-JP", "voice": "ja-JP-Wavenet-B"},
    "zh": {"name": "Simplified Chinese (简体中文)", "tts_lang": "cmn-CN", "voice": "cmn-CN-Wavenet-A"},
    "vi": {"name": "Vietnamese (Tiếng Việt)", "tts_lang": "vi-VN", "voice": "vi-VN-Wavenet-A"},
}
for _code, _cfg in LANG_CONFIG.items():
    _cfg["voice"] = os.environ.get(f"TTS_VOICE_{_code.upper()}", _cfg["voice"])
DEFAULT_LANG = "ko"


def norm_lang(code: str | None) -> str:
    c = (code or "").strip().lower()
    if c.startswith("zh"):
        return "zh"
    c = c.split("-")[0]
    return c if c in LANG_CONFIG else DEFAULT_LANG


MSG = {
    "sign_no_match": {
        "ko": "간판에 해당하는 안내 정보를 찾지 못했어요. 간판 글자가 잘 보이도록 가까이에서 다시 비춰 주세요.",
        "en": "I couldn't find information for this sign. Please move closer so the text on the sign is clearly visible and try again.",
        "ja": "この看板に該当する案内情報が見つかりませんでした。看板の文字がはっきり見えるよう、近づいてもう一度かざしてください。",
        "zh": "没有找到与该指示牌相关的信息。请靠近一些，让指示牌上的文字清晰可见后再试一次。",
        "vi": "Tôi không tìm thấy thông tin cho bảng hiệu này. Vui lòng đến gần hơn để thấy rõ chữ trên bảng hiệu và thử lại.",
    },
    "sign_unreadable": {
        "ko": "사진에서 간판이나 글자를 알아보기 어려워요. 조금 더 가까이, 흔들리지 않게 비춰 주세요.",
        "en": "I couldn't read a sign in the photo. Please hold the camera steady a little closer and try again.",
        "ja": "写真から看板や文字を読み取れませんでした。もう少し近づいて、ぶれないようにかざしてください。",
        "zh": "无法从照片中识别指示牌或文字。请靠近一些并保持相机稳定后再试。",
        "vi": "Tôi không thể đọc được bảng hiệu hoặc chữ trong ảnh. Vui lòng đến gần hơn, giữ chắc máy ảnh và thử lại.",
    },
}

IMAGE_INTENT_WORDS = [
    # ko
    "보여", "사진", "그림", "이미지", "지도", "약도", "배치도", "위치도",
    # en
    "show", "picture", "photo", "image", "map",
    # ja
    "見せ", "写真", "画像", "地図", "マップ",
    # zh
    "看看", "照片", "图片", "图像", "地图",
    # vi
    "xem", "ảnh", "bức ảnh", "hình ảnh", "bản đồ",
]


# ══════════════════════════════════════════════════════════════
# 2. 상태
# ══════════════════════════════════════════════════════════════
class State:
    ready = False
    error: str | None = None
    client: genai.Client | None = None
    tts: texttospeech.TextToSpeechClient | None = None
    sections: list[dict] = []
    full_context = ""
    catalog = ""
    use_full_context = True
    matrix: np.ndarray | None = None  # 정규화된 임베딩 (n, d)


S = State()


# ══════════════════════════════════════════════════════════════
# 3. 지식 파일 파싱 → 섹션
#    섹션 시작: ⚫ ● ■ ◆ ▶ ◉ 로 시작하는 줄, 또는 Markdown '#', '##', '###'
#    섹션 안 메타 줄:
#      키워드: 정문, Main Gate, 正門, 正门      ← 간판 매칭·검색 정확도 향상
#      이미지: venue_map.png                      ← static/kb/ 안의 파일
# ══════════════════════════════════════════════════════════════
# 마커 뒤에 제목이 없으면(PDF에서 '⚫'만 한 줄로 추출되는 경우) 다음 줄을 제목으로 사용
SECTION_MARK = re.compile(r"^\s*(?:[⚫●■◆▶◉]|#{1,3}(?=\s|$))\s*(.*?)\s*$")
KEYWORD_LINE = re.compile(r"^\s*(?:키워드|keywords?|キーワード|关键词)\s*[:：]\s*(.+)$", re.I)
IMAGE_LINE = re.compile(r"^\s*(?:이미지|images?|画像|图片)\s*[:：]\s*(.+)$", re.I)
BROWSER_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}


def _split_list(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,、，]", s) if x.strip()]


def _read_pdf(path: Path) -> list[dict]:
    doc = fitz.open(path)
    # 여러 페이지에 반복되는 이미지(로고·배경)는 제외하기 위해 사용 횟수 집계
    usage: dict[int, int] = defaultdict(int)
    for page in doc:
        for xref in {img[0] for img in page.get_images(full=True)}:
            usage[xref] += 1

    pages = []
    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text")
        imgs = []
        for idx, img in enumerate(page.get_images(full=True)):
            xref, w, h = img[0], img[2], img[3]
            if usage[xref] >= 3 or min(w, h) < 150:
                continue
            try:
                base = doc.extract_image(xref)
                ext = (base.get("ext") or "png").lower()
                if ext in BROWSER_IMAGE_EXT:
                    name = f"p{pno}_{idx}.{ext}"
                    (PDF_IMAGES_DIR / name).write_bytes(base["image"])
                else:  # jpx, jb2 등 브라우저가 못 여는 형식은 PNG로 변환
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    name = f"p{pno}_{idx}.png"
                    pix.save(PDF_IMAGES_DIR / name)
                imgs.append(f"/static/images/{name}")
            except Exception as e:
                print(f"⚠️ 이미지 추출 실패 (p{pno}, #{idx}): {e}")
        pages.append({"page": pno, "text": text, "images": imgs})
    doc.close()
    return pages


def _read_text(path: Path) -> list[dict]:
    return [{"page": 1, "text": path.read_text(encoding="utf-8"), "images": []}]


def _split_sections(pages: list[dict]) -> list[dict]:
    sections: list[dict] = []
    cur: dict | None = None
    any_marker = False

    def start(title: str, page: int) -> dict:
        sec = {"title": title.strip(), "lines": [], "keywords": [], "images": [], "page": page}
        sections.append(sec)
        return sec

    pending_title = False
    for p in pages:
        started_here: list[dict] = []
        for line in p["text"].splitlines():
            m = SECTION_MARK.match(line)
            if m:
                any_marker = True
                cur = start(m.group(1), p["page"])
                started_here.append(cur)
                pending_title = not cur["title"]
                continue
            if pending_title:
                if line.strip():
                    cur["title"] = line.strip()
                    pending_title = False
                continue
            if cur is None:
                if not line.strip():
                    continue
                cur = start(line, p["page"])  # 첫 마커 이전의 머리말: 첫 줄을 제목으로
                started_here.append(cur)
                continue
            km = KEYWORD_LINE.match(line)
            if km:
                cur["keywords"] += _split_list(km.group(1))
                cur["lines"].append(line)
                continue
            im = IMAGE_LINE.match(line)
            if im:
                for name in _split_list(im.group(1)):
                    if (KB_IMAGES_DIR / name).is_file():
                        cur["images"].append(f"/static/kb/{quote(name)}")
                    else:
                        print(f"⚠️ 이미지 파일 없음: static/kb/{name}")
                continue
            cur["lines"].append(line)

        # PDF 추출 이미지 배정: 그 페이지에서 시작한 섹션이 하나뿐일 때만 (애매하면 '이미지:' 줄로 지정)
        if p["images"]:
            if len(started_here) == 1:
                started_here[0]["images"] += p["images"]
            elif not started_here and cur is not None:
                cur["images"] += p["images"]

    # 마커가 전혀 없는 여러 페이지 PDF → 페이지 단위로 폴백
    if not any_marker and len(pages) > 1:
        sections = []
        for p in pages:
            lines = [ln for ln in p["text"].splitlines()]
            first = next((ln for ln in lines if ln.strip()), "")
            if not first:
                continue
            sections.append({"title": first.strip(), "lines": lines, "keywords": [],
                             "images": p["images"], "page": p["page"]})

    result = []
    for sec in sections:
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(sec["lines"])).strip()
        if not text and not sec["keywords"]:
            continue
        result.append({
            "id": len(result),
            "title": sec["title"],
            "text": text,
            "keywords": sec["keywords"],
            "images": list(dict.fromkeys(sec["images"])),
            "page": sec["page"],
        })
    return result


def section_block(s: dict) -> str:
    return f"[#{s['id']}] {s['title']}\n{s['text']}"


def build_catalog(sections: list[dict]) -> str:
    rows = []
    for s in sections:
        row = f"[#{s['id']}] {s['title']}"
        if s["keywords"]:
            row += f" | keywords: {', '.join(s['keywords'])}"
        rows.append(row)
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════
# 4. Gemini / 임베딩 헬퍼
# ══════════════════════════════════════════════════════════════
def _gen_config(system: str | None = None, max_tokens: int = 1024,
                json_mode: bool = False, temperature: float = 0.4) -> types.GenerateContentConfig:
    kw = dict(system_instruction=system, max_output_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if THINKING_BUDGET is not None:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    return types.GenerateContentConfig(**kw)


# ── 분당 입력 토큰 예산 ──
#   최근 60초 동안 보낸 입력 토큰을 기록해 두고, 이번 요청까지 보내면 한도를 넘을 것 같으면
#   실패(429)시키는 대신 자리가 날 때까지 잠깐 기다림 (사람이 몰리는 순간을 줄 세워서 흘려보냄)
_tok_log: deque = deque()          # (보낸 시각, 토큰 수)
_tok_per_char = 1.0                # 글자당 토큰 추정치 — 실제 응답의 사용량으로 계속 보정 (처음엔 넉넉하게)
_tok_lock = asyncio.Lock()


def _tok_used(now: float) -> int:
    while _tok_log and now - _tok_log[0][0] > 60:
        _tok_log.popleft()
    return sum(t for _, t in _tok_log)


class GeminiBusy(Exception):
    """분당 토큰 한도가 꽉 차서 잠시 후 다시 시도해야 하는 상태"""


async def _reserve_tokens(est: int, max_wait: float | None = None) -> list:
    deadline = time.monotonic() + (TPM_MAX_WAIT if max_wait is None else max_wait)
    while True:
        async with _tok_lock:
            now = time.monotonic()
            if _tok_used(now) + est <= TPM_BUDGET:
                entry = [now, est]
                _tok_log.append(entry)
                return entry
            if now >= deadline:
                # 오래 기다리게 하거나 어차피 거절될 요청을 보내 한도를 더 깎지 않고 바로 '혼잡' 안내
                raise GeminiBusy()
        await asyncio.sleep(0.4)


def _contents_chars(contents: list, system: str | None) -> int:
    n = len(system or "")
    for c in contents:
        n += len(c) if isinstance(c, str) else 1500   # 이미지 1장 ≈ 토큰 수백~천여 개
    return n


_primary_off_until = 0.0   # 기본 모델 하루 한도 소진 시, 한도가 초기화되는 시각(epoch)까지 예비 모델 사용
_fb_count = 0


def _next_quota_reset() -> float:
    """Gemini 하루 한도는 미국 태평양 시간 자정에 초기화 (한국 시간 오후 4시, 서머타임 해제 후 오후 5시)"""
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return nxt.timestamp()


def _is_daily_quota(msg: str) -> bool:
    return bool(re.search(r"per_?day|perday|requests per day|daily", msg, re.I))


async def _gemini_fallback(contents: list, kw: dict, why: str) -> str:
    global _fb_count
    _fb_count += 1
    print(f"🔀 예비 모델({FALLBACK_MODEL})로 답변 — {why}")
    for attempt in range(2):
        try:
            resp = await asyncio.wait_for(
                S.client.aio.models.generate_content(model=FALLBACK_MODEL, contents=contents, config=_gen_config(**kw)),
                timeout=GEMINI_TIMEOUT,
            )
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            if attempt == 0 and any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded")):
                await asyncio.sleep(2)
                continue
            raise


async def gemini(contents: list, **kw) -> str:
    global _tok_per_char, _primary_off_until
    if FALLBACK_MODEL and time.time() < _primary_off_until:
        return await _gemini_fallback(contents, kw, "기본 모델 하루 한도 소진")
    chars = _contents_chars(contents, kw.get("system"))
    # 사람이 몰려 Gemini 가 일시적으로 거절(429 요청 한도 / 503 과부하)하면 예비 모델로, 예비 모델이 없으면 잠깐 쉬었다가 한 번 더
    for attempt in range(2):
        try:
            # 예비 모델이 있으면 오래 기다리지 않고(최대 2초) 바로 예비 모델로
            entry = await _reserve_tokens(int(chars * _tok_per_char), min(TPM_MAX_WAIT, 2) if FALLBACK_MODEL else None)
        except GeminiBusy:
            if FALLBACK_MODEL:
                return await _gemini_fallback(contents, kw, "분당 토큰 한도 꽉 참")
            raise
        try:
            resp = await asyncio.wait_for(
                S.client.aio.models.generate_content(model=CHAT_MODEL, contents=contents, config=_gen_config(**kw)),
                timeout=GEMINI_TIMEOUT,
            )
            used = getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", None)
            if used:
                entry[1] = int(used)                                  # 추정치를 실제 사용량으로 교체
                _tok_per_char = 0.8 * _tok_per_char + 0.2 * (used / max(chars, 1))
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            busy = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
            if busy and FALLBACK_MODEL:
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and _is_daily_quota(msg):
                    _primary_off_until = _next_quota_reset()
                    print(f"🛑 기본 모델 하루 한도 소진 → {datetime.fromtimestamp(_primary_off_until, ZoneInfo(EVENT_TZ)):%m/%d %H:%M} 까지 예비 모델 사용")
                    return await _gemini_fallback(contents, kw, "기본 모델 하루 한도 소진")
                return await _gemini_fallback(contents, kw, msg[:60])
            if attempt == 0 and busy:
                print(f"⏳ Gemini 혼잡 → 재시도: {msg[:80]}")
                await asyncio.sleep(3 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 1.5)
                continue
            raise


async def embed_texts(texts: list[str], task: str) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 20):
        r = await S.client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[i:i + 20],
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
        )
        out.extend(e.values for e in r.embeddings)
    return np.array(out, dtype=np.float32)


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def clean_answer(text: str) -> str:
    """TTS·말풍선용 정리: 마크다운 기호 제거, 문단 간격 정리."""
    t = text.replace("\r", "")
    t = re.sub(r"\*\*|__|`", "", t)
    t = re.sub(r"^[ \t]*[-*•#]+[ \t]+", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def now_str() -> str:
    try:
        return datetime.now(ZoneInfo(EVENT_TZ)).strftime("%Y-%m-%d %A %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %A %H:%M")


async def _build_embeddings():
    cache: dict = {}
    if EMBED_CACHE_PATH.exists():
        try:
            cache = json.loads(EMBED_CACHE_PATH.read_text())
        except Exception:
            cache = {}
    texts = [f"{s['title']}\n{' '.join(s['keywords'])}\n{s['text']}"[:6000] for s in S.sections]
    keys = [hashlib.sha256(f"{EMBEDDING_MODEL}|{EMBED_DIM}|{t}".encode()).hexdigest()[:32] for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        print(f"🧮 임베딩 생성: {len(missing)}개 (캐시 {len(keys) - len(missing)}개)")
        vecs = await embed_texts([texts[i] for i in missing], "RETRIEVAL_DOCUMENT")
        for i, v in zip(missing, vecs):
            cache[keys[i]] = [round(float(x), 6) for x in v]
        try:
            EMBED_CACHE_PATH.write_text(json.dumps({k: cache[k] for k in keys}))
        except Exception as e:
            print(f"⚠️ 임베딩 캐시 저장 실패: {e}")
    m = np.array([cache[k] for k in keys], dtype=np.float32)
    S.matrix = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


# ══════════════════════════════════════════════════════════════
# 5. 검색
# ══════════════════════════════════════════════════════════════
def keyword_search(text: str, limit: int = TOP_K) -> list[dict]:
    low = text.lower()
    # 영문·숫자는 3자 이상(of, to 같은 잡음 제외), 한·중·일 문자는 2자 이상
    toks = {t for t in re.findall(r"\w{2,}", low) if len(t) >= 3 or not t.isascii()}
    scored = []
    for s in S.sections:
        head = (s["title"] + " " + " ".join(s["keywords"])).lower()
        body = s["text"].lower()
        sc = sum(3 for t in toks if t in head) + sum(1 for t in toks if t in body)
        # 반대 방향: 키워드/제목 조각이 입력(OCR 등)에 그대로 들어있는 경우 (띄어쓰기 없는 중·일 텍스트 대응)
        names = s["keywords"] + [x.strip() for x in s["title"].split("/")]
        sc += sum(5 for n in names if len(n) >= 2 and n.lower() in low)
        if sc:
            scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:limit]]


async def retrieve(query: str) -> list[dict]:
    if S.matrix is None:
        return keyword_search(query)
    try:
        v = (await embed_texts([query], "RETRIEVAL_QUERY"))[0]
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = S.matrix @ v
        order = np.argsort(-sims)[:TOP_K]
        best = float(sims[order[0]])
        picked = [S.sections[i] for i in order if sims[i] >= SIM_THRESHOLD and sims[i] >= best - 0.08]
        print(f"🔍 벡터 검색 최고 {best:.3f} → {[s['id'] for s in picked]}")
        if picked:
            return picked
    except Exception as e:
        print(f"⚠️ 벡터 검색 실패, 키워드 검색으로 대체: {e}")
    return keyword_search(query)


# ══════════════════════════════════════════════════════════════
# 6. 프롬프트
# ══════════════════════════════════════════════════════════════
SYSTEM_INSTRUCTION = f"""You are the friendly AI information guide for the event "{EVENT_NAME}".
Visitors ask about the event overview, programs and schedules, facilities (restrooms, lockers, information desk, first aid, parking, etc.), and nearby shops, restaurants and cafes.

[Rules]
1. Reply ONLY in the target language given in each request. Translate the reference information when needed.
2. Use ONLY the facts in [Reference]. Never invent times, prices, locations, phone numbers or shop names.
   If the answer is not in [Reference], say politely that you do not have that information and suggest asking at the information desk.
3. Use [Current time] to interpret words like "now", "today", "next", "still open". Do not recommend programs that have already ended today unless asked.
4. Keep it short: 2 to 4 sentences. When listing schedules or places, mention at most 4 items, woven into natural sentences.
5. Plain text only. No markdown, no bullet symbols (*, -, #, •), no emojis, no URLs. The text will be read aloud by text-to-speech.
6. Put a blank line after every 1 to 2 sentences.
7. Warm, natural spoken style, like a helpful staff member. Do not open with a greeting unless the visitor greets you.
8. Write proper nouns (venue, booth, shop names) the way visitors will see them: use the official name in the target language if the reference gives one; otherwise transliterate the Korean name.
"""


def _lang_name(lang: str) -> str:
    return LANG_CONFIG[lang]["name"]


# ══════════════════════════════════════════════════════════════
# 7. 초기화 (백그라운드)
# ══════════════════════════════════════════════════════════════
async def initialize():
    try:
        print("✨ 초기화 시작")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        S.client = genai.Client(api_key=GEMINI_API_KEY)

        try:
            S.tts = await asyncio.to_thread(
                texttospeech.TextToSpeechClient.from_service_account_file, TTS_CREDENTIALS_PATH
            )
        except Exception as e:  # TTS가 없어도 텍스트 안내는 동작하도록
            S.tts = None
            print(f"⚠️ TTS 초기화 실패 (음성 재생 비활성): {e}")

        if PDF_IMAGES_DIR.exists():
            shutil.rmtree(PDF_IMAGES_DIR)
        PDF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        if not KNOWLEDGE_PATH.exists():
            raise FileNotFoundError(f"지식 파일 없음: {KNOWLEDGE_PATH}")
        reader = _read_pdf if KNOWLEDGE_PATH.suffix.lower() == ".pdf" else _read_text
        pages = await asyncio.to_thread(reader, KNOWLEDGE_PATH)

        S.sections = _split_sections(pages)
        if not S.sections:
            raise RuntimeError("지식 파일에서 텍스트를 추출하지 못했습니다. (스캔본 PDF인지 확인)")
        S.full_context = "\n\n".join(section_block(s) for s in S.sections)
        S.catalog = build_catalog(S.sections)
        S.use_full_context = len(S.full_context) <= FULL_CONTEXT_MAX_CHARS

        print(f"📄 지식: {KNOWLEDGE_PATH.name} → 섹션 {len(S.sections)}개, {len(S.full_context):,}자")
        for s in S.sections:
            print(f"   [#{s['id']}] {s['title'][:60]}  (키워드 {len(s['keywords'])}, 이미지 {len(s['images'])})")

        if S.use_full_context:
            print("🧠 검색 방식: 전체 컨텍스트" + (f" (혼잡 시 자동 top-k, 기준 {AUTO_RAG_RATIO:.0%})" if AUTO_RAG_RATIO > 0 else ""))
            if AUTO_RAG_RATIO > 0 and len(S.sections) > TOP_K:
                try:
                    await _build_embeddings()   # 혼잡 시 자동 전환용 (실패해도 키워드 검색으로 대체)
                except Exception as e:
                    print(f"⚠️ 자동 절약용 임베딩 실패 → 혼잡 시 키워드 검색 사용: {e}")
        else:
            print("🧠 검색 방식: 임베딩 top-k")
            await _build_embeddings()

        S.ready = True
        print("🎉 초기화 완료")
    except Exception as e:
        S.error = f"[{type(e).__name__}] {e}"
        print(f"💥 초기화 실패: {S.error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    KB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(initialize())
    yield
    task.cancel()
    print("👋 서버 종료")


app = FastAPI(lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ══════════════════════════════════════════════════════════════
# 8. 공통 가드
# ══════════════════════════════════════════════════════════════
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def check_rate(request: Request, bucket: str = "chat"):
    """IP당 분당 요청 제한. chat / tts / image 를 따로 셈 (질문 1개 = chat 1회 + tts 1회라 함께 세면 금방 참)"""
    now = time.monotonic()
    if len(_hits) > 20000:
        _hits.clear()
    dq = _hits[f"{bucket}|{_client_ip(request)}"]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="rate_limited")
    dq.append(now)


def ensure_ready():
    if S.error:
        raise HTTPException(status_code=503, detail="init_failed")
    if not S.ready:
        raise HTTPException(status_code=503, detail="not_ready")


def wants_image(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in IMAGE_INTENT_WORDS)


def images_of(sections: list[dict], limit: int = 2) -> list[dict]:
    for s in sections:
        if s["images"]:
            return [{"url": u, "title": s["title"]} for u in s["images"][:limit]]
    return []


# ══════════════════════════════════════════════════════════════
# 9. 엔드포인트
# ══════════════════════════════════════════════════════════════
class HistoryItem(BaseModel):
    role: str
    text: str


class ChatReq(BaseModel):
    message: str
    lang: str = DEFAULT_LANG
    history: list[HistoryItem] = Field(default_factory=list)


class TTSReq(BaseModel):
    text: str = ""
    text_to_speak: str = ""  # 구버전 프론트 호환
    lang: str = ""


class ImageReq(BaseModel):
    image: str
    lang: str = DEFAULT_LANG


@app.get("/health")
async def health():
    if S.error:
        return JSONResponse(status_code=503, content={"status": "error", "detail": S.error})
    return {"status": "ok" if S.ready else "initializing"}


@app.get("/api/config")
async def config():
    return {
        "event_name": EVENT_NAME,
        "languages": [{"code": c, "name": v["name"]} for c, v in LANG_CONFIG.items()],
        "ready": S.ready,
        "error": bool(S.error),
        "tts": S.tts is not None,
        "sections": len(S.sections),
        "mode": "full" if S.use_full_context else "rag",
        "auto_lite": AUTO_RAG_RATIO > 0 and S.use_full_context,
        "tpm_used": _tok_used(time.monotonic()),
        "tpm_budget": TPM_BUDGET,
        "fallback_model": FALLBACK_MODEL or None,
        "fallback_answers": _fb_count,
        "primary_daily_exhausted": time.time() < _primary_off_until,
    }


@app.post("/api/chat")
async def chat(req: ChatReq, request: Request):
    check_rate(request)
    ensure_ready()
    lang = norm_lang(req.lang)
    message = req.message.strip()[:MAX_QUESTION_CHARS]
    if not message:
        raise HTTPException(status_code=400, detail="empty_message")
    history = [h for h in req.history[-6:] if h.role in ("user", "assistant")]

    # ── 같은 질문 답변 재사용 ── (저장된 답이 있으면 검색·Gemini 호출 없이 바로 응답)
    #   같은 언어 · 같은 질문(공백·문장부호 무시) · 같은 직전 대화면 ANSWER_CACHE_SEC 동안 저장된 답을 그대로 돌려줌
    #   (동시에 같은 질문이 들어오면 Gemini 는 한 번만 부르고 나머지는 그 결과를 같이 받음)
    last_ans = next((h.text for h in reversed(history) if h.role == "assistant"), "")
    norm_q = re.sub(r"[\s\W_]+", "", message.lower())
    ckey = hashlib.sha1(f"{lang}|{norm_q}|{last_ans[-200:]}".encode()).hexdigest() if ANSWER_CACHE_SEC > 0 else None
    if ckey:
        hit = _ans_cache.get(ckey)
        if hit and time.monotonic() - hit[0] < ANSWER_CACHE_SEC:
            print(f"♻️ [{lang}] {message[:40]} → 저장된 답변")
            return hit[1]
        fut = _ans_inflight.get(ckey)
        if fut:
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        _ans_inflight[ckey] = fut
        try:
            result = await _chat_answer(lang, message, history)
            _ans_cache[ckey] = (time.monotonic(), result)
            if len(_ans_cache) > 2000:
                for k in list(_ans_cache)[:500]:
                    _ans_cache.pop(k, None)
            fut.set_result(result)
            return result
        except Exception as e:
            fut.set_exception(e)
            fut.exception()   # 기다리는 쪽이 없어도 경고가 안 나도록
            raise
        finally:
            _ans_inflight.pop(ckey, None)
    return await _chat_answer(lang, message, history)


async def _build_context(message: str, history: list) -> tuple[str, bool]:
    """(컨텍스트, 절약모드 여부) — 평소엔 지식 전체, 혼잡하거나 지식이 크면 관련 섹션만"""
    lite = False
    if S.use_full_context and AUTO_RAG_RATIO > 0 and len(S.sections) > TOP_K:
        lite = _tok_used(time.monotonic()) >= TPM_BUDGET * AUTO_RAG_RATIO
    if S.use_full_context and not lite:
        return S.full_context, False
    last_user = next((h.text for h in reversed(history) if h.role == "user"), "")
    secs = await retrieve(f"{last_user}\n{message}" if last_user else message)
    if secs:
        return "\n\n".join(section_block(s) for s in secs), lite
    return section_block(S.sections[0]) + "\n\n(No closely related section. Available topics:)\n" + S.catalog, lite


_ans_cache: dict = {}        # 키 → (저장 시각, 응답)
_ans_inflight: dict = {}     # 키 → 진행 중인 Future


async def _chat_answer(lang: str, message: str, history: list) -> dict:
    context, lite = await _build_context(message, history)
    hist_text = "\n".join(
        f"{'Visitor' if h.role == 'user' else 'Guide'}: {h.text[:300]}" for h in history
    ) or "(none)"
    name = _lang_name(lang)
    # 바뀌지 않는 [Reference] 를 맨 앞에 둬야 Gemini 자동 캐시가 적중 (같은 앞부분이 반복되면 입력 토큰 할인 · 응답도 빨라짐)
    prompt = f"""[Reference]
{context}

[Target language] {name}
[Current time] {now_str()} ({EVENT_TZ})

[Recent conversation]
{hist_text}

[Visitor question]
{message}

Respond as JSON: {{"answer": "<reply in {name}>", "used_ids": [<ids of the reference sections you used>]}}"""

    try:
        raw = await gemini([prompt], system=SYSTEM_INSTRUCTION, max_tokens=1024, json_mode=True)
    except GeminiBusy:
        print(f"🚦 /api/chat 혼잡 → 잠시 후 다시 안내")
        raise HTTPException(status_code=503, detail="busy")
    except Exception as e:
        print(f"💥 /api/chat Gemini 오류: {type(e).__name__} {e}")
        raise HTTPException(status_code=502, detail="ai_error")

    data = parse_json(raw)
    answer = clean_answer(str(data.get("answer", "")) if data else raw)
    if not answer:
        raise HTTPException(status_code=502, detail="ai_empty")

    used = []
    for i in (data or {}).get("used_ids", []) or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(S.sections) and S.sections[i] not in used:
            used.append(S.sections[i])

    print(f"💬 [{lang}] {message[:40]} → 섹션 {[s['id'] for s in used]}" + (" 🪶절약" if lite else ""))
    return {
        "text": answer,
        "lang": lang,
        "sources": [s["title"] for s in used],
        "images": images_of(used) if wants_image(message) else [],
    }


_tts_cache: "OrderedDict[str, str]" = OrderedDict()


@app.post("/api/tts")
async def tts(req: TTSReq, request: Request):
    check_rate(request, "tts")
    if not S.tts:
        raise HTTPException(status_code=503, detail="tts_unavailable")
    text = (req.text or req.text_to_speak).strip()[:MAX_TTS_CHARS]
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")
    lang = norm_lang(req.lang)
    cfg = LANG_CONFIG[lang]

    key = hashlib.sha1(f"{lang}|{cfg['voice']}|{text}".encode()).hexdigest()
    if key in _tts_cache:
        _tts_cache.move_to_end(key)
        return {"audio": _tts_cache[key], "mime": "audio/mpeg"}

    try:
        r = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code=cfg["tts_lang"], name=cfg["voice"]),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=TTS_SPEAKING_RATE,
            ),
        )
        resp = await asyncio.to_thread(S.tts.synthesize_speech, request=r)
    except Exception as e:
        print(f"💥 TTS 오류: {e}")
        raise HTTPException(status_code=502, detail="tts_error")

    audio = base64.b64encode(resp.audio_content).decode("ascii")
    _tts_cache[key] = audio
    if len(_tts_cache) > 300:
        _tts_cache.popitem(last=False)
    return {"audio": audio, "mime": "audio/mpeg"}


def _prepare_image(data: str) -> bytes:
    b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    raw = base64.b64decode(b64)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.post("/api/recognize-image")
async def recognize_image(req: ImageReq, request: Request):
    check_rate(request, "image")
    ensure_ready()
    lang = norm_lang(req.lang)
    name = _lang_name(lang)
    try:
        jpeg = await asyncio.to_thread(_prepare_image, req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="bad_image")

    # 1단계: 간판 OCR + 섹션 분류
    classify_prompt = f"""A visitor at the event "{EVENT_NAME}" took this photo, most likely of a sign: a signboard, banner, booth name, shop sign, direction board, or facility pictogram.
1. Read all legible text in the photo (any language).
2. Decide which catalog items the photo refers to. Match by the name on the sign, the keywords, or clear meaning (for example a restroom pictogram matches the restroom item). Prefer exact name matches.
3. Return JSON only: {{"ocr_text": "...", "matches": [best id first, at most 2], "confidence": "high" | "medium" | "low"}}. Use [] when nothing matches.

[Catalog]
{S.catalog}"""
    try:
        raw = await gemini(
            [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), classify_prompt],
            max_tokens=512, json_mode=True, temperature=0.1,
        )
    except Exception as e:
        print(f"💥 간판 인식 오류: {type(e).__name__} {e}")
        raise HTTPException(status_code=502, detail="ai_error")

    data = parse_json(raw) or {}
    ocr = str(data.get("ocr_text") or "").strip()
    ids = []
    for i in data.get("matches", []) or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(S.sections) and i not in ids:
            ids.append(i)
    secs = [S.sections[i] for i in ids[:2]]
    if not secs and ocr:
        secs = keyword_search(ocr, limit=2)
    print(f"📷 OCR: {ocr[:60]!r} → 섹션 {[s['id'] for s in secs]} ({data.get('confidence')})")

    if not secs:
        key = "sign_no_match" if ocr else "sign_unreadable"
        return {"status": "no_match", "title": "", "description": MSG[key][lang],
                "ocr_text": ocr, "images": []}

    # 2단계: 선택 언어로 안내문 생성
    context = "\n\n".join(section_block(s) for s in secs)
    explain_prompt = f"""[Target language] {name}
[Current time] {now_str()} ({EVENT_TZ})

The visitor scanned a sign at the venue. Text read from the sign: "{ocr or '(none)'}"

[Reference]
{context}

Explain what this place, booth, program or shop is and the most useful facts for a visitor right now (location, hours, what to do there, next schedule), using ONLY [Reference].
Respond as JSON: {{"title": "<short name in {name}>", "description": "<2 to 4 sentences in {name}>"}}"""
    try:
        raw2 = await gemini([explain_prompt], system=SYSTEM_INSTRUCTION, max_tokens=1024, json_mode=True)
    except Exception as e:
        print(f"💥 간판 안내 생성 오류: {type(e).__name__} {e}")
        raise HTTPException(status_code=502, detail="ai_error")

    d2 = parse_json(raw2) or {}
    description = clean_answer(str(d2.get("description") or raw2))
    title = str(d2.get("title") or secs[0]["title"]).strip()
    return {
        "status": "success",
        "title": title,
        "description": description,
        "images": images_of(secs),
        "section_ids": [s["id"] for s in secs],
    }


# ── 정적 파일 ──
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")
