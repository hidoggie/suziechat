/* ──────────────────────────────────────────
   배포 환경별 설정 (이 파일만 고치면 됨)
────────────────────────────────────────── */
window.APP_CONFIG = {
  // API 서버 주소. suziechat 에서 열든 stechart 에 올려서 열든 이 서버로 요청합니다.
  // (stechart 에서 부를 때는 서버의 CORS_ORIGINS 에 stechart 도메인이 있어야 함 — main.py 기본값에 포함)
  API_BASE: 'https://suziechat.onrender.com',

  // 뒤로가기 버튼: 이전 화면 기록이 없을 때 이동할 주소 (스탬프투어 메인)
  HOME_URL: 'https://stechart.stampquest.kr/',

  // 헤더 "길안내 받기" 버튼 주소
  ROUTE_URL: 'https://stechart.stampquest.kr/index.html?view=venue',
};
