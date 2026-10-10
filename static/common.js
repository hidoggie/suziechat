/* ══════════════════════════════════════════════════════════
   common.js — 공용 스크립트
   - UI 다국어(ko/en/ja/zh)
   - API 호출 (타임아웃·에러 메시지)
   - TTS 플레이어 (Web Audio, iOS 자동재생 대응)
   - 카카오톡 등 인앱 브라우저 안내
   페이지에서는 config.js → common.js 순으로 불러온 뒤 window.App 사용
══════════════════════════════════════════════════════════ */
(function () {
  const CFG = window.APP_CONFIG || {};
  const API_BASE = (CFG.API_BASE || '').replace(/\/+$/, '');

  const LANGS = [
    { code: 'ko', label: '한국어', speech: 'ko-KR', html: 'ko' },
    { code: 'en', label: 'English', speech: 'en-US', html: 'en' },
    { code: 'ja', label: '日本語', speech: 'ja-JP', html: 'ja' },
    { code: 'zh', label: '中文', speech: 'zh-CN', html: 'zh-CN' },
    { code: 'vi', label: 'Tiếng Việt', speech: 'vi-VN', html: 'vi' },
  ];

  /* ── UI 문구 (행사에 맞게 자유롭게 수정) ── */
  const T = {
    ko: {
      app_title: '글로벌 AI 도슨트',
      welcome: '안녕하세요! 행사 안내봇입니다. 궁금한 것을 물어보시거나 행사장 간판을 비춰 보세요. 프로그램, 편의시설, 주변 맛집까지 안내해 드릴게요.',
      btn_chat: 'AI 안내 챗봇', btn_chat_sub: '음성·텍스트로 질문하기',
      btn_scan: '간판 스캔', btn_scan_sub: '간판을 비추면 정보를 알려드려요',
      chat_intro: '안녕하세요! 행사 프로그램, 편의시설, 주변 맛집 등 궁금한 것을 물어보세요. 아래 버튼을 눌러도 돼요.',
      placeholder: '궁금한 것을 입력하세요...',
      preparing: '서버를 준비하고 있어요. 잠시만 기다려 주세요...',
      ready_msg: '준비가 완료되었어요.',
      err_not_ready: '서버를 준비하고 있어요. 잠시 후 다시 시도해 주세요.',
      err_init: '서버 설정 오류로 서비스를 이용할 수 없어요. 운영자에게 문의해 주세요.',
      err_net: '네트워크 연결을 확인한 뒤 다시 시도해 주세요.',
      err_rate: '요청이 많아요. 잠시 후 다시 시도해 주세요.',
      err_server: '일시적인 오류가 발생했어요. 다시 시도해 주세요.',
      listening: '듣고 있어요...', speak_now: '질문을 말씀해 주세요',
      cancel: '취소', send: '보내기',
      voice_unsupported: '이 브라우저에서는 음성 입력을 지원하지 않아요. 크롬이나 사파리에서 열어 주세요.',
      mic_denied: '마이크 권한이 필요해요. 브라우저 설정에서 허용해 주세요.',
      voice_error: '음성을 알아듣지 못했어요. 다시 시도해 주세요.',
      tts_error: '음성을 재생하지 못했어요.',
      lang_changed: '언어를 {lang}(으)로 바꿨어요.',
      chips: [
        { label: '행사 개요', q: '이번 행사를 간단히 소개해 주세요.' },
        { label: '프로그램', q: '오늘 프로그램 일정을 알려주세요.' },
        { label: '편의시설', q: '화장실, 물품보관소 같은 편의시설은 어디에 있나요?' },
        { label: '주변 맛집', q: '행사장 주변 맛집이나 카페를 추천해 주세요.' },
      ],
      inapp_msg: '앱 안의 브라우저에서는 마이크·카메라가 동작하지 않을 수 있어요.',
      inapp_btn: '외부 브라우저로 열기',
      inapp_copied: '주소를 복사했어요. 사파리나 크롬에 붙여넣어 열어 주세요.',
      scan_hint: '간판이 화면에 잘 보이도록 맞춘 뒤 버튼을 눌러 주세요',
      scanning: '간판을 읽는 중...', align: '카메라를 간판에 맞춰 주세요',
      camera_error: '카메라를 사용할 수 없어요. 권한을 확인해 주세요.',
      camera_not_ready: '카메라가 아직 준비되지 않았어요. 잠시 후 다시 눌러 주세요.',
      ask_more: '이곳에 대해 더 물어보기',
      ask_more_q: '{title}에 대해 더 자세히 알려주세요.',
      home: '홈', chat: '챗봇', scan: '스캔',
    },
    en: {
      app_title: 'Global AI Docent',
      welcome: 'Hello! I am your event guide. Ask me anything, or point your camera at a sign at the venue. I can tell you about programs, facilities and places to eat nearby.',
      btn_chat: 'AI Guide Chat', btn_chat_sub: 'Ask by voice or text',
      btn_scan: 'Scan a Sign', btn_scan_sub: 'Point your camera at a sign',
      chat_intro: 'Hello! Ask me about programs, facilities, nearby restaurants and more. You can also tap a button below.',
      placeholder: 'Type your question...',
      preparing: 'The server is starting up. Please wait a moment...',
      ready_msg: 'Ready to go.',
      err_not_ready: 'The server is starting up. Please try again shortly.',
      err_init: 'The service is unavailable due to a server configuration error. Please contact the staff.',
      err_net: 'Please check your network connection and try again.',
      err_rate: 'Too many requests. Please try again in a moment.',
      err_server: 'Something went wrong. Please try again.',
      listening: 'Listening...', speak_now: 'Please say your question',
      cancel: 'Cancel', send: 'Send',
      voice_unsupported: 'Voice input is not supported in this browser. Please open this page in Chrome or Safari.',
      mic_denied: 'Microphone permission is required. Please allow it in your browser settings.',
      voice_error: "Sorry, I didn't catch that. Please try again.",
      tts_error: 'Could not play the audio.',
      lang_changed: 'Language changed to {lang}.',
      chips: [
        { label: 'About', q: 'Please give me a short introduction to this event.' },
        { label: 'Programs', q: "What is today's program schedule?" },
        { label: 'Facilities', q: 'Where are the facilities like restrooms and lockers?' },
        { label: 'Food nearby', q: 'Can you recommend restaurants or cafes near the venue?' },
      ],
      inapp_msg: 'The microphone and camera may not work in this in-app browser.',
      inapp_btn: 'Open in browser',
      inapp_copied: 'Link copied. Paste it into Safari or Chrome to open.',
      scan_hint: 'Fit the sign in the frame, then tap the button',
      scanning: 'Reading the sign...', align: 'Point your camera at the sign',
      camera_error: 'The camera is unavailable. Please check the permission.',
      camera_not_ready: 'The camera is not ready yet. Please try again in a moment.',
      ask_more: 'Ask more about this place',
      ask_more_q: 'Tell me more about {title}.',
      home: 'Home', chat: 'Chat', scan: 'Scan',
    },
    ja: {
      app_title: 'グローバルAIドーセント',
      welcome: 'こんにちは！イベント案内ボットです。質問するか、会場の看板にカメラをかざしてください。プログラム、施設、周辺のグルメまでご案内します。',
      btn_chat: 'AI案内チャット', btn_chat_sub: '音声・テキストで質問',
      btn_scan: '看板スキャン', btn_scan_sub: '看板にカメラをかざしてください',
      chat_intro: 'こんにちは！プログラム、施設、周辺のお店など、気になることを聞いてください。下のボタンも使えます。',
      placeholder: '質問を入力してください...',
      preparing: 'サーバーを準備しています。少々お待ちください...',
      ready_msg: '準備ができました。',
      err_not_ready: 'サーバーを準備しています。しばらくしてからもう一度お試しください。',
      err_init: 'サーバー設定エラーのため利用できません。スタッフにお問い合わせください。',
      err_net: 'ネットワーク接続を確認して、もう一度お試しください。',
      err_rate: 'リクエストが多すぎます。しばらくしてからお試しください。',
      err_server: '一時的なエラーが発生しました。もう一度お試しください。',
      listening: '聞いています...', speak_now: '質問をお話しください',
      cancel: 'キャンセル', send: '送信',
      voice_unsupported: 'このブラウザは音声入力に対応していません。ChromeまたはSafariで開いてください。',
      mic_denied: 'マイクの許可が必要です。ブラウザの設定で許可してください。',
      voice_error: '聞き取れませんでした。もう一度お試しください。',
      tts_error: '音声を再生できませんでした。',
      lang_changed: '言語を{lang}に変更しました。',
      chips: [
        { label: 'イベント概要', q: 'このイベントについて簡単に紹介してください。' },
        { label: 'プログラム', q: '今日のプログラムのスケジュールを教えてください。' },
        { label: '施設案内', q: 'トイレやコインロッカーなどの施設はどこですか？' },
        { label: '周辺グルメ', q: '会場周辺のおすすめの飲食店やカフェを教えてください。' },
      ],
      inapp_msg: 'アプリ内ブラウザではマイクやカメラが動作しない場合があります。',
      inapp_btn: '外部ブラウザで開く',
      inapp_copied: 'リンクをコピーしました。SafariまたはChromeに貼り付けて開いてください。',
      scan_hint: '看板が画面に収まるように合わせてボタンを押してください',
      scanning: '看板を読み取り中...', align: 'カメラを看板に向けてください',
      camera_error: 'カメラを使用できません。許可設定を確認してください。',
      camera_not_ready: 'カメラの準備ができていません。少し待ってからもう一度押してください。',
      ask_more: 'ここについてもっと聞く',
      ask_more_q: '{title}についてもっと詳しく教えてください。',
      home: 'ホーム', chat: 'チャット', scan: 'スキャン',
    },
    zh: {
      app_title: '全球AI讲解员',
      welcome: '您好！我是活动导览机器人。您可以直接提问，或将相机对准会场的指示牌。我会为您介绍活动节目、便民设施和周边美食。',
      btn_chat: 'AI导览聊天', btn_chat_sub: '语音或文字提问',
      btn_scan: '扫描指示牌', btn_scan_sub: '将相机对准指示牌',
      chat_intro: '您好！欢迎询问活动节目、便民设施、周边美食等信息，也可以点击下方按钮。',
      placeholder: '请输入您的问题...',
      preparing: '服务器正在启动，请稍候...',
      ready_msg: '准备就绪。',
      err_not_ready: '服务器正在启动，请稍后再试。',
      err_init: '服务器配置错误，暂时无法使用。请联系工作人员。',
      err_net: '请检查网络连接后重试。',
      err_rate: '请求过多，请稍后再试。',
      err_server: '出现暂时性错误，请重试。',
      listening: '正在聆听...', speak_now: '请说出您的问题',
      cancel: '取消', send: '发送',
      voice_unsupported: '此浏览器不支持语音输入，请使用 Chrome 或 Safari 打开。',
      mic_denied: '需要麦克风权限，请在浏览器设置中允许。',
      voice_error: '没有听清，请再试一次。',
      tts_error: '无法播放语音。',
      lang_changed: '语言已切换为{lang}。',
      chips: [
        { label: '活动介绍', q: '请简单介绍一下这次活动。' },
        { label: '节目安排', q: '请告诉我今天的节目安排。' },
        { label: '便民设施', q: '洗手间、寄存处等便民设施在哪里？' },
        { label: '周边美食', q: '请推荐会场周边的餐厅或咖啡馆。' },
      ],
      inapp_msg: '在应用内浏览器中，麦克风和相机可能无法使用。',
      inapp_btn: '在浏览器中打开',
      inapp_copied: '链接已复制，请粘贴到 Safari 或 Chrome 中打开。',
      scan_hint: '让指示牌完整出现在画面中，然后点击按钮',
      scanning: '正在识别指示牌...', align: '请将相机对准指示牌',
      camera_error: '无法使用相机，请检查权限设置。',
      camera_not_ready: '相机尚未就绪，请稍后再试。',
      ask_more: '进一步了解这里',
      ask_more_q: '请详细介绍一下{title}。',
      home: '首页', chat: '聊天', scan: '扫描',
    },
    // 아래부터 베트남어(vi) 번역 블록 추가
    vi: {
      app_title: 'AI Docent toàn cầu',
      welcome: 'Xin chào! Tôi là trợ lý AI hướng dẫn sự kiện. Bạn có thể đặt câu hỏi hoặc hướng camera vào bảng hiệu tại hội trường. Tôi sẽ hướng dẫn bạn về các chương trình, tiện ích và địa điểm ăn uống lân cận.',
      btn_chat: 'Trò chuyện AI', btn_chat_sub: 'Hỏi bằng giọng nói/văn bản',
      btn_scan: 'Quét bảng hiệu', btn_scan_sub: 'Hướng camera để nhận thông tin',
      chat_intro: 'Xin chào! Hãy hỏi tôi về chương trình, tiện ích, nhà hàng lân cận hoặc chọn một nút bên dưới.',
      placeholder: 'Nhập câu hỏi của bạn...',
      preparing: 'Máy chủ đang khởi động. Vui lòng đợi trong giây lát...',
      ready_msg: 'Đã sẵn sàng.',
      err_not_ready: 'Máy chủ đang khởi động. Vui lòng thử lại sau một lát.',
      err_init: 'Dịch vụ hiện không khả dụng do lỗi cấu hình máy chủ. Vui lòng liên hệ nhân viên.',
      err_net: 'Vui lòng kiểm tra kết nối mạng và thử lại.',
      err_rate: 'Quá nhiều yêu cầu. Vui lòng thử lại sau.',
      err_server: 'Đã xảy ra lỗi tạm thời. Vui lòng thử lại.',
      listening: 'Đang nghe...', speak_now: 'Vui lòng nói câu hỏi của bạn',
      cancel: 'Hủy', send: 'Gửi',
      voice_unsupported: 'Trình duyệt này không hỗ trợ giọng nói. Vui lòng mở bằng Chrome hoặc Safari.',
      mic_denied: 'Cần có quyền truy cập micro. Vui lòng cho phép trong cài đặt trình duyệt.',
      voice_error: 'Không nghe rõ. Vui lòng thử lại.',
      tts_error: 'Không thể phát âm thanh.',
      lang_changed: 'Ngôn ngữ đã được chuyển sang {lang}.',
      chips: [
        { label: 'Giới thiệu', q: 'Hãy giới thiệu ngắn gọn về sự kiện này.' },
        { label: 'Chương trình', q: 'Lịch trình sự kiện hôm nay là gì?' },
        { label: 'Tiện ích', q: 'Các tiện ích như nhà vệ sinh, tủ khóa ở đâu?' },
        { label: 'Ẩm thực', q: 'Vui lòng gợi ý các nhà hàng hoặc quán cà phê gần đây.' },
      ],
      inapp_msg: 'Micro và camera có thể không hoạt động trong trình duyệt của ứng dụng này.',
      inapp_btn: 'Mở trình duyệt ngoài',
      inapp_copied: 'Đã sao chép liên kết. Hãy dán vào Safari hoặc Chrome để mở.',
      scan_hint: 'Chỉnh bảng hiệu vừa với màn hình rồi nhấn nút',
      scanning: 'Đang đọc bảng hiệu...', align: 'Hướng camera về phía bảng hiệu',
      camera_error: 'Không thể sử dụng camera. Vui lòng kiểm tra quyền truy cập.',
      camera_not_ready: 'Camera chưa sẵn sàng. Vui lòng thử lại sau.',
      ask_more: 'Hỏi thêm về nơi này',
      ask_more_q: 'Hãy cho tôi biết thêm chi tiết về {title}.',
      home: 'Trang chủ', chat: 'Trò chuyện', scan: 'Quét',
    },
  };

  /* ── 언어 상태 ── */
  function normalize(code) {
    if (!code) return null;
    const c = String(code).toLowerCase();
    if (c.startsWith('zh')) return 'zh';
    const s = c.split('-')[0];
    return LANGS.some(l => l.code === s) ? s : null;
  }
  function readStore() { try { return localStorage.getItem('guide_lang'); } catch (e) { return null; } }
  function writeStore(c) { try { localStorage.setItem('guide_lang', c); } catch (e) {} }

  let current = normalize(new URLSearchParams(location.search).get('lang'))
    || normalize(readStore())
    || normalize(navigator.language)
    || 'ko';

  function langInfo(code) { return LANGS.find(l => l.code === (code || current)) || LANGS[0]; }

  function t(key, vars) {
    let s = (T[current] && T[current][key] !== undefined) ? T[current][key] : (T.ko[key] !== undefined ? T.ko[key] : key);
    if (vars && typeof s === 'string') s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] !== undefined ? vars[k] : m));
    return s;
  }

  function applyI18n(root) {
    root = root || document;
    root.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    root.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); el.setAttribute('aria-label', el.title); });
    document.documentElement.lang = langInfo().html;
    document.title = t('app_title');
  }

  function setLang(code) {
    const c = normalize(code) || 'ko';
    if (c === current) return;
    current = c;
    writeStore(c);
    applyI18n();
    window.dispatchEvent(new CustomEvent('langchange', { detail: c }));
  }

  /* <select> 하나로 언어 선택 UI 구성 */
  function bindLangSelect(sel) {
    sel.innerHTML = '';
    LANGS.forEach(l => {
      const o = document.createElement('option');
      o.value = l.code; o.textContent = l.label;
      sel.appendChild(o);
    });
    sel.value = current;
    sel.addEventListener('change', () => setLang(sel.value));
    window.addEventListener('langchange', () => { sel.value = current; });
  }

  /* 페이지 이동 시 언어 유지 (localStorage가 막힌 환경 대비) */
  function withLang(url, extra) {
    const u = new URL(url, location.href);
    u.searchParams.set('lang', current);
    if (extra) Object.entries(extra).forEach(([k, v]) => u.searchParams.set(k, v));
    return u.origin === location.origin ? u.pathname + u.search : u.toString();
  }

  /* ── API ── */
  function apiUrl(path) { return /^https?:\/\//.test(path) ? path : API_BASE + path; }

  async function request(method, path, body, timeoutMs, signal) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs || 30000);
    if (signal) {
      if (signal.aborted) ctrl.abort();
      else signal.addEventListener('abort', () => ctrl.abort(), { once: true });
    }
    try {
      const res = await fetch(apiUrl(path), {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: ctrl.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (e) {}
      if (!res.ok) {
        const err = new Error((data && data.detail) || ('HTTP ' + res.status));
        err.status = res.status;
        err.code = data && data.detail;
        throw err;
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }
  const apiGet = (path, timeoutMs, signal) => request('GET', path, null, timeoutMs, signal);
  const apiPost = (path, body, timeoutMs, signal) => request('POST', path, body, timeoutMs, signal);

  function errorText(err) {
    if (err && err.status === 429) return t('err_rate');
    if (err && err.status === 503) return err.code === 'init_failed' ? t('err_init') : t('err_not_ready');
    if (err && (err.name === 'AbortError' || err instanceof TypeError)) return t('err_net');
    return t('err_server');
  }

  /* ── 텍스트 렌더링 (\n\n → 문단, \n → 줄바꿈 / innerHTML 미사용) ── */
  function renderText(el, raw) {
    el.textContent = '';
    String(raw || '').split(/\n{2,}/).forEach((para, i) => {
      if (i) { el.appendChild(document.createElement('br')); el.appendChild(document.createElement('br')); }
      para.split('\n').forEach((line, j) => {
        if (j) el.appendChild(document.createElement('br'));
        el.appendChild(document.createTextNode(line));
      });
    });
  }

  /* ── TTS 플레이어 ──
     Web Audio 사용. unlock()을 반드시 '사용자 탭' 이벤트 안에서 먼저 호출해야
     이후 비동기로 받은 음성도 iOS에서 재생됨. */
  class TTSPlayer {
    constructor() {
      this.ctx = null; this.buffer = null; this.source = null;
      this.offset = 0; this.startedAt = 0; this.playing = false;
      this.key = null; this.onstate = () => {}; this._raf = 0;
    }
    unlock() {
      try {
        if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.ctx.state === 'suspended') this.ctx.resume();
        const b = this.ctx.createBuffer(1, 1, 22050);
        const s = this.ctx.createBufferSource();
        s.buffer = b; s.connect(this.ctx.destination); s.start(0);
      } catch (e) {}
    }
    get duration() { return this.buffer ? this.buffer.duration : 0; }
    get currentTime() {
      return this.playing ? Math.min(this.ctx.currentTime - this.startedAt, this.duration) : this.offset;
    }
    isLoaded(text, lang) { return !!this.buffer && this.key === lang + '|' + text; }
    async load(text, lang) {
      const key = lang + '|' + text;
      if (this.key === key && this.buffer) return;
      if (!this.ctx) this.unlock();
      this.stop(); this.buffer = null; this.key = key;
      const data = await apiPost('/api/tts', { text, lang }, 30000);
      const bin = Uint8Array.from(atob(data.audio), c => c.charCodeAt(0));
      const buf = await this.ctx.decodeAudioData(bin.buffer);
      if (this.key !== key) return;
      this.buffer = buf; this.offset = 0;
      this._emit('loaded');
    }
    play() {
      if (!this.buffer) return;
      if (this.ctx.state === 'suspended') this.ctx.resume();
      this._stopSource();
      if (this.offset >= this.duration - 0.05) this.offset = 0;
      const s = this.ctx.createBufferSource();
      s.buffer = this.buffer;
      s.connect(this.ctx.destination);
      s.start(0, this.offset);
      this.startedAt = this.ctx.currentTime - this.offset;
      s.onended = () => {
        if (this.source !== s) return;
        this.source = null; this.playing = false; this.offset = 0;
        this._emit('ended');
      };
      this.source = s; this.playing = true;
      this._emit('play'); this._tick();
    }
    pause() {
      if (!this.playing) return;
      this.offset = this.currentTime;
      this._stopSource(); this.playing = false;
      this._emit('pause');
    }
    seek(t) {
      if (!this.buffer) return;
      const was = this.playing;
      if (was) { this._stopSource(); this.playing = false; }
      this.offset = Math.max(0, Math.min(t, this.duration));
      if (was) this.play(); else this._emit('seek');
    }
    stop() { this._stopSource(); this.playing = false; this.offset = 0; this._emit('stop'); }
    reset() { this.stop(); this.buffer = null; this.key = null; }
    _stopSource() {
      if (this.source) {
        this.source.onended = null;
        try { this.source.stop(); } catch (e) {}
        this.source = null;
      }
    }
    _emit(ev) { try { this.onstate(ev, this); } catch (e) { console.error(e); } }
    _tick() {
      cancelAnimationFrame(this._raf);
      const f = () => { if (!this.playing) return; this._emit('progress'); this._raf = requestAnimationFrame(f); };
      this._raf = requestAnimationFrame(f);
    }
  }

  /* ── 인앱 브라우저 안내 (카카오톡·네이버·인스타 등) ── */
  function isInApp() {
    return /KAKAOTALK|NAVER\(inapp|Instagram|FBAN|FBAV|Line\/|DaumApps|everytimeApp|BAND\//i.test(navigator.userAgent);
  }
  function openExternal() {
    const url = location.href, ua = navigator.userAgent;
    if (/KAKAOTALK/i.test(ua)) {
      location.href = 'kakaotalk://web/openExternal?url=' + encodeURIComponent(url);
      return;
    }
    if (/Android/i.test(ua)) {
      location.href = 'intent://' + url.replace(/^https?:\/\//, '') + '#Intent;scheme=https;package=com.android.chrome;end';
      return;
    }
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(() => alert(t('inapp_copied'))).catch(() => prompt('', url));
    } else {
      prompt('', url);
    }
  }
  function mountInAppBanner(parent) {
    if (!parent || !isInApp()) return;
    const bar = document.createElement('div');
    bar.style.cssText = 'display:flex;align-items:center;gap:10px;padding:10px 16px;background:#17171A;border-bottom:1px solid #2A2A30;color:#E4E4EA;font-size:13px;line-height:1.4;';
    const msg = document.createElement('span');
    msg.style.flex = '1';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.style.cssText = 'flex-shrink:0;height:32px;padding:0 12px;border-radius:999px;border:none;background:#F5FB20;color:#0A0A0B;font-weight:700;font-size:12px;font-family:inherit;';
    btn.addEventListener('click', openExternal);
    const update = () => { msg.textContent = t('inapp_msg'); btn.textContent = t('inapp_btn'); };
    update();
    window.addEventListener('langchange', update);
    bar.append(msg, btn);
    parent.prepend(bar);
  }

  /* 서버 준비 대기 (콜드스타트 대응). onWaiting은 처음 대기할 때 1회 호출 */
  async function waitReady(onWaiting) {
    let waited = false;
    for (let i = 0; i < 80; i++) {
      try {
        const c = await apiGet('/api/config', 8000);
        if (c.error) { const e = new Error('init_failed'); e.status = 503; e.code = 'init_failed'; throw e; }
        if (c.ready) return { waited, config: c };
      } catch (e) {
        if (e.code === 'init_failed') throw e;
      }
      if (!waited) { waited = true; if (onWaiting) onWaiting(); }
      await new Promise(r => setTimeout(r, 3000));
    }
    throw new Error('timeout');
  }

  function formatTime(sec) {
    sec = Math.max(0, sec || 0);
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  document.documentElement.lang = langInfo().html;

  window.App = {
    LANGS, t, applyI18n, setLang, bindLangSelect, withLang,
    getLang: () => current,
    langLabel: (c) => langInfo(c).label,
    speechCode: () => langInfo().speech,
    apiUrl, apiGet, apiPost, errorText, renderText,
    TTSPlayer, mountInAppBanner, waitReady, formatTime,
    homeUrl: CFG.HOME_URL || '/',
    routeUrl: CFG.ROUTE_URL || '',
  };
})();
