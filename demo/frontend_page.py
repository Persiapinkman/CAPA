HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RD-CLAW</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --panel-soft: #fafbff;
      --line: #e3e8f2;
      --text: #1f2937;
      --muted: #6b7280;
      --brand: #2563eb;
      --brand-soft: #eef4ff;
      --radius: 14px;
      --font: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font);
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
    }
    .layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .brand {
      font-size: 18px;
      letter-spacing: .12em;
      font-weight: 700;
      color: #111827;
      padding: 4px 8px 10px;
    }
    .new-chat-btn {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      color: #1f2937;
      padding: 10px 12px;
      cursor: pointer;
      text-align: left;
      font-size: 14px;
      transition: all .2s ease;
    }
    .new-chat-btn:hover {
      background: #f3f6fd;
      border-color: #cfd8ea;
    }
    .session-list {
      margin-top: 2px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-right: 2px;
    }
    .session-item {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 12px;
      padding: 10px;
      cursor: pointer;
      transition: all .2s ease;
      position: relative;
      padding-right: 40px;
    }
    .session-item:hover {
      border-color: #cfd8ea;
      background: #f8faff;
    }
    .session-item.active {
      border-color: #b9ccf3;
      background: #eef4ff;
    }
    .session-title {
      font-size: 13px;
      font-weight: 600;
      color: #111827;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .session-meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .session-delete {
      position: absolute;
      right: 8px;
      top: 8px;
      border: 1px solid #e4e9f5;
      background: #fff;
      color: #6b7280;
      border-radius: 8px;
      width: 24px;
      height: 24px;
      line-height: 20px;
      text-align: center;
      cursor: pointer;
      font-size: 14px;
      padding: 0;
      visibility: hidden;
      opacity: 0;
    }
    .session-item:hover .session-delete {
      visibility: visible;
      opacity: 1;
    }
    .session-delete:hover {
      color: #b91c1c;
      border-color: #f3c4c4;
      background: #fff5f5;
    }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: 1fr auto;
    }
    #chat {
      padding: 18px 28px 24px;
      overflow-y: auto;
      min-height: 0;
    }
    .msg {
      display: flex;
      margin: 12px 0;
    }
    .msg.user { justify-content: flex-end; }
    .bubble {
      width: min(100%, 1200px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px 16px;
      background: #fff;
      box-shadow: 0 4px 14px rgba(15, 23, 42, .06);
      line-height: 1.6;
    }
    .msg.user .bubble {
      width: auto;
      max-width: min(76%, 860px);
      background: #edf3ff;
      border-color: #d5e3ff;
    }
    .bubble img.preview {
      margin-top: 10px;
      max-width: 260px;
      max-height: 220px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,.25);
      object-fit: contain;
      background: #fff;
    }
    .bubble .preview-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .bubble .preview-grid img.preview {
      margin-top: 0;
      max-width: 120px;
      max-height: 120px;
    }
    .section {
      margin-top: 10px;
      border-top: 1px dashed #d7e0ef;
      padding-top: 10px;
      opacity: 0;
      transform: translateY(8px);
      animation: fadeIn .25s ease forwards;
    }
    @keyframes fadeIn {
      to { opacity: 1; transform: translateY(0); }
    }
    .section-title {
      color: #2563eb;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .05em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .pending-msg { color: var(--muted); font-size: 12px; }
    .kv-row { margin: 4px 0; }
    .kv-row b { color: var(--muted); margin-right: 6px; }
    .adela-flowchart {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 24px minmax(0, 1fr) 24px minmax(0, 1fr);
      align-items: start;
      gap: 8px;
      margin-top: 4px;
    }
    .adela-stage {
      border: 1px solid #dbe5f6;
      border-radius: 12px;
      background: #f8fbff;
      padding: 10px;
    }
    .adela-stage-title {
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
      letter-spacing: .03em;
    }
    .adela-arrow {
      text-align: center;
      color: #9ca3af;
      font-size: 18px;
      line-height: 1;
      margin-top: 28px;
      user-select: none;
    }
    .adela-node {
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fff;
      padding: 8px 10px;
      margin-bottom: 8px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, .03);
    }
    .adela-node:last-child { margin-bottom: 0; }
    .adela-node__title {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      font-weight: 600;
      color: #374151;
    }
    .adela-node__sub {
      margin-top: 4px;
      font-size: 12px;
      color: #6b7280;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .adela-node--success { border-color: #86efac; background: #f0fdf4; }
    .adela-node--running { border-color: #bfdbfe; background: #eff6ff; }
    .adela-node--failed { border-color: #fecaca; background: #fff1f2; }
    .adela-node--skipped { border-color: #e5e7eb; background: #f9fafb; opacity: .86; }
    .adela-node--pending { border-style: dashed; background: #fff; }
    .adela-dots {
      display: inline-block;
      margin-left: 2px;
      letter-spacing: 1px;
    }
    .adela-dots span {
      display: inline-block;
      animation: adelaDotBounce 1.1s ease-in-out infinite;
      opacity: 0.35;
    }
    .adela-dots span:nth-child(2) { animation-delay: 0.18s; }
    .adela-dots span:nth-child(3) { animation-delay: 0.36s; }
    @keyframes adelaDotBounce {
      0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-2px); }
    }
    .adela-slot-form {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .adela-slot-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .adela-field label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .adela-field input,
    .adela-field select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
      outline: none;
    }
    .adela-field input:focus,
    .adela-field select:focus {
      border-color: #9bbcf4;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, .10);
    }
    .adela-form-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .adela-submit {
      height: 36px;
      border: 0;
      border-radius: 8px;
      padding: 0 14px;
      color: #fff;
      background: #2563eb;
      cursor: pointer;
      font-weight: 600;
    }
    .adela-submit:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .adela-form-error {
      color: #b91c1c;
      font-size: 12px;
    }
    .rag-feedback {
      display: grid;
      gap: 8px;
    }
    .rag-feedback textarea {
      width: 100%;
      min-height: 72px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      resize: vertical;
      outline: none;
    }
    .rag-feedback-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .rag-feedback button {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      cursor: pointer;
      padding: 0 10px;
      color: #1f2937;
    }
    .rag-feedback button.primary {
      border-color: transparent;
      background: #2563eb;
      color: #fff;
      font-weight: 600;
    }
    .rag-feedback button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .advisor-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .advisor-actions button {
      border: 1px solid #cfd8ea;
      background: #fff;
      color: #1f2937;
      border-radius: 10px;
      padding: 8px 10px;
      cursor: pointer;
      font-size: 13px;
    }
    .advisor-actions button.primary {
      background: #2563eb;
      color: #fff;
      border-color: #2563eb;
    }
    .advisor-actions button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .rag-feedback-status {
      color: var(--muted);
      font-size: 12px;
    }
    .thinking-toggle {
      border: 1px solid #d5e1f7;
      border-radius: 10px;
      padding: 8px 10px;
      background: #f7faff;
    }
    .thinking-toggle summary {
      cursor: pointer;
      color: #2563eb;
      font-weight: 600;
      font-size: 12px;
      line-height: 1.4;
      user-select: none;
    }
    .thinking-content { margin-top: 8px; }
    .thumb-row {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
      white-space: nowrap;
    }
    .thumb-row img {
      width: 150px;
      height: 150px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      flex: 0 0 auto;
    }
    .composer-wrap {
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, .96);
      backdrop-filter: blur(6px);
      padding: 12px 22px 16px;
    }
    .status { color: var(--muted); font-size: 12px; padding: 0 0 8px 2px; }
    .row {
      display: grid;
      grid-template-columns: 40px 1fr 76px;
      gap: 8px;
      align-items: end;
    }
    .btn, .send {
      height: 40px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      color: #1f2937;
      cursor: pointer;
    }
    .send {
      border-color: transparent;
      color: #fff;
      background: linear-gradient(135deg, #3b82f6, #2f67d8);
      font-weight: 600;
      font-size: 14px;
    }
    #taskText {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      min-height: 40px;
      max-height: 160px;
      resize: vertical;
      font: inherit;
      outline: none;
      background: #fff;
      color: var(--text);
    }
    .picked { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .picked-preview {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .picked-item {
      position: relative;
      width: 72px;
      height: 72px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
    }
    .picked-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .picked-item button {
      position: absolute;
      top: 2px;
      right: 2px;
      width: 18px;
      height: 18px;
      border: none;
      border-radius: 50%;
      background: rgba(0,0,0,.55);
      color: #fff;
      font-size: 12px;
      line-height: 18px;
      padding: 0;
      cursor: pointer;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--line); }
      .adela-slot-grid { grid-template-columns: 1fr; }
      .adela-flowchart { grid-template-columns: 1fr; }
      .adela-arrow { display: none; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">RD-CLAW</div>
      <button class="new-chat-btn" id="newChatBtn">＋ 新建聊天</button>
      <div id="sessionList" class="session-list"></div>
    </aside>
    <section class="main">
      <main id="chat"></main>
      <div class="composer-wrap">
        <div id="status" class="status">等待输入</div>
        <div class="row">
          <button class="btn" id="pickBtn" title="添加图片，可多次点击累计（最多10张）">＋</button>
          <textarea id="taskText" placeholder="输入问题，按 Enter 发送（Shift+Enter 换行）"></textarea>
          <button class="send" id="runBtn">发送</button>
        </div>
        <input id="imageFile" type="file" hidden multiple accept="image/jpeg,image/png,image/webp,image/bmp,.jpg,.jpeg,.png,.webp,.bmp"/>
        <div id="pickedLabel" class="picked"></div>
        <div id="pickedPreview" class="picked-preview"></div>
      </div>
    </section>
  </div>

<script>
const chatEl = document.getElementById("chat");
const statusEl = document.getElementById("status");
const textEl = document.getElementById("taskText");
const runBtn = document.getElementById("runBtn");
const pickBtn = document.getElementById("pickBtn");
const imageFileInput = document.getElementById("imageFile");
const pickedLabel = document.getElementById("pickedLabel");
const pickedPreviewEl = document.getElementById("pickedPreview");
const sessionListEl = document.getElementById("sessionList");
const newChatBtn = document.getElementById("newChatBtn");

let pendingImageFiles = [];
let currentSessionId = "";
let currentThreadId = "";
let isRunning = false;
const CLIENT_ID_STORAGE_KEY = "rd_claw_browser_id";
const CLIENT_ID = getOrCreateClientId();
const MAX_UPLOAD_IMAGES = 10;
const KNOWN_AGENT_ACTIONS = new Set([
  "rag_answer",
  "re_question",
  "answerer",
  "flux-image-generation",
  "qwen_detection",
  "rexomni_detection",
  "pipeline_eval",
  "adela_cli_eval",
  "migration_advisor_offer",
  "migration_advisor",
  "final_answer",
  "clarify",
]);

function createFreshSessionId() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
  } catch (_) {}
  return "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10);
}

function getOrCreateClientId() {
  try {
    const old = String(window.localStorage.getItem(CLIENT_ID_STORAGE_KEY) || "").trim();
    if (old) return old;
    const created = createFreshSessionId().replace(/^sess_/, "browser_");
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, created);
    return created;
  } catch (_) {
    return createFreshSessionId().replace(/^sess_/, "browser_");
  }
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setStatus(t) { statusEl.textContent = t; }
function scrollBottom() { window.requestAnimationFrame(() => { chatEl.scrollTop = chatEl.scrollHeight; }); }

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html || "";
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  scrollBottom();
  return bubble;
}

/**
 * 下拉框 value：精度评测 = "0"，性能评测 = "1"（与后端 eval_type 整数 0/1 一致）。
 */
function adelaEvalTypeSelectValue(v) {
  if (v === 0 || v === "0") return "0";
  if (v === 1 || v === "1") return "1";
  const raw = String(v == null ? "" : v).trim();
  if (!raw) return "";
  const low = raw.toLowerCase();
  const perfHit =
    ["性能", "性能评测", "速度", "吞吐", "延迟", "推理速度"].some((t) => raw.includes(t)) ||
    /\bfps\b|latency|throughput|\bperformance\b|\bspeed\b/i.test(low);
  const accHit =
    ["精度", "精度评测", "准确率", "准度", "准确度"].some((t) => raw.includes(t)) ||
    /\bmap\b|\baccuracy\b|\bprecision\b/i.test(low);
  if (perfHit && accHit) return "0";
  if (perfHit) return "1";
  if (accHit) return "0";
  return "";
}

async function submitRagFeedback(runId, feedbackType, correctedAnswer, expectedEvidenceIds, statusEl, buttons) {
  const rid = String(runId || "").trim();
  if (!rid) {
    statusEl.textContent = "缺少 run_id，无法提交反馈";
    return;
  }
  const disabledButtons = Array.isArray(buttons) ? buttons : [];
  disabledButtons.forEach((btn) => { btn.disabled = true; });
  statusEl.textContent = "提交中...";
  try {
    const resp = await fetch("/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": CLIENT_ID,
      },
      body: JSON.stringify({
        session_id: currentSessionId,
        run_id: rid,
        feedback_type: feedbackType || "other",
        rating: feedbackType === "helpful" ? 5 : null,
        corrected_answer: String(correctedAnswer || "").trim(),
        expected_evidence_ids: Array.isArray(expectedEvidenceIds) ? expectedEvidenceIds : [],
        comment: String(correctedAnswer || "").trim(),
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
      statusEl.textContent = String((data && data.error) || ("提交失败: HTTP " + resp.status));
      return;
    }
    statusEl.textContent = data.status === "queued" ? "反馈已提交，后台处理中" : "已提交反馈";
  } catch (e) {
    statusEl.textContent = "提交失败: " + String(e);
  } finally {
    disabledButtons.forEach((btn) => { btn.disabled = false; });
  }
}

function adelaEvalTypeLabel(v) {
  const sel = adelaEvalTypeSelectValue(v);
  if (sel === "0") return "精度";
  if (sel === "1") return "性能";
  return String(v == null ? "" : v).trim();
}

function adelaTaskArgsFromPayload(payload) {
  const ts = payload && typeof payload.task_state === "object" && payload.task_state ? payload.task_state : {};
  const args = ts && typeof ts.tool_args === "object" && ts.tool_args ? ts.tool_args : {};
  const rawEval = args.eval_type;
  return {
    model_name: String(args.model_name || ""),
    rawmodel_id: String(args.rawmodel_id || ""),
    platform: String(args.platform || ""),
    eval_type: adelaEvalTypeLabel(rawEval),
    eval_type_select: adelaEvalTypeSelectValue(rawEval),
  };
}

function adelaModelReleaseUrl(rawmodelId) {
  const r = String(rawmodelId == null ? "" : rawmodelId).trim();
  if (!r) return "";
  return "http://scg-adela.sensetime.com/dashboard#/mainpage/project/3/release?rid=" + encodeURIComponent(r);
}

async function submitAdelaSlotForm(form) {
  if (isRunning || !form) return;
  const modelName = String((form.querySelector("[name='model_name']") || {}).value || "").trim();
  const rawmodelId = String((form.querySelector("[name='rawmodel_id']") || {}).value || "").trim();
  const platform = String((form.querySelector("[name='platform']") || {}).value || "").trim();
  const evalType = String((form.querySelector("[name='eval_type']") || {}).value || "").trim();
  const errEl = form.querySelector(".adela-form-error");
  if (errEl) errEl.textContent = "";
  if (!rawmodelId && !modelName) {
    if (errEl) errEl.textContent = "请填写模型名称或 rawmodel_id";
    return;
  }
  if (!platform) {
    if (errEl) errEl.textContent = "请填写目标平台";
    return;
  }
  if (evalType !== "0" && evalType !== "1") {
    if (errEl) errEl.textContent = "请选择评测类型";
    return;
  }
  // 精度评测 -> eval_type 0；性能评测 -> eval_type 1（JSON 中显式整数，避免 0 被序列化链路误判为「空」）
  const eval_type = evalType === "1" ? 1 : 0;
  const payload = {
    _structured_type: "adela_slot_form",
    model_name: modelName,
    rawmodel_id: rawmodelId,
    platform,
    eval_type,
  };
  textEl.value = JSON.stringify(payload);
  await sendMessage({ hiddenStructured: true, displayText: "提交 Adela 参数" });
}

async function submitMigrationAdvisorChoice(choice, label) {
  if (isRunning) return;
  const payload = {
    _structured_type: "migration_advisor_choice",
    choice: String(choice || "").trim(),
  };
  textEl.value = JSON.stringify(payload);
  await sendMessage({ hiddenStructured: true, displayText: label || "选择迁移顾问操作" });
}

function createStreamingTextRenderer() {
  let rendered = "";
  let queued = "";
  let running = false;
  const drainedCallbacks = [];
  function notifyDrained() {
    while (drainedCallbacks.length) {
      const cb = drainedCallbacks.shift();
      try { cb(); } catch (_) {}
    }
  }
  function drain(renderFn) {
    if (running) return;
    running = true;
    const loop = () => {
      if (!queued) {
        running = false;
        notifyDrained();
        return;
      }
      const first = queued.charAt(0);
      const take = /[\\n，。！？；,.!?;:]/.test(first) ? 1 : (queued.length > 120 ? 4 : 2);
      rendered += queued.slice(0, take);
      queued = queued.slice(take);
      renderFn(rendered);
      scrollBottom();
      window.setTimeout(loop, /[\\n，。！？；,.!?;:]/.test(first) ? 42 : 18);
    };
    loop();
  }
  return {
    push(text, renderFn) {
      const t = String(text || "");
      if (!t) return;
      queued += t;
      drain(renderFn);
    },
    onDrained(cb) {
      if (typeof cb !== "function") return;
      if (!running && !queued) {
        try { cb(); } catch (_) {}
        return;
      }
      drainedCallbacks.push(cb);
    }
  };
}

function createAssistantStepForFlow(parent, flow, decision, options) {
  decision = decision || {};
  options = options || {};
  const streamingEnabled = options.streaming !== false;
  const sections = {};
  const streamers = {};
  const syncBuffers = {};
  const MSG_RUNNING = "<div class='pending-msg'>正在执行，请稍候...</div>";
  const MSG_QUEUED = "<div class='pending-msg'>即将执行，请稍候...</div>";
  const noop = () => {};

  function section(key, title) {
    if (sections[key]) return sections[key];
    const d = document.createElement("div");
    d.className = "section";
    d.innerHTML = '<div class="section-title">' + esc(title) + "</div>";
    parent.appendChild(d);
    sections[key] = d;
    scrollBottom();
    return d;
  }
  function replace(d, html) {
    const titleEl = d.querySelector(".section-title");
    const title = titleEl ? titleEl.outerHTML : "";
    d.innerHTML = title + html;
  }
  function stream(key, title, text) {
    const d = section(key, title);
    if (!streamingEnabled) {
      syncBuffers[key] = (syncBuffers[key] || "") + String(text || "");
      replace(d, "<div style='line-height:1.6;white-space:pre-wrap'>" + esc(syncBuffers[key]).replace(/\\n/g, "<br/>") + "</div>");
      return;
    }
    if (!streamers[key]) streamers[key] = createStreamingTextRenderer();
    streamers[key].push(text, (cur) => {
      replace(d, "<div style='line-height:1.6;white-space:pre-wrap'>" + esc(cur).replace(/\\n/g, "<br/>") + "</div>");
    });
  }
  function err(message) {
    const d = section("error", "错误");
    replace(d, "<div class='pending-msg'>" + esc(message || "未知错误") + "</div>");
  }

  let routeD = null;
  function onDecision(nextDecision) {
    decision = nextDecision || decision || {};
    const action = String(decision.action || "");
    let reason = String(decision.reason || "").trim();
    if (isIgnorableDecisionReason(reason)) {
      reason = "";
    }
    if (!action && !reason) return;
    if (!routeD) routeD = section("route", "推理");
    let html = "";
    if (action) html += '<div class="kv-row"><b>Action</b><code style="margin-left:6px;background:var(--brand-soft);padding:2px 6px;border-radius:6px;border:1px solid var(--line)">' + esc(action) + "</code></div>";
    if (reason) html += '<details class="thinking-toggle"><summary>推理说明（点击展开）</summary><div class="thinking-content">' + esc(reason).replace(/\\n/g, "<br/>") + "</div></details>";
    replace(routeD, html);
  }
  onDecision(decision);

  if (flow === "direct_answer" || flow === "self_intro") {
    const ansD = section("final", "回答");
    replace(ansD, MSG_RUNNING);
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop, solution: noop, reQuestion: noop, adela: noop,
      finalAnswer(t) { stream("final", "回答", t); },
      error: err,
    };
  }
  if (flow === "rag") {
    const replyD = section("reply", "知识库回答");
    replace(replyD, MSG_RUNNING);
    let hasReplyStarted = false;
    let hasReplyFinished = false;
    let hasStreamedReply = false;
    let pendingRefs = null;
    let ragRunId = "";
    let ragEvidenceIds = [];
    let feedbackRendered = false;
    function renderRefs(refs) {
      const d = section("refs", "参考链接");
      const valid = (Array.isArray(refs) ? refs : []).filter(x => x && x.url);
      if (!valid.length) return;
      let html = "";
      valid.forEach((it, idx) => {
        const name = String(it.doc_name || ("参考 " + (idx + 1)));
        const u = String(it.url || "");
        html += '<div style="margin:6px 0"><a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(name) + "</a></div>";
      });
      replace(d, html);
    }
    function renderFeedback() {
      if (feedbackRendered || !ragRunId) return;
      feedbackRendered = true;
      const d = section("feedback", "反馈");
      replace(
        d,
        '<div class="rag-feedback">' +
          '<div class="rag-feedback-row">' +
            '<button type="button" data-feedback="helpful">有帮助</button>' +
            '<button type="button" data-feedback="harmful">不准确</button>' +
          '</div>' +
          '<textarea placeholder="输入修正答案或补充说明"></textarea>' +
          '<div class="rag-feedback-row">' +
            '<button type="button" class="primary" data-feedback="correction">提交修正</button>' +
            '<span class="rag-feedback-status"></span>' +
          '</div>' +
        '</div>'
      );
      const textarea = d.querySelector("textarea");
      const status = d.querySelector(".rag-feedback-status");
      const buttons = Array.from(d.querySelectorAll("button"));
      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          const type = String(btn.getAttribute("data-feedback") || "other");
          submitRagFeedback(ragRunId, type, textarea ? textarea.value : "", ragEvidenceIds, status, buttons);
        });
      });
    }
    return {
      onDecision,
      direct(t) {
        if (!hasReplyStarted) {
          replace(replyD, MSG_RUNNING);
          hasReplyStarted = true;
        }
        hasStreamedReply = true;
        stream("reply", "知识库回答", t);
        if (!streamingEnabled) {
          hasReplyFinished = true;
          if (pendingRefs) {
            renderRefs(pendingRefs);
            pendingRefs = null;
          }
          return;
        }
        if (streamers.reply && !hasReplyFinished) {
          streamers.reply.onDrained(() => {
            hasReplyFinished = true;
            if (pendingRefs) {
              renderRefs(pendingRefs);
              pendingRefs = null;
            }
          });
        }
      },
      stepDone: noop,
      ragRefs(refs) {
        if (!hasReplyFinished) {
          pendingRefs = refs;
          return;
        }
        renderRefs(refs);
        renderFeedback();
      },
      ragRun(payload) {
        ragRunId = String((payload && payload.run_id) || "").trim();
        ragEvidenceIds = Array.isArray(payload && payload.evidence_ids) ? payload.evidence_ids : [];
        if (hasReplyFinished) renderFeedback();
      },
      intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop, solution: noop, reQuestion: noop, adela: noop,
      finalAnswer(t) {
        if (hasStreamedReply) {
          hasReplyFinished = true;
          if (pendingRefs) {
            renderRefs(pendingRefs);
            pendingRefs = null;
          }
          renderFeedback();
          return;
        }
        if (!hasReplyStarted) {
          replace(replyD, MSG_RUNNING);
          hasReplyStarted = true;
        }
        stream("reply", "知识库回答", t);
        if (!streamingEnabled) {
          hasReplyFinished = true;
          if (pendingRefs) {
            renderRefs(pendingRefs);
            pendingRefs = null;
          }
          renderFeedback();
          return;
        }
        if (streamers.reply && !hasReplyFinished) {
          streamers.reply.onDrained(() => {
            hasReplyFinished = true;
            if (pendingRefs) {
              renderRefs(pendingRefs);
              pendingRefs = null;
            }
            renderFeedback();
          });
        }
      },
      error: err
    };
  }
  if (flow === "re_question") {
    const d = section("rewrite", "检索改写结果");
    replace(d, MSG_RUNNING);
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop, adela: noop,
      reQuestion(payload) {
        const src = String(payload && payload.source_query || "");
        const rw = String(payload && payload.rewritten_query || "");
        const round = payload && payload.retrieval_round != null ? String(payload.retrieval_round) : "";
        let html = "";
        if (round) html += "<div class='kv-row'><b>检索轮次</b>第 " + esc(round) + " 轮</div>";
        if (src) html += "<div class='kv-row'><b>原查询</b>" + esc(src) + "</div>";
        if (rw) html += "<div class='kv-row'><b>改写后查询</b>" + esc(rw) + "</div>";
        replace(d, html || "<div class='pending-msg'>改写结果为空</div>");
      },
      finalAnswer: noop, error: err
    };
  }
  if (flow === "clarify") {
    const d = section("clarify", "需要确认");
    replace(d, MSG_RUNNING);
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop,
      reQuestion: noop, adela: noop,
      clarification(text, payload) {
        const q = String(text || (payload && payload.text) || "");
        const missing = Array.isArray(payload && payload.missing_slots) ? payload.missing_slots : [];
        const taskState = payload && typeof payload.task_state === "object" && payload.task_state ? payload.task_state : {};
        const isAdela = String(taskState.candidate_tool || "") === "adela_cli_eval";
        let html = q ? ("<div>" + esc(q).replace(/\\n/g, "<br/>") + "</div>") : "";
        if (missing.length) {
          html += "<div class='kv-row'><b>缺少条件</b>" + esc(missing.join(" / ")) + "</div>";
        }
        if (isAdela) {
          const args = adelaTaskArgsFromPayload(payload);
          const evalValue = args.eval_type_select || "";
          html +=
            '<form class="adela-slot-form">' +
              '<div class="adela-slot-grid">' +
                '<div class="adela-field"><label>模型名称</label><input name="model_name" autocomplete="off" value="' + esc(args.model_name) + '" placeholder="例如：人脸检测模型"></div>' +
                '<div class="adela-field"><label>rawmodel_id</label><input name="rawmodel_id" inputmode="numeric" autocomplete="off" value="' + esc(args.rawmodel_id) + '" placeholder="已知 ID 时填写"></div>' +
                '<div class="adela-field"><label>目标平台</label><input name="platform" autocomplete="off" value="' + esc(args.platform) + '" placeholder="cuda11.0-trt7.1-int8-T4"></div>' +
                '<div class="adela-field"><label>评测类型</label><select name="eval_type" autocomplete="off">' +
                  '<option value="">请选择</option>' +
                  '<option value="0"' + (evalValue === "0" ? " selected" : "") + '>精度评测（0）</option>' +
                  '<option value="1"' + (evalValue === "1" ? " selected" : "") + '>性能评测（1）</option>' +
                '</select></div>' +
              '</div>' +
              '<div class="adela-form-actions"><button class="adela-submit" type="submit">执行 Adela 评测</button><span class="adela-form-error"></span></div>' +
            '</form>';
        }
        replace(d, html || "<div class='pending-msg'>等待澄清问题...</div>");
        const form = d.querySelector(".adela-slot-form");
        if (form) {
          form.addEventListener("submit", (e) => {
            e.preventDefault();
            submitAdelaSlotForm(form);
          });
        }
      },
      finalAnswer: noop, error: err
    };
  }
  if (flow === "migration_advisor_offer") {
    const d = section("advisor_offer", "迁移顾问");
    replace(d, MSG_RUNNING);
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop,
      reQuestion: noop, adela: noop,
      migrationOffer(payload) {
        const text = String((payload && payload.text) || "知识库没有找到可直接回答的信息。是否生成迁移顾问报告？");
        const opts = Array.isArray(payload && payload.options) ? payload.options : [
          { id: "start", label: "生成迁移顾问报告" },
          { id: "fallback_answer", label: "直接回答" },
          { id: "cancel", label: "取消" },
        ];
        let html = "<div>" + esc(text).replace(/\\n/g, "<br/>") + "</div><div class='advisor-actions'>";
        opts.forEach((opt) => {
          const id = String(opt && opt.id || "");
          const label = String(opt && opt.label || id);
          const cls = id === "start" ? " class='primary'" : "";
          html += "<button type='button'" + cls + " data-choice='" + esc(id) + "'>" + esc(label) + "</button>";
        });
        html += "</div>";
        replace(d, html);
        Array.from(d.querySelectorAll("button[data-choice]")).forEach((btn) => {
          btn.addEventListener("click", () => {
            Array.from(d.querySelectorAll("button[data-choice]")).forEach((b) => { b.disabled = true; });
            const choice = String(btn.getAttribute("data-choice") || "");
            submitMigrationAdvisorChoice(choice, btn.textContent || "选择迁移顾问操作");
          });
        });
      },
      finalAnswer: noop, error: err
    };
  }
  if (flow === "migration_advisor") {
    const planD = section("advisor_plan", "迁移顾问规划");
    replace(planD, MSG_RUNNING);
    const retrieveD = section("advisor_retrieve", "分字段检索");
    replace(retrieveD, MSG_QUEUED);
    const rexD = section("advisor_rex", "Rex-Omni模型标注");
    replace(rexD, MSG_QUEUED);
    const reportD = section("advisor_report", "迁移评估报告");
    replace(reportD, MSG_QUEUED);
    let migrationReportRendered = false;
    let migrationRexHandled = false;
    const fields = {};
    function renderFields() {
      const vals = Object.keys(fields).map((k) => fields[k]);
      if (!vals.length) {
        replace(retrieveD, MSG_RUNNING);
        return;
      }
      let html = "";
      vals.forEach((it) => {
        const queries = Array.isArray(it.queries) && it.queries.length ? it.queries : (it.query ? [it.query] : []);
        html += "<div class='kv-row'><b>" + esc(it.field) + "</b>" + esc(it.status || "") +
          (it.count != null ? (" / " + esc(String(it.count)) + " 条") : "") +
          (it.coverage ? (" / " + esc(it.coverage)) : "") +
          "<div class='pending-msg'>" + esc(queries.join(" / ")) + "</div></div>";
      });
      replace(retrieveD, html);
    }
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop,
      reQuestion: noop, adela: noop, migrationOffer: noop,
      migrationPlan(payload) {
        const plan = payload && payload.plan ? payload.plan : payload;
        const req = plan && plan.abstract_requirement ? plan.abstract_requirement : {};
        const fieldsArr = Array.isArray(plan && plan.retrieve_fields) ? plan.retrieve_fields : [];
        let html = "";
        html += "<div class='kv-row'><b>任务类型</b>" + esc(req.task_type || "") + "</div>";
        html += "<div class='kv-row'><b>对象/属性</b>" + esc([req.object, req.attribute].filter(Boolean).join(" / ")) + "</div>";
        if (fieldsArr.length) {
          html += "<div class='kv-row'><b>检索字段</b></div>";
          fieldsArr.forEach((f) => {
            const queries = Array.isArray(f.queries) && f.queries.length ? f.queries : (f.query ? [f.query] : []);
            html += "<div class='pending-msg'>" + esc(f.field || "") + "： " + esc(queries.join(" / ")) + "</div>";
          });
        }
        replace(planD, html || "<div class='pending-msg'>已生成检索规划</div>");
      },
      migrationRetrieve(payload) {
        const field = String(payload && payload.field || "");
        if (!field) return;
        fields[field] = {
          field,
          query: String(payload.query || ""),
          queries: Array.isArray(payload.queries) ? payload.queries : [],
          status: String(payload.status || ""),
          coverage: String(payload.coverage || ""),
          count: payload.count,
        };
        renderFields();
      },
      migrationRex(payload) {
        migrationRexHandled = true;
        const status = String((payload && payload.status) || "");
        if (status === "running") {
          replace(rexD, MSG_RUNNING);
          return;
        }
        if (status === "error") {
          replace(
            rexD,
            "<div class='pending-msg'>标注失败：" + esc(String((payload && payload.error) || "未知错误")) + "</div>"
          );
          return;
        }
        if (status !== "done") return;
        let html = "";
        const label = String((payload && payload.label) || "");
        const numBoxes = payload && payload.num_boxes != null ? String(payload.num_boxes) : "";
        const det = payload && payload.detection_targets ? payload.detection_targets : {};
        const classes = Array.isArray(det.classes) ? det.classes : [];
        if (label) html += "<div class='kv-row'><b>检测任务</b>" + esc(label) + "</div>";
        if (classes.length) {
          html += "<div class='kv-row'><b>待检类别（" + esc(String(classes.length)) + "）</b>" +
            esc(classes.map((c) => String((c && c.label) || "")).filter(Boolean).join("、")) + "</div>";
        }
        if (numBoxes !== "") html += "<div class='kv-row'><b>预测框总数</b>" + esc(numBoxes) + "</div>";
        const perImage = Array.isArray(payload && payload.per_image) ? payload.per_image : [];
        if (perImage.length) {
          html += "<div class='kv-row'><b>各图类别命中</b></div>";
          perImage.forEach((row) => {
            const hits = Array.isArray(row && row.hit_labels) ? row.hit_labels.join("、") : "无";
            const misses = Array.isArray(row && row.miss_labels) ? row.miss_labels.join("、") : "无";
            html += "<div class='pending-msg'>" + esc(String((row && row.file_name) || "图片")) +
              "：命中 " + esc(hits) + "；未命中 " + esc(misses) + "</div>";
          });
        }
        const urls = Array.isArray(payload && payload.annotated_urls) ? payload.annotated_urls : [];
        if (urls.length) {
          html += '<div class="thumb-row">' + urls.map((u) => '<img src="' + esc(u) + '" loading="lazy" />').join("") + "</div>";
        }
        const acc = payload && payload.accuracy_estimate ? payload.accuracy_estimate : {};
        const accuracy = String(acc.accuracy || "").trim();
        const reason = String(acc.reason || "").trim();
        if (accuracy) html += "<div class='kv-row'><b>准确率预估</b>" + esc(accuracy) + "</div>";
        if (reason) html += "<div class='pending-msg'>" + esc(reason) + "</div>";
        replace(rexD, html || "<div class='pending-msg'>Rex-Omni 标注已完成。</div>");
      },
      migrationReport(payload) {
        if (!migrationRexHandled) {
          replace(rexD, "<div class='pending-msg'>未上传参考图，已跳过 Rex-Omni 标注。</div>");
        }
        const md = String((payload && payload.markdown) || "");
        if (md) {
          migrationReportRendered = true;
          replace(reportD, "<div style='line-height:1.6;white-space:pre-wrap'>" + esc(md).replace(/\\n/g, "<br/>") + "</div>");
        } else {
          replace(reportD, "<div class='pending-msg'>报告已生成。</div>");
        }
      },
      finalAnswer(t) {
        const md = String(t || "").trim();
        if (!md) return;
        if (migrationReportRendered) return;
        if (!migrationRexHandled) {
          replace(rexD, "<div class='pending-msg'>未上传参考图，已跳过 Rex-Omni 标注。</div>");
        }
        replace(reportD, "<div style='line-height:1.6;white-space:pre-wrap'>" + esc(md).replace(/\\n/g, "<br/>") + "</div>");
      },
      error: err
    };
  }
  if (flow === "flux") {
    const d = section("gen", "图片生成（Flux）");
    replace(d, MSG_RUNNING);
    const urls = [];
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop,
      gen(url) {
        if (!url) return;
        urls.push(url);
        replace(d, '<div class="thumb-row">' + urls.map(u => '<img src="' + esc(u) + '" loading="lazy" />').join("") + "</div>");
      },
      annotated: noop, eval: noop, detectionInfo: noop, reQuestion: noop, adela: noop, finalAnswer: noop, error: err
    };
  }
  if (flow === "qwen_detect" || flow === "rexomni_detect") {
    const det = section("det", "检测设置");
    replace(det, MSG_RUNNING);
    const annoTitle = flow === "rexomni_detect" ? "标注结果（Rexomni）" : "标注结果（Qwen）";
    return {
      onDecision, direct: noop, stepDone: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop,
      annotated(urls) {
        const d = section("anno", annoTitle);
        replace(d, '<div class="thumb-row">' + (urls || []).map(u => '<img src="' + esc(u) + '" loading="lazy" />').join("") + "</div>");
      },
      eval: noop,
      detectionInfo(label) {
        replace(det, "<div class='kv-row'><b>开集标签</b>" + esc(label || "") + "</div>");
        const d = section("anno", annoTitle);
        replace(d, MSG_RUNNING);
      },
      reQuestion: noop, adela: noop, finalAnswer: noop, error: err
    };
  }
  if (flow === "adela_eval") {
    const retrievalD = section("adela_retrieval", "模型检索");
    replace(retrievalD, '<div class="pending-msg">正在解析模型与平台参数…</div>');
    const d = section("adela", "ADELA平台");
    replace(d, MSG_RUNNING);
    const state = {
      retrievalStatus: "pending",
      retrievalRid: "",
      retrievalModelName: "",
      retrievalPlatform: "",
      retrievalUrl: "",
      retrievalNote: "",
      deploymentId: "",
      benchmarkId: "",
      datasetId: "",
      finalResultText: "",
      errorMessage: "",
      step1DeployQuery: "pending",
      step1BenchQuery: "pending",
      step1QuantQuery: "pending",
      step1EvalDsQuery: "pending",
      step1DeployNote: "",
      step1BenchNote: "",
      step1QuantNote: "",
      step1EvalNote: "",
      step2SubmitDeploy: "pending",
      step2SubmitNote: "",
      step2Deploying: "pending",
      step2DeployComplete: "pending",
      step3SubmitEval: "pending",
      step3SubmitNote: "",
      step3Evaluating: "pending",
      step3EvalComplete: "pending",
      sawSubmitDeploy: false,
      /** 目标平台已有部署但未命中历史评测：复用部署直接评测，Step2 保留展示为不执行（skipped） */
      skipStep2Execution: false,
      showQuantDatasetStep: false,
    };
    function renderRetrieval() {
      const rid = String(state.retrievalRid || "").trim();
      const modelName = String(state.retrievalModelName || "").trim();
      const plat = String(state.retrievalPlatform || "").trim();
      const url = String(state.retrievalUrl || "").trim() || (rid ? adelaModelReleaseUrl(rid) : "");
      const note = String(state.retrievalNote || "").trim();
      if (state.retrievalStatus === "pending" && !rid) {
        replace(retrievalD, '<div class="pending-msg">正在解析模型与平台参数…</div>');
        return;
      }
      let body = "";
      if (rid) {
        body += '<div class="kv-row"><b>原始模型ID</b>' + esc(rid) + "</div>";
      } else {
        body += '<div class="kv-row"><b>原始模型ID</b><span style="color:#6b7280;">（待定）</span></div>';
      }
      if (modelName) {
        body += '<div class="kv-row"><b>模型名称</b>' + esc(modelName) + "</div>";
      } else {
        body += '<div class="kv-row"><b>模型名称</b><span style="color:#6b7280;">（待定）</span></div>';
      }
      if (url) {
        body +=
          '<div class="kv-row"><b>模型链接</b><a href="' +
          esc(url) +
          '" target="_blank" rel="noopener noreferrer">' +
          esc(url) +
          "</a></div>";
      } else if (rid) {
        body += '<div class="kv-row"><b>模型链接</b><span style="color:#6b7280;">（待定）</span></div>';
      }
      if (plat) {
        body += '<div class="kv-row"><b>部署平台</b>' + esc(plat) + "</div>";
      } else {
        body += '<div class="kv-row"><b>部署平台</b><span style="color:#6b7280;">（待定）</span></div>';
      }
      if (note) body += '<div class="kv-row" style="margin-top:6px;color:#6b7280;font-size:12px;">' + esc(note) + "</div>";
      const wrap =
        '<div style="border:1px solid #86efac;border-radius:10px;background:#f0fdf4;padding:10px 12px;">' +
        body +
        "</div>";
      replace(retrievalD, wrap);
    }
    function applyModelRetrievalFromPayload(p) {
      const pp = p || {};
      const rid =
        pp.rawmodel_id != null && String(pp.rawmodel_id).trim() ? String(pp.rawmodel_id).trim() : "";
      const plat = String(pp.platform || "").trim();
      if (!rid || !plat) return false;
      state.retrievalRid = rid;
      state.retrievalPlatform = plat;
      const resolvedName = String(pp.matched_name || pp.model_name || "").trim();
      if (resolvedName) state.retrievalModelName = resolvedName;
      const u = String(pp.model_url || "").trim();
      state.retrievalUrl = u || adelaModelReleaseUrl(rid);
      const msg = String(pp.message || "").trim();
      if (msg) state.retrievalNote = msg;
      state.retrievalStatus = "success";
      renderRetrieval();
      return true;
    }
    function mark(status) {
      if (status === "success") return "✅";
      if (status === "running") return "⏳";
      if (status === "failed") return "❌";
      if (status === "skipped") return "—";
      return "○";
    }
    function nodeClass(status) {
      if (status === "success") return "adela-node adela-node--success";
      if (status === "running") return "adela-node adela-node--running";
      if (status === "failed") return "adela-node adela-node--failed";
      if (status === "skipped") return "adela-node adela-node--skipped";
      return "adela-node adela-node--pending";
    }
    function row(title, status, extra, withDots) {
      const dots = withDots && status === "running"
        ? '<span class="adela-dots"><span>.</span><span>.</span><span>.</span></span>'
        : "";
      const icon = mark(status);
      let line = '<div class="' + nodeClass(status) + '">';
      line += '<div class="adela-node__title"><span>' + icon + "</span><span>" + esc(title) + dots + "</span></div>";
      if (extra) line += '<div class="adela-node__sub">' + esc(extra) + "</div>";
      line += "</div>";
      return line;
    }
    function setResultFromPreview(preview) {
      const raw = String(preview || "").trim();
      if (!raw) return;
      let parsed = null;
      try { parsed = JSON.parse(raw); } catch (_) {}
      if (parsed && typeof parsed === "object" && Object.prototype.hasOwnProperty.call(parsed, "result")) {
        const r = parsed.result;
        if (r == null) return;
        if (typeof r === "string") {
          state.finalResultText = r;
          return;
        }
        try { state.finalResultText = JSON.stringify(r, null, 2); } catch (_) {}
      }
    }
    function markFollowingSkippedAfterDataStop() {
      if (state.step2SubmitDeploy === "pending") state.step2SubmitDeploy = "skipped";
      if (state.step2Deploying === "pending") state.step2Deploying = "skipped";
      if (state.step2DeployComplete === "pending") state.step2DeployComplete = "skipped";
      if (state.step3SubmitEval === "pending") state.step3SubmitEval = "skipped";
      if (state.step3Evaluating === "pending") state.step3Evaluating = "skipped";
      if (state.step3EvalComplete === "pending") state.step3EvalComplete = "skipped";
    }
    function render() {
      let stage1 = "";
      stage1 += '<div class="adela-stage-title">Step1 · 数据查询</div>';
      /* Step1 仅展示各查询阶段固化下来的文案，不回填后续 Step2/3 产生的 deploymentId/benchmarkId */
      const deploySub = String(state.step1DeployNote || "").trim();
      stage1 += row("部署模型查询", state.step1DeployQuery, deploySub, false);
      const benchSub = String(state.step1BenchNote || "").trim();
      stage1 += row("评测结果查询", state.step1BenchQuery, benchSub, false);
      if (state.showQuantDatasetStep) {
        const quantSub = String(state.step1QuantNote || "").trim();
        stage1 += row("量化数据集查询", state.step1QuantQuery, quantSub, false);
      }
      const evalDsSub = String(state.step1EvalNote || "").trim();
      stage1 += row("评测数据集查询", state.step1EvalDsQuery, evalDsSub, false);

      let stage2 = "";
      stage2 += '<div class="adela-stage-title">Step2 · 模型部署</div>';
      const step2SubmitSub = String(state.step2SubmitNote || "").trim();
      stage2 += row("发起模型部署", state.step2SubmitDeploy, step2SubmitSub, false);
      stage2 += row("模型部署中", state.step2Deploying, "", state.step2Deploying === "running");
      stage2 += row("完成模型部署", state.step2DeployComplete, "", false);

      let stage3 = "";
      stage3 += '<div class="adela-stage-title">Step3 · 模型评测</div>';
      const step3SubmitSub = String(state.step3SubmitNote || "").trim();
      stage3 += row("发起模型评测", state.step3SubmitEval, step3SubmitSub, false);
      stage3 += row("模型评测中", state.step3Evaluating, "", state.step3Evaluating === "running");
      stage3 += row("完成模型评测", state.step3EvalComplete, "", false);

      let html = '<div class="adela-flowchart">'
        + '<div class="adela-stage">' + stage1 + "</div>"
        + '<div class="adela-arrow">→</div>'
        + '<div class="adela-stage">' + stage2 + "</div>"
        + '<div class="adela-arrow">→</div>'
        + '<div class="adela-stage">' + stage3 + "</div>"
        + "</div>";
      if (state.errorMessage) {
        html += '<div class="kv-row" style="margin-top:8px;"><b>异常</b> <span style="color:#dc2626;">' + esc(state.errorMessage) + "</span></div>";
      }
      replace(d, html || MSG_RUNNING);
    }
    renderRetrieval();
    render();
    return {
      onDecision, direct: noop, ragRefs: noop, ragRun: noop, intent: noop, gen: noop, annotated: noop, eval: noop, detectionInfo: noop, reQuestion: noop,
      adela(payload) {
        const p = payload || {};
        const evt = String(p.event || "");
        if (evt === "model_retrieval_result" || evt === "adela_model_resolved") {
          applyModelRetrievalFromPayload(p);
        } else if (state.retrievalStatus === "pending" && evt === "deployment_list_result") {
          applyModelRetrievalFromPayload(p);
        } else if (
          state.retrievalStatus === "pending" &&
          (evt === "adela_dual_eval_notice" || evt === "adela_dual_eval_phase") &&
          p.rawmodel_id != null &&
          String(p.rawmodel_id).trim() &&
          String(p.platform || "").trim()
        ) {
          applyModelRetrievalFromPayload(p);
        }

        if (evt === "benchmark_probe_result") {
          state.step1BenchQuery = "success";
          const bid =
            p.benchmark_id != null &&
            p.benchmark_id !== "" &&
            String(p.benchmark_id).trim() !== "" &&
            String(p.benchmark_id).toLowerCase() !== "null"
              ? String(p.benchmark_id).trim()
              : "";
          state.benchmarkId = bid;
          state.step1BenchNote = bid ? "benchmark_id " + bid : "benchmark_id NULL";
          const depProbe = p.deployment_id != null && String(p.deployment_id).trim() ? String(p.deployment_id) : "";
          const matchedProbe =
            p.matched === true || p.matched === 1 || String(p.matched || "").toLowerCase() === "true";
          state.skipStep2Execution = !!depProbe && !matchedProbe;
          if (state.skipStep2Execution) {
            state.step2SubmitDeploy = "skipped";
            state.step2SubmitNote = "";
            state.step2Deploying = "skipped";
            state.step2DeployComplete = "skipped";
          }
          if (depProbe) {
            state.showQuantDatasetStep = false;
            state.deploymentId = depProbe;
            if (!state.step1DeployNote) state.step1DeployNote = "部署ID " + depProbe;
          }
        } else {
          const rawBid = p.benchmark_id != null && p.benchmark_id !== "" ? p.benchmark_id : p.benchmarkId;
          if (rawBid != null && String(rawBid).trim() !== "" && String(rawBid).toLowerCase() !== "null") {
            state.benchmarkId = String(rawBid).trim();
          }
        }

        const rawDs = p.dataset_id != null && p.dataset_id !== "" ? p.dataset_id : p.datasetId;
        if (rawDs != null && String(rawDs).trim() !== "") state.datasetId = String(rawDs);

        if (evt === "deployment_list_result") {
          state.step1DeployQuery = "success";
          const tid = p.target_deployment_id != null && String(p.target_deployment_id).trim()
            ? String(p.target_deployment_id)
            : "";
          if (tid) {
            state.deploymentId = tid;
            state.step1DeployNote = "部署ID " + tid;
          } else {
            state.deploymentId = "";
            state.step1DeployNote = "部署ID NULL";
          }
          if (p.quant_dataset_step_needed === true || p.quant_dataset_step_needed === 1) {
            state.showQuantDatasetStep = true;
          } else {
            state.showQuantDatasetStep = false;
          }
        } else if ((evt === "quant_dataset_result" || evt === "quant_dataset_ready") && state.showQuantDatasetStep) {
          state.step1QuantQuery = "success";
          if (p.dataset_id != null && String(p.dataset_id).trim()) {
            state.datasetId = String(p.dataset_id);
            state.step1QuantNote = "dataset_id " + state.datasetId;
          }
        } else if (evt === "eval_dataset_result" || evt === "eval_dataset_ready") {
          if (state.showQuantDatasetStep && state.step1QuantQuery === "pending") state.step1QuantQuery = "skipped";
          state.step1EvalDsQuery = "success";
          if (p.dataset_id != null && String(p.dataset_id).trim()) {
            state.datasetId = String(p.dataset_id);
            state.step1EvalNote = "dataset_id " + state.datasetId;
          }
        } else if (evt === "adela_api_result") {
          /* 每步 adela 命令的细粒度回流，当前 UI 不展示 */
        } else if (evt === "adela_existing_result") {
          state.step1DeployQuery = "success";
          state.step1BenchQuery = "success";
          if (state.showQuantDatasetStep) state.step1QuantQuery = "skipped";
          const depH = p.deployment_id != null && String(p.deployment_id).trim() ? String(p.deployment_id) : "";
          const bidH = state.benchmarkId || "";
          const dsH = p.dataset_id != null && String(p.dataset_id).trim() ? String(p.dataset_id) : "";
          if (depH) {
            state.deploymentId = depH;
            state.step1DeployNote = "部署ID " + depH;
          }
          state.step1BenchNote = bidH ? "benchmark_id " + bidH : "benchmark_id NULL";
          if (dsH) {
            state.datasetId = dsH;
            state.step1EvalDsQuery = "success";
            state.step1EvalNote = "dataset_id " + dsH;
          } else {
            state.step1EvalDsQuery = "skipped";
            state.step1EvalNote = "";
          }
          state.step2SubmitDeploy = "skipped";
          state.step2SubmitNote = "";
          state.step2Deploying = "skipped";
          state.step2DeployComplete = "skipped";
          state.step3SubmitEval = "skipped";
          state.step3SubmitNote = "";
          state.step3Evaluating = "skipped";
          state.step3EvalComplete = "skipped";
          setResultFromPreview(p.result_preview);
        } else if (evt === "quant_dataset_missing" && state.showQuantDatasetStep) {
          state.finalResultText = "";
          state.step1QuantQuery = "failed";
          state.step1QuantNote = String(p.message || "缺少量化数据集");
          state.errorMessage = String(p.message || "缺少量化数据集");
          if (state.step1BenchQuery === "pending") state.step1BenchQuery = "skipped";
          if (state.step1EvalDsQuery === "pending") state.step1EvalDsQuery = "skipped";
          markFollowingSkippedAfterDataStop();
        } else if (evt === "eval_dataset_missing") {
          state.finalResultText = "";
          state.step1EvalDsQuery = "failed";
          state.step1EvalNote = String(p.message || "缺少评测数据集");
          state.errorMessage = String(p.message || "缺少评测数据集");
          if (state.step1BenchQuery === "pending") state.step1BenchQuery = "skipped";
          if (state.showQuantDatasetStep && state.step1QuantQuery === "pending") state.step1QuantQuery = "skipped";
          markFollowingSkippedAfterDataStop();
        } else if (evt === "submit_model_deployment") {
          state.skipStep2Execution = false;
          state.sawSubmitDeploy = true;
          if (p.deployment_id != null && String(p.deployment_id).trim()) {
            state.deploymentId = String(p.deployment_id);
          }
          if (state.deploymentId) state.step2SubmitNote = "部署ID " + state.deploymentId;
          state.step2SubmitDeploy = "success";
          state.step2Deploying = "running";
          state.step2DeployComplete = "pending";
        } else if (evt === "model_deployment_result") {
          if (p.deployment_id != null && String(p.deployment_id).trim()) {
            state.deploymentId = String(p.deployment_id);
          }
          const st = String(p.status || "").toUpperCase();
          if (st === "SUCCESS") {
            state.step2Deploying = "success";
            state.step2DeployComplete = "success";
          } else {
            state.step2Deploying = "failed";
            state.step2DeployComplete = "failed";
            state.errorMessage = String(p.message || p.result_preview || "模型部署失败").trim() || "模型部署失败";
            markFollowingSkippedAfterDataStop();
          }
        } else if (evt === "submit_model_evaluation") {
          if (p.deployment_id != null && String(p.deployment_id).trim()) {
            state.deploymentId = String(p.deployment_id);
          }
          const bidSubmit =
            p.benchmark_id != null && String(p.benchmark_id).trim() ? String(p.benchmark_id) : "";
          if (bidSubmit) {
            state.benchmarkId = bidSubmit;
            state.step3SubmitNote = "评测ID " + bidSubmit;
          }
          if (!state.sawSubmitDeploy && !state.skipStep2Execution) {
            state.step2SubmitDeploy = "success";
            state.step2Deploying = "skipped";
            state.step2DeployComplete = "success";
          }
          state.step3SubmitEval = "success";
          state.step3Evaluating = "running";
          state.step3EvalComplete = "pending";
        } else if (evt === "model_evluation_result") {
          const st = String(p.status || "").toUpperCase();
          if (p.deployment_id != null && String(p.deployment_id).trim()) {
            state.deploymentId = String(p.deployment_id);
          }
          const bidEv = p.benchmark_id != null && String(p.benchmark_id).trim() ? String(p.benchmark_id) : "";
          if (bidEv) {
            state.benchmarkId = bidEv;
          }
          if (st === "SUCCESS") {
            state.step3Evaluating = "success";
            state.step3EvalComplete = "success";
            setResultFromPreview(p.result_preview);
          } else {
            state.step3Evaluating = "failed";
            state.step3EvalComplete = "failed";
            state.errorMessage = String(p.message || "模型评测失败");
          }
        } else if (evt === "adela_pipeline_error") {
          state.finalResultText = "";
          const base = String(p.message || "Adela流程执行失败").trim();
          const detail = String(p.error_detail || "").trim();
          const blob = (base + " " + detail).toLowerCase();
          if (blob.indexOf("unsupported platform") >= 0 || blob.indexOf("unsupported_platform") >= 0) {
            state.errorMessage =
              "模型部署失败：当前填写的部署平台不被 Adela 支持。请提供规范的部署平台，例如 cuda11.0-trt7.1-fp16-T4、cuda11.0-trt7.1-int8-T4、cuda11.0-trt7.1-fp32-T4。格式通常为 cuda<版本>-trt<版本>-<精度>-<GPU型号>，也可使用 acl-、cpu-、rknn- 前缀的平台标识。";
          } else {
            state.errorMessage = detail && base.indexOf(detail) < 0 ? base + " 原因：" + detail : base;
          }
          if (state.step1DeployQuery === "pending") state.step1DeployQuery = "failed";
          if (state.step1BenchQuery === "pending") state.step1BenchQuery = "skipped";
          if (state.showQuantDatasetStep && state.step1QuantQuery === "pending") state.step1QuantQuery = "skipped";
          if (state.step1EvalDsQuery === "pending") state.step1EvalDsQuery = "skipped";
          markFollowingSkippedAfterDataStop();
        }
        render();
      },
      stepDone(step, elapsedMs) {
        if (step !== "adela_cli") return;
        render();
      },
      finalAnswer(t) {
        if (state.finalResultText) {
          stream("final", "回答", state.finalResultText);
          return;
        }
        stream("final", "回答", t);
      },
      error: err,
    };
  }
  const intent = section("intent", "任务理解");
  replace(intent, MSG_RUNNING);
  const genUrls = [];
  return {
    onDecision, direct: noop, ragRefs: noop, ragRun: noop, detectionInfo: noop, adela: noop,
    intent(s) {
      const keys = [["task_name","任务名称"],["scene","场景"],["target","目标"],["camera","视角"]];
      const html = keys.map(([k,n]) => '<div class="kv-row"><b>' + n + "</b>" + esc((s || {})[k] || "") + "</div>").join("");
      replace(intent, html || "<div class='pending-msg'>任务理解结果为空</div>");
      const gen = section("gen", "生成图片（Flux）");
      replace(gen, MSG_RUNNING);
    },
    gen(url) {
      if (!url) return;
      genUrls.push(url);
      const gen = section("gen", "生成图片（Flux）");
      replace(gen, '<div class="thumb-row">' + genUrls.map(u => '<img src="' + esc(u) + '" loading="lazy" />').join("") + "</div>");
      const anno = section("anno", "标注结果（红:Qwen 蓝:Rex）");
      replace(anno, MSG_RUNNING);
    },
    annotated(urls) {
      const anno = section("anno", "标注结果（红:Qwen 蓝:Rex）");
      replace(anno, '<div class="thumb-row">' + (urls || []).map(u => '<img src="' + esc(u) + '" loading="lazy" />').join("") + "</div>");
      const evalD = section("eval", "评估报告");
      replace(evalD, MSG_RUNNING);
    },
    eval(data) {
      const evalD = section("eval", "评估报告");
      let html = "";
      if (data && data.overall_conclusion) {
        html += '<div class="kv-row"><b>总体结论</b>' + esc(data.overall_conclusion) + "</div>";
      }
      if (data && data.model_results && typeof data.model_results === "object") {
        const modelItems = Object.entries(data.model_results)
          .map(([name, result]) => {
            const acc = result && result.accuracy ? String(result.accuracy) : "";
            const reason = result && result.reason ? String(result.reason) : "";
            let block = '<div class="kv-row"><b>' + esc(name) + "</b>";
            if (acc) block += "准确率：" + esc(acc);
            if (reason) block += (acc ? "；" : "") + "说明：" + esc(reason);
            block += "</div>";
            return block;
          })
          .join("");
        if (modelItems) html += '<div class="kv-row"><b>分模型结果</b></div>' + modelItems;
      }
      if (data && data.recommendation) html += '<div class="kv-row"><b>建议</b>' + esc(data.recommendation) + "</div>";
      replace(evalD, html || "<div class='pending-msg'>（无摘要字段）</div>");
    },
    stepDone: noop, reQuestion: noop, adela: noop, finalAnswer(t) { stream("final", "回答", t); }, error: err
  };
}

function createAssistantTurn(options) {
  options = options || {};
  const steps = new Map();
  let current = "step_1";
  function key(i) { return "step_" + String(Math.max(1, parseInt(String(i || 1), 10) || 1)); }
  function ensure(flow, decision, idx) {
    const k = key(idx);
    current = k;
    const targetFlow = flow || "direct_answer";
    const cached = steps.get(k);
    if (!cached) {
      const bubble = addMessage("assistant", "");
      steps.set(k, {
        flow: targetFlow,
        renderer: createAssistantStepForFlow(bubble, targetFlow, decision || {}, options),
      });
    } else if (cached.flow !== targetFlow) {
      // 同一 step 里 flow 可能从 re_question 切到 rag，必须重建渲染器避免事件落到错误视图。
      const bubble = addMessage("assistant", "");
      steps.set(k, {
        flow: targetFlow,
        renderer: createAssistantStepForFlow(bubble, targetFlow, decision || {}, options),
      });
    } else if (decision && cached.renderer && cached.renderer.onDecision) {
      cached.renderer.onDecision(decision);
    }
    const now = steps.get(k);
    return now ? now.renderer : null;
  }
  function cur() {
    const cached = steps.get(current);
    return (cached && cached.renderer) || ensure("direct_answer", {}, 1);
  }
  return {
    onMeta(evt) { return ensure((evt && evt.flow) || "direct_answer", (evt && evt.decision) || {}, (evt && evt.step_index) || 1); },
    direct(t) { cur().direct(t); },
    stepDone(s, ms) { if (cur().stepDone) cur().stepDone(s, ms); },
    ragRefs(r) { cur().ragRefs(r); },
    ragRun(p) { if (cur().ragRun) cur().ragRun(p); },
    detectionInfo(l) { cur().detectionInfo(l); },
    intent(s) { cur().intent(s); },
    gen(u) { cur().gen(u); },
    annotated(u) { cur().annotated(u); },
    eval(d) { cur().eval(d); },
    reQuestion(p) { if (cur().reQuestion) cur().reQuestion(p); },
    adela(p) { if (cur().adela) cur().adela(p); },
    migrationOffer(p) { if (cur().migrationOffer) cur().migrationOffer(p); },
    migrationPlan(p) { if (cur().migrationPlan) cur().migrationPlan(p); },
    migrationRetrieve(p) { if (cur().migrationRetrieve) cur().migrationRetrieve(p); },
    migrationRex(p) { if (cur().migrationRex) cur().migrationRex(p); },
    migrationReport(p) { if (cur().migrationReport) cur().migrationReport(p); },
    clarification(t, p) {
      const stepIndex = (p && p.step_index) || 1;
      const clarifyRenderer = ensure("clarify", { action: "clarify", reason: "" }, stepIndex);
      if (clarifyRenderer && clarifyRenderer.clarification) clarifyRenderer.clarification(t, p);
    },
    finalAnswer(t) { cur().finalAnswer(t); },
    error(m) { cur().error(m); },
  };
}

function limitImageFiles(files) {
  const arr = Array.isArray(files) ? files.filter((f) => String(f.type || "").startsWith("image/")) : [];
  if (arr.length <= MAX_UPLOAD_IMAGES) return arr;
  return arr.slice(0, MAX_UPLOAD_IMAGES);
}

function imageFileKey(file) {
  return [file.name, file.size, file.lastModified].join("|");
}

function getPendingImageFiles() {
  return pendingImageFiles.slice();
}

function clearPendingImageFiles() {
  pendingImageFiles = [];
  imageFileInput.value = "";
  renderPendingImages();
}

function renderPendingImages() {
  pickedPreviewEl.innerHTML = "";
  const files = getPendingImageFiles();
  if (!files.length) {
    pickedLabel.textContent = "";
    return;
  }
  pickedLabel.textContent = "已添加 " + files.length + " 张图片（最多 " + MAX_UPLOAD_IMAGES + " 张，可继续点 ＋ 添加）: "
    + files.map((f) => f.name).join(", ");
  files.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "picked-item";
    item.title = file.name;
    const img = document.createElement("img");
    img.alt = file.name;
    const objUrl = URL.createObjectURL(file);
    img.src = objUrl;
    img.onload = () => URL.revokeObjectURL(objUrl);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "×";
    rm.title = "移除";
    rm.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      pendingImageFiles = pendingImageFiles.filter((_, i) => i !== idx);
      renderPendingImages();
    });
    item.appendChild(img);
    item.appendChild(rm);
    pickedPreviewEl.appendChild(item);
  });
}

function addPendingImageFiles(newFiles) {
  const incoming = limitImageFiles(newFiles);
  if (!incoming.length) return { added: 0, truncated: false };
  const seen = new Set(pendingImageFiles.map(imageFileKey));
  let added = 0;
  let truncated = false;
  for (const file of incoming) {
    if (pendingImageFiles.length >= MAX_UPLOAD_IMAGES) {
      truncated = true;
      break;
    }
    const key = imageFileKey(file);
    if (seen.has(key)) continue;
    seen.add(key);
    pendingImageFiles.push(file);
    added += 1;
  }
  renderPendingImages();
  return { added, truncated };
}

function onPickChanged() {
  const raw = imageFileInput.files ? Array.from(imageFileInput.files) : [];
  imageFileInput.value = "";
  if (!raw.length) return;
  const { added, truncated } = addPendingImageFiles(raw);
  if (truncated) {
    setStatus("最多 " + MAX_UPLOAD_IMAGES + " 张图片，多余的未添加");
  } else if (added > 0) {
    setStatus("已添加 " + added + " 张图片");
  } else {
    setStatus("图片已存在，未重复添加");
  }
}
imageFileInput.addEventListener("change", onPickChanged);
pickBtn.addEventListener("click", () => imageFileInput.click());

function renderSessionList(items) {
  const arr = Array.isArray(items) ? items : [];
  sessionListEl.innerHTML = "";
  if (!arr.length) {
    sessionListEl.innerHTML = "<div class='session-meta'>暂无会话记录</div>";
    return;
  }
  arr.forEach((s) => {
    const d = document.createElement("div");
    d.className = "session-item" + (s.session_id === currentSessionId ? " active" : "");
    d.innerHTML =
      '<div class="session-title">' + esc(s.title || s.session_id || "") + '</div>' +
      '<div class="session-meta">' + esc((s.preview || ("会话ID: " + s.session_id || "")).slice(0, 80)) + "</div>" +
      '<button class="session-delete" title="删除会话">×</button>';
    d.addEventListener("click", () => loadSession(s.session_id));
    const delBtn = d.querySelector(".session-delete");
    if (delBtn) {
      delBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!window.confirm("确认删除该会话记录？删除后不可恢复。")) return;
        await deleteSession(s.session_id);
      });
    }
    sessionListEl.appendChild(d);
  });
}

async function deleteSession(sessionId) {
  if (!sessionId) return;
  try {
    const r = await fetch("/session/delete", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": CLIENT_ID,
      },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data.ok) {
      setStatus("删除失败");
      return;
    }
    if (currentSessionId === sessionId) {
      currentSessionId = createFreshSessionId();
      currentThreadId = "";
      chatEl.innerHTML = "";
    }
    setStatus("会话已删除");
    await refreshSessionList();
  } catch (_) {
    setStatus("删除失败");
  }
}

async function refreshSessionList() {
  try {
    const r = await fetch("/sessions", {
      headers: { "X-Client-Id": CLIENT_ID },
    });
    if (!r.ok) return;
    const data = await r.json();
    renderSessionList(data.sessions || []);
  } catch (_) {}
}

function newChat() {
  currentSessionId = createFreshSessionId();
  currentThreadId = "";
  chatEl.innerHTML = "";
  clearPendingImageFiles();
  setStatus("新会话已创建");
  refreshSessionList();
  textEl.focus();
}

function canBootstrapTurnFromEvent(evt) {
  if (!evt || typeof evt !== "object") return false;
  return evt.type === "direct_reply" || evt.type === "final_answer" || evt.type === "clarification" || evt.type === "migration_advisor_offer" || evt.type === "error";
}

function isIgnorableUnknownActionError(evt) {
  if (!evt || typeof evt !== "object") return false;
  const msg = String(evt.message || "").trim();
  return msg.startsWith("未知 Agent 工具 action：");
}

function isIgnorableDecisionReason(reason) {
  const text = String(reason || "").trim();
  return text === "用户已补充澄清信息，继续执行上次已确定的工具动作。";
}

function shouldIgnoreMetaEvent(evt) {
  if (!evt || typeof evt !== "object" || evt.type !== "meta") return false;
  const action = String((evt.decision && evt.decision.action) || "").trim();
  const reason = String((evt.decision && evt.decision.reason) || "").trim();
  if (!action && !reason) return true;
  if (!action && isIgnorableDecisionReason(reason)) return true;
  if (!action) return false;
  return !KNOWN_AGENT_ACTIONS.has(action);
}

function finalAnswerText(evt) {
  if (!evt || typeof evt !== "object") return "";
  return String(evt.text || evt.final_answer || "").trim();
}

async function loadSession(sessionId) {
  if (!sessionId) return;
  currentSessionId = sessionId;
  currentThreadId = "";
  chatEl.innerHTML = "";
  try {
    const r = await fetch("/session?session_id=" + encodeURIComponent(sessionId), {
      headers: { "X-Client-Id": CLIENT_ID },
    });
    if (!r.ok) {
      setStatus("会话加载失败");
      refreshSessionList();
      return;
    }
    const data = await r.json();
    currentThreadId = String(data.active_thread_id || "").trim();
    const turns = Array.isArray(data.chat_turns) ? data.chat_turns : [];
    if (turns.length) {
      turns.forEach((turn) => {
        const q = String((turn && turn.user_text) || "");
        if (q) addMessage("user", "<div>" + esc(q).replace(/\\n/g, "<br/>") + "</div>");
        const events = Array.isArray(turn && turn.events) ? turn.events : [];
        let turnRenderer = null;
        let flowHint = "";
        events.forEach((evt) => {
          if (!evt || typeof evt !== "object") return;
          if (evt.type === "session" || evt.type === "done") return;
          if (evt.type === "meta") {
            if (shouldIgnoreMetaEvent(evt)) return;
            if (!turnRenderer) turnRenderer = createAssistantTurn({ streaming: false });
            flowHint = evt.flow || flowHint || "direct_answer";
            turnRenderer.onMeta(evt);
            return;
          }
          if (isIgnorableUnknownActionError(evt)) return;
          if (!turnRenderer) {
            if (!canBootstrapTurnFromEvent(evt)) return;
            turnRenderer = createAssistantTurn({ streaming: false });
            turnRenderer.onMeta({ flow: flowHint || "direct_answer", decision: {}, step_index: 1 });
          }
          if (evt.type === "direct_reply") turnRenderer.direct(evt.text || "");
          if (evt.type === "step_timing") turnRenderer.stepDone(evt.step, evt.elapsed_ms || 0);
          if (evt.type === "rag_run") turnRenderer.ragRun(evt);
          if (evt.type === "rag_references") turnRenderer.ragRefs(evt.references || []);
          if (evt.type === "detection_info") turnRenderer.detectionInfo(evt.label || "");
          if (evt.type === "intent_summary") turnRenderer.intent(evt.summary || {});
          if (evt.type === "generated_one" && evt.url) turnRenderer.gen(evt.url);
          if (evt.type === "annotated" && evt.urls) turnRenderer.annotated(evt.urls);
          if (evt.type === "evaluation" && evt.data) turnRenderer.eval(evt.data);
          if (evt.type === "re_question") turnRenderer.reQuestion(evt);
          if (evt.type === "adela_event") turnRenderer.adela(evt);
          if (evt.type === "migration_advisor_offer") turnRenderer.migrationOffer(evt);
          if (evt.type === "migration_advisor_plan") turnRenderer.migrationPlan(evt);
          if (evt.type === "migration_advisor_retrieve") turnRenderer.migrationRetrieve(evt);
          if (evt.type === "migration_advisor_rex") turnRenderer.migrationRex(evt);
          if (evt.type === "migration_advisor_report") turnRenderer.migrationReport(evt);
          if (evt.type === "clarification") turnRenderer.clarification(evt.text || "", evt);
          if (evt.type === "final_answer") turnRenderer.finalAnswer(finalAnswerText(evt));
          if (evt.type === "error") turnRenderer.error(evt.message || "执行失败");
        });
      });
    } else {
      const history = Array.isArray(data.history) ? data.history : [];
      const historyRefs = data && typeof data.history_refs === "object" && data.history_refs ? data.history_refs : {};
      history.forEach((item) => {
        const q = String((item && item.query) || "");
        const a = String((item && item.final_answer) || "");
        if (q) addMessage("user", "<div>" + esc(q).replace(/\\n/g, "<br/>") + "</div>");
        if (a) {
          let html = "<div>" + esc(a).replace(/\\n/g, "<br/>") + "</div>";
          const refs = Array.isArray(historyRefs[q]) ? historyRefs[q] : [];
          const valid = refs.filter(x => x && x.url);
          if (valid.length) {
            html += '<div class="section"><div class="section-title">参考链接</div>';
            valid.forEach((it, idx) => {
              const name = String(it.doc_name || ("参考 " + (idx + 1)));
              const u = String(it.url || "");
              html += '<div style="margin:6px 0"><a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(name) + "</a></div>";
            });
            html += "</div>";
          }
          addMessage("assistant", html);
        }
      });
    }
    setStatus("已加载历史会话");
  } catch (_) {
    setStatus("会话加载失败");
  }
  refreshSessionList();
}

async function sendMessage(options) {
  options = options || {};
  if (isRunning) return;
  const text = textEl.value.trim();
  const files = getPendingImageFiles();
  if (!text && !files.length) return;
  if (!currentSessionId) currentSessionId = createFreshSessionId();

  isRunning = true;
  setStatus("发送中...");

  const shownText = options.hiddenStructured ? String(options.displayText || "提交参数") : text;
  const userHtml = shownText
    ? ("<div>" + esc(shownText).replace(/\\n/g, "<br/>") + "</div>")
    : (files.length > 1
      ? "<div class='pending-msg'>[上传 " + files.length + " 张图片]</div>"
      : "<div class='pending-msg'>[仅上传图片]</div>");
  const userBubble = addMessage("user", userHtml);
  if (!options.hiddenStructured && files.length) {
    const grid = document.createElement("div");
    grid.className = files.length > 1 ? "preview-grid" : "";
    files.forEach((f) => {
      const img = document.createElement("img");
      img.className = "preview";
      img.src = URL.createObjectURL(f);
      img.onload = () => URL.revokeObjectURL(img.src);
      grid.appendChild(img);
    });
    userBubble.appendChild(grid);
  }

  const fd = new FormData();
  fd.append("session_id", currentSessionId);
  if (currentThreadId) fd.append("thread_id", currentThreadId);
  fd.append("text", text);
  files.forEach((f) => fd.append("image", f));

  textEl.value = "";
  clearPendingImageFiles();

  let turn = null;
  let currentFlow = "";
  let requestHadException = false;

  try {
    const resp = await fetch("/run", {
      method: "POST",
      body: fd,
      headers: { "X-Client-Id": CLIENT_ID },
    });
    if (!resp.ok) {
      const t = await resp.text();
      turn = createAssistantTurn();
      turn.onMeta({ flow: "direct_answer", decision: {}, step_index: 1 });
      turn.error("HTTP " + resp.status + ": " + t.slice(0, 500));
      setStatus("请求失败");
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.type === "thread_routed" && evt.thread_id) {
          currentThreadId = String(evt.thread_id);
          continue;
        }
        if (evt.type === "session") {
          if (evt.session_id) currentSessionId = String(evt.session_id);
          if (evt.thread_id) currentThreadId = String(evt.thread_id);
        }
        if (evt.type === "meta") {
          if (shouldIgnoreMetaEvent(evt)) continue;
          const nextFlow = evt.flow || currentFlow || "direct_answer";
          if (!turn) turn = createAssistantTurn();
          currentFlow = nextFlow;
          turn.onMeta(evt);
          continue;
        }
        if (isIgnorableUnknownActionError(evt)) continue;
        if (!turn && evt.type !== "session") {
          if (!canBootstrapTurnFromEvent(evt)) continue;
          turn = createAssistantTurn();
          turn.onMeta({ flow: currentFlow || "direct_answer", decision: {}, step_index: 1 });
        }
        if (!turn) continue;
        if (evt.type === "direct_reply") turn.direct(evt.text || "");
        if (evt.type === "step_timing") turn.stepDone(evt.step, evt.elapsed_ms || 0);
        if (evt.type === "rag_run") turn.ragRun(evt);
        if (evt.type === "rag_references") turn.ragRefs(evt.references || []);
        if (evt.type === "detection_info") turn.detectionInfo(evt.label || "");
        if (evt.type === "intent_summary") turn.intent(evt.summary || {});
        if (evt.type === "generated_one" && evt.url) turn.gen(evt.url);
        if (evt.type === "annotated" && evt.urls) turn.annotated(evt.urls);
        if (evt.type === "evaluation" && evt.data) turn.eval(evt.data);
        if (evt.type === "re_question") turn.reQuestion(evt);
        if (evt.type === "adela_event") turn.adela(evt);
        if (evt.type === "migration_advisor_offer") turn.migrationOffer(evt);
        if (evt.type === "migration_advisor_plan") turn.migrationPlan(evt);
        if (evt.type === "migration_advisor_retrieve") turn.migrationRetrieve(evt);
        if (evt.type === "migration_advisor_rex") turn.migrationRex(evt);
        if (evt.type === "migration_advisor_report") turn.migrationReport(evt);
        if (evt.type === "clarification") turn.clarification(evt.text || "", evt);
        if (evt.type === "final_answer") turn.finalAnswer(finalAnswerText(evt));
        if (evt.type === "error") turn.error(evt.message || "执行失败");
        if (evt.type === "done") setStatus(evt.ok ? "已完成，可继续提问" : "执行失败，可继续提问");
      }
    }
  } catch (e) {
    requestHadException = true;
    if (!turn) {
      turn = createAssistantTurn();
      turn.onMeta({ flow: "direct_answer", decision: {}, step_index: 1 });
    }
    turn.error(String(e));
    setStatus("请求异常");
  } finally {
    isRunning = false;
    runBtn.disabled = false;
    if (!requestHadException) {
      setStatus("已完成，可继续提问");
      refreshSessionList();
    }
  }
}

newChatBtn.addEventListener("click", newChat);
runBtn.addEventListener("click", () => sendMessage());
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

(async function boot() {
  await refreshSessionList();
  newChat();
})();
</script>
</body>
</html>
"""
