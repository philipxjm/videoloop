const progressMetric = document.getElementById('progressMetric');
const accuracyMetric = document.getElementById('accuracyMetric');
const rateMetric = document.getElementById('rateMetric');
const progressBar = document.getElementById('progressBar');
const workerGrid = document.getElementById('workerGrid');
const totalQuestions = document.getElementById('totalQuestions');
const estimatedCorrect = document.getElementById('estimatedCorrect');
const runMetric = document.getElementById('runMetric');
const videoMetric = document.getElementById('videoMetric');
const questionMetric = document.getElementById('questionMetric');
const questionList = document.getElementById('questionList');
const detailHeader = document.getElementById('detailHeader');
const detailBody = document.getElementById('detailBody');
const tokensMetric = document.getElementById('tokensMetric');
const tokensBreakdown = document.getElementById('tokensBreakdown');
const costMetric = document.getElementById('costMetric');
const costBreakdown = document.getElementById('costBreakdown');
const tokensAgentBreakdown = document.getElementById('tokensAgentBreakdown');
const costAgentBreakdown = document.getElementById('costAgentBreakdown');

// Pricing: loaded from /api/config, with sensible defaults
let PRICE_INPUT_PER_M = 0;   // per-M input tokens (default / main_agent)
let PRICE_OUTPUT_PER_M = 0;  // per-M output tokens (default / main_agent)
let PRICE_CURRENCY = 'USD';
// Per-agent pricing overrides (vlm and summarizer may use different models/rates)
let AGENT_PRICING = {};  // e.g. { vlm: {input_per_million: 0.75, output_per_million: 4.5}, ... }
const runSelect = document.getElementById('runSelect');
const refreshRun = document.getElementById('refreshRun');

let questionsCache = [];
let activeQuestionId = null;
let displayedQuestionId = null;  // Track which question is actually rendered in detail view
let runIndex = [];
let currentRun = null;
let _lastDetailFingerprint = '';  // Track whether detail view needs re-render

// --- Performance: trajectory is fetched on demand per question ---
// Static mode: re-fetches the run file and extracts one question's detail.
// Live mode: uses /api/questions/{id} or /api/live/workers/{id}.
let _currentRunFile = null;  // Track which run file is loaded for on-demand fetch

let _pendingRemainingGroups = null;  // Temp storage for lazy trajectory rendering
const _expandedQuestionIds = new Set();  // qids where "Show more" was clicked — keep them fully expanded across re-renders

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function fmt(value, suffix = '') {
  return value === null || value === undefined ? '--' : `${value}${suffix}`;
}

function fmtTokens(n) {
  if (n === null || n === undefined || n === 0) return '--';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function fmtCost(amount) {
  if (amount === null || amount === undefined || amount === 0) return '--';
  const sym = PRICE_CURRENCY === 'USD' ? '$' : '¥';
  if (amount >= 1) return `${sym}${amount.toFixed(2)}`;
  return `${sym}${amount.toFixed(3)}`;
}

function calcCost(inputTokens, outputTokens, agentType) {
  const ap = agentType && AGENT_PRICING[agentType];
  const inRate = ap ? ap.input_per_million : PRICE_INPUT_PER_M;
  const outRate = ap ? ap.output_per_million : PRICE_OUTPUT_PER_M;
  const inCost = (inputTokens || 0) / 1_000_000 * inRate;
  const outCost = (outputTokens || 0) / 1_000_000 * outRate;
  return { inCost, outCost, total: inCost + outCost };
}

const AGENT_LABELS = { main_agent: 'Main', vlm: 'VLM', summarizer: 'Summ', skill_selector: 'Skill', transcript_prescreen: 'Prescreen' };

function renderAgentBreakdowns(byAgent) {
  if (!byAgent || !Object.keys(byAgent).length) {
    tokensAgentBreakdown.innerHTML = '';
    costAgentBreakdown.innerHTML = '';
    return;
  }
  const sorted = Object.entries(byAgent).sort((a, b) => (b[1].total_tokens || 0) - (a[1].total_tokens || 0));
  let tokHtml = '';
  let costHtml = '';
  for (const [agent, c] of sorted) {
    const label = AGENT_LABELS[agent] || agent;
    const tok = c.total_tokens || 0;
    if (tok === 0) continue;
    tokHtml += `<div class="agent-row"><span class="agent-name">${label}</span><span class="agent-value">${fmtTokens(c.input_tokens)}→${fmtTokens(c.output_tokens)}</span></div>`;
    const ac = calcCost(c.input_tokens, c.output_tokens, agent);
    costHtml += `<div class="agent-row"><span class="agent-name">${label}</span><span class="agent-value">${fmtCost(ac.total)}</span></div>`;
  }
  tokensAgentBreakdown.innerHTML = tokHtml;
  costAgentBreakdown.innerHTML = costHtml;
}

function calcTotalCostByAgent(byAgent) {
  if (!byAgent) return { inCost: 0, outCost: 0, total: 0 };
  let totalIn = 0, totalOut = 0;
  for (const [agent, c] of Object.entries(byAgent)) {
    const ac = calcCost(c.input_tokens, c.output_tokens, agent);
    totalIn += ac.inCost;
    totalOut += ac.outCost;
  }
  return { inCost: totalIn, outCost: totalOut, total: totalIn + totalOut };
}

function renderWorkerGrid() {
  if (!liveWorkers || liveWorkers.length === 0) {
    // In archived/non-live mode, show nothing useful here
    if (!liveMode) {
      workerGrid.textContent = 'Worker grid available during live evaluations.';
      return;
    }
    workerGrid.textContent = 'No workers active.';
    return;
  }

  // Determine total worker slots (use highest worker_id + 1, or num_workers from meta)
  const numWorkers = _lastGoodQMeta.num_workers
    || Math.max(...liveWorkers.map(w => (w.worker_id || 0) + 1), liveWorkers.length);

  // Build a map of active workers by ID
  const activeMap = {};
  liveWorkers.forEach(w => { activeMap[w.worker_id ?? 0] = w; });

  // Also map completed questions by worker_id for "done" status
  const doneMap = {};
  questionsCache.forEach(q => {
    if (q.worker_id !== undefined) doneMap[q.worker_id] = q;
  });

  let html = '';
  for (let i = 0; i < numWorkers; i++) {
    const w = activeMap[i];
    if (w) {
      const steps = (w.trajectory || []).length;
      const qId = w.question_id || '';
      const qText = (w.question || '').slice(0, 40);
      html += `<div class="worker-cell active">` +
        `<div class="worker-cell-id">W${i}</div>` +
        `<div class="worker-cell-question" title="${escapeHtml(w.question || '')}">${escapeHtml(qId)}</div>` +
        `<div class="worker-cell-steps">${steps} steps</div>` +
        `</div>`;
    } else {
      // Check if this worker recently finished
      const done = doneMap[i];
      if (done) {
        const status = done.correct === true ? 'correct' : done.correct === false ? 'wrong' : 'done';
        const pill = done.correct === true ? '✓' : done.correct === false ? '✗' : '—';
        html += `<div class="worker-cell done">` +
          `<div class="worker-cell-id">W${i} ${pill}</div>` +
          `<div class="worker-cell-question">${escapeHtml(done.question_id || '')}</div>` +
          `</div>`;
      } else {
        html += `<div class="worker-cell idle">` +
          `<div class="worker-cell-id">W${i}</div>` +
          `<div class="worker-cell-question">idle</div>` +
          `</div>`;
      }
    }
  }
  workerGrid.innerHTML = html;
}

function updateSummary(run) {
  const summary = run.summary || {};
  const total = summary.total_questions ?? (run.questions ? run.questions.length : 0);
  const correct = summary.correct ?? null;
  const accuracy = summary.accuracy ?? (correct !== null && total ? correct / total : null);
  const rate = summary.elapsed_seconds && total ? summary.elapsed_seconds / total : null;

  progressMetric.textContent = `${fmt(correct)} / ${fmt(total)}`;
  accuracyMetric.textContent = accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : '--%';
  rateMetric.textContent = rate !== null ? `${rate.toFixed(1)} s/q` : '-- s/q';

  if (total && correct !== null) {
    const pct = Math.min(100, Math.max(0, (correct / total) * 100));
    progressBar.style.width = `${pct}%`;
  }

  totalQuestions.textContent = `Total questions: ${fmt(total)}`;
  estimatedCorrect.textContent = correct !== null ? `Correct: ${correct}` : 'Correct: --';
  runMetric.textContent = run.run_id || '—';
  videoMetric.textContent = run.model || '—';
  questionMetric.textContent = run.dataset || '—';

  // Token usage for archived runs (aggregated from per-question data)
  if (run.questions && run.questions.length) {
    let aggIn = 0, aggOut = 0;
    const aggByAgent = {};
    run.questions.forEach(q => {
      const t = (q.token_usage || {}).total || {};
      aggIn += (t.input_tokens || 0);
      aggOut += (t.output_tokens || 0);
      const ba = (q.token_usage || {}).by_agent || {};
      for (const [agent, c] of Object.entries(ba)) {
        if (!aggByAgent[agent]) aggByAgent[agent] = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
        aggByAgent[agent].input_tokens += (c.input_tokens || 0);
        aggByAgent[agent].output_tokens += (c.output_tokens || 0);
        aggByAgent[agent].total_tokens += (c.total_tokens || 0);
      }
    });
    tokensMetric.textContent = fmtTokens(aggIn + aggOut);
    tokensBreakdown.textContent = `In: ${fmtTokens(aggIn)} | Out: ${fmtTokens(aggOut)}`;
    const cost = calcTotalCostByAgent(aggByAgent);
    costMetric.textContent = fmtCost(cost.total);
    costBreakdown.textContent = `In: ${fmtCost(cost.inCost)} | Out: ${fmtCost(cost.outCost)}`;
    renderAgentBreakdowns(aggByAgent);
  } else {
    tokensMetric.textContent = '--';
    tokensBreakdown.textContent = 'In: -- | Out: --';
    costMetric.textContent = '--';
    costBreakdown.textContent = 'In: -- | Out: --';
    renderAgentBreakdowns(null);
  }
}

// DOM node cache for completed questions — avoids full list rebuild on every poll.
// Keys are String(question_id). Signature encodes the fields visible in the list row;
// we only rebuild a row's innerHTML when its signature changes.
const _questionNodes = new Map();    // qid → DOM element
const _questionNodeSigs = new Map(); // qid → last-rendered signature
const _questionCostCache = new Map(); // qid → calcTotalCostByAgent result (immutable once set)
const _liveWorkerNodes = new Map();  // wkey → DOM element (in-place updates)
const _liveWorkerSigs = new Map();   // wkey → signature to skip unchanged renders

function _qListSig(q) {
  // Fields that affect the rendered row content (excluding active highlight)
  return `${q.correct}|${q.predicted}|${q.elapsed_seconds}|${!!q.error}`;
}

function _qCost(q) {
  const qid = String(q.question_id);
  if (_questionCostCache.has(qid)) return _questionCostCache.get(qid);
  const qTokTotal = q.token_usage?.total?.total_tokens;
  if (!qTokTotal) return null;
  const cost = calcTotalCostByAgent(q.token_usage?.by_agent || {});
  _questionCostCache.set(qid, cost);
  return cost;
}

function _clearQuestionNodeCaches() {
  _questionNodes.clear();
  _questionNodeSigs.clear();
  _questionCostCache.clear();
  // Clear question list DOM (except live workers)
  questionList.querySelectorAll('.question-item:not(.ql-live-worker)').forEach(el => el.remove());
}

function _buildQuestionItemHTML(q) {
  const isFatal = q.error && !q.predicted;
  const statusPill = isFatal ? '<span class="pill error">Error</span>'
    : q.correct === true  ? '<span class="pill">Correct</span>'
    : q.correct === false ? '<span class="pill bad">Wrong</span>'
    : '<span class="pill neutral">Pending</span>';
  const workerId = q.worker_id !== undefined ? `W${q.worker_id} • ` : '';
  const qTokIn  = q.token_usage?.total?.input_tokens  || 0;
  const qTokOut = q.token_usage?.total?.output_tokens || 0;
  const qTokTotal = q.token_usage?.total?.total_tokens;
  const qCost = _qCost(q);
  const qTokText  = qTokTotal ? ` | ${fmtTokens(qTokIn)}→${fmtTokens(qTokOut)}` : '';
  const qCostText = qCost ? ` ${fmtCost(qCost.total)}` : '';
  const qTime     = q.elapsed_seconds ? `${Math.round(q.elapsed_seconds)}s` : '';
  const qTimeTok  = (qTime || qTokText) ? ` | ${qTime}${qTokText}${qCostText}` : '';
  const errorLine = isFatal ? `<div class="question-error">${escapeHtml(q.error).slice(0, 120)}</div>` : '';
  return `
    <div class="question-meta">${workerId}${q.video_id || ''} • ${q.question_id || ''}</div>
    <div class="question-title">${statusPill}${(q.question || '').slice(0, 80)}${(q.question || '').length > 80 ? '…' : ''}</div>
    <div class="question-meta">Pred: ${q.predicted || '—'} | Exp: ${q.expected || '—'}${qTimeTok}</div>
    ${errorLine}
  `;
}

let _lastQuestionListFingerprint = '';

function _questionListFingerprint() {
  const workerPart = (liveWorkers || []).map(w => `${w.worker_id}:${w.question_id}:${w.trajectory_steps ?? (w.trajectory||[]).length}`).join(',');
  const livePart = liveQuestion ? `${liveQuestion.question_id}:${liveQuestion.trajectory_steps ?? (liveQuestion.trajectory||[]).length}` : '';
  const cachePart = questionsCache.length + ':' + (questionsCache[questionsCache.length - 1]?.question_id || '');
  return `${workerPart}|${livePart}|${cachePart}|${activeQuestionId}`;
}

function renderQuestionList() {
  // Skip re-render if nothing changed (prevents scroll reset during polling)
  const fp = _questionListFingerprint();
  if (fp === _lastQuestionListFingerprint) return;
  _lastQuestionListFingerprint = fp;

  const savedScroll = questionList.scrollTop;

  // --- Live workers section (in-place updates to avoid scroll resets) ---
  // Also remove the empty-state placeholder if it exists
  questionList.querySelectorAll('.ql-empty').forEach(el => el.remove());

  const activeLiveIds = new Set();

  if (liveWorkers && liveWorkers.length > 0) {
    liveWorkers.forEach((worker) => {
      const wkey = `live-w${worker.worker_id ?? 0}`;
      activeLiveIds.add(wkey);
      const isActive = worker.question_id === activeQuestionId;
      const stepCount = worker.trajectory_steps ?? (worker.trajectory || []).length;
      const workerId = worker.worker_id !== undefined ? `W${worker.worker_id}` : '';
      const workerTokens = worker.token_usage?.total?.total_tokens;
      const tokenText = workerTokens ? ` | Tokens: ${fmtTokens(workerTokens)}` : '';
      const sig = `${worker.question_id}:${stepCount}:${workerTokens}:${isActive}`;

      let el = _liveWorkerNodes.get(wkey);
      if (!el) {
        el = document.createElement('div');
        el.className = 'question-item working ql-live-worker';
        _liveWorkerNodes.set(wkey, el);
        // Insert at top, after any existing live worker nodes
        const lastLive = questionList.querySelector('.ql-live-worker:last-of-type');
        if (lastLive && lastLive.nextSibling) questionList.insertBefore(el, lastLive.nextSibling);
        else if (lastLive) lastLive.after(el);
        else questionList.prepend(el);
      }

      if (_liveWorkerSigs.get(wkey) !== sig) {
        el.className = `question-item working ql-live-worker${isActive ? ' active' : ''}`;
        el.innerHTML = `
          <div class="question-meta">${workerId} • ${worker.video_id || ''} • ${worker.question_id || ''}</div>
          <div class="question-title"><span class="pill working">Working...</span>${(worker.question || '').slice(0, 70)}${(worker.question || '').length > 70 ? '…' : ''}</div>
          <div class="question-meta">Steps: ${stepCount}${tokenText} | Exp: ${worker.expected || '—'}${worker.predicted ? ' | Pred: ' + worker.predicted : ''}</div>
        `;
        el.onclick = () => selectWorkerQuestion(worker);
        _liveWorkerSigs.set(wkey, sig);
      }
    });
  } else if (liveQuestion) {
    const wkey = 'live-single';
    activeLiveIds.add(wkey);
    const isActive = liveQuestion.question_id === activeQuestionId;
    const stepCount = liveQuestion.trajectory_steps ?? (liveQuestion.trajectory || []).length;
    const sig = `${liveQuestion.question_id}:${stepCount}:${isActive}`;

    let el = _liveWorkerNodes.get(wkey);
    if (!el) {
      el = document.createElement('div');
      _liveWorkerNodes.set(wkey, el);
      questionList.prepend(el);
    }

    if (_liveWorkerSigs.get(wkey) !== sig) {
      el.className = `question-item working ql-live-worker${isActive ? ' active' : ''}`;
      el.innerHTML = `
        <div class="question-meta">${liveQuestion.video_id || ''} • ${liveQuestion.question_id || ''}</div>
        <div class="question-title"><span class="pill working">Working...</span>${(liveQuestion.question || '').slice(0, 80)}${(liveQuestion.question || '').length > 80 ? '…' : ''}</div>
        <div class="question-meta">Steps: ${stepCount} | Exp: ${liveQuestion.expected || '—'}</div>
      `;
      el.onclick = () => selectLiveQuestion();
      _liveWorkerSigs.set(wkey, sig);
    }
  }

  // Remove live worker nodes that are no longer active
  for (const [wkey, el] of _liveWorkerNodes) {
    if (!activeLiveIds.has(wkey)) {
      el.remove();
      _liveWorkerNodes.delete(wkey);
      _liveWorkerSigs.delete(wkey);
    }
  }

  const hasLive = (liveWorkers && liveWorkers.length > 0) || liveQuestion;
  if (!questionsCache.length && !hasLive) {
    const msg = document.createElement('div');
    msg.className = 'ql-empty';
    msg.textContent = 'No questions processed yet.';
    questionList.appendChild(msg);
    return;
  }

  // Clear any stale text nodes (e.g. "Loading questions..." from initial HTML)
  for (const node of [...questionList.childNodes]) {
    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
      node.remove();
    }
  }

  // --- Completed questions section (incremental DOM updates) ---
  // Track which IDs are still present so we can prune removed ones
  const seenIds = new Set();

  questionsCache.forEach((q) => {
    const qid = String(q.question_id);
    seenIds.add(qid);
    const sig = _qListSig(q);
    const isActive = qid === String(activeQuestionId);
    const activeSig = `${sig}|${isActive}`;

    let el = _questionNodes.get(qid);
    if (!el) {
      // First time we've seen this question — create and append the node
      el = document.createElement('div');
      el.addEventListener('click', () => selectQuestion(q.question_id));
      _questionNodes.set(qid, el);
      questionList.appendChild(el);
    }

    const prevSig = _questionNodeSigs.get(qid);
    if (activeSig === prevSig) return; // Nothing changed — skip

    // Update class (cheap — always do this)
    const statusClass = q.correct === true ? 'correct' : q.correct === false ? 'wrong' : 'pending';
    el.className = `question-item ${statusClass}${isActive ? ' active' : ''}`;

    // Rebuild innerHTML only when content fields changed (not just active toggle)
    if (sig !== (prevSig || '').replace(/\|[^|]*$/, '')) {
      el.innerHTML = _buildQuestionItemHTML(q);
    }

    _questionNodeSigs.set(qid, activeSig);
  });

  // Prune nodes for questions that are no longer in the cache (edge case)
  for (const [qid, el] of _questionNodes) {
    if (!seenIds.has(qid)) {
      el.remove();
      _questionNodes.delete(qid);
      _questionNodeSigs.delete(qid);
      _questionCostCache.delete(qid);
    }
  }

  questionList.scrollTop = savedScroll;
}

async function selectWorkerQuestion(worker) {
  if (!worker) return;
  activeQuestionId = worker.question_id;
  _lastQuestionListFingerprint = '';  // Force re-render for active highlight
  _lastDetailFingerprint = '';  // Force fresh render on manual selection
  renderQuestionList();
  // Show lightweight data immediately, then fetch full detail with trajectory
  displayQuestionDetail({ ...worker, _loading: true });
  try {
    const wid = worker.worker_id ?? 0;
    const res = await fetch(`/api/live/workers/${wid}`);
    if (res.ok) {
      const full = await res.json();
      displayQuestionDetail(full);
    }
  } catch (e) { /* keep showing lightweight version */ }
}

async function selectLiveQuestion() {
  if (!liveQuestion) return;
  activeQuestionId = liveQuestion.question_id;
  _lastQuestionListFingerprint = '';  // Force re-render for active highlight
  _lastDetailFingerprint = '';  // Force fresh render on manual selection
  renderQuestionList();
  displayQuestionDetail({ ...liveQuestion, _loading: true });
  try {
    const res = await fetch('/api/live');
    if (res.ok) {
      const data = await res.json();
      if (data.active && data.question) displayQuestionDetail(data.question);
    }
  } catch (e) { /* keep showing lightweight version */ }
}

function _detailFingerprint(data) {
  const trajLen = (data.trajectory || []).length;
  return `${data.question_id}:${trajLen}:${data.status || ''}:${data.predicted || ''}`;
}

function displayQuestionDetail(data, preserveOpenState = false) {
  // Skip re-render if nothing changed (prevents scroll reset during polling)
  if (preserveOpenState) {
    const fp = _detailFingerprint(data);
    if (fp === _lastDetailFingerprint && displayedQuestionId === data.question_id) {
      return;  // No new steps, don't touch the DOM
    }
    _lastDetailFingerprint = fp;
  } else {
    _lastDetailFingerprint = _detailFingerprint(data);
  }

  displayedQuestionId = data.question_id;

  // Save scroll state — "sticky bottom" behavior:
  // If user was at the bottom, auto-scroll after update.
  // If user scrolled up to read, preserve their position.
  const scrollTop = detailBody.scrollTop;
  const scrollHeight = detailBody.scrollHeight;
  const clientHeight = detailBody.clientHeight;
  const wasAtBottom = (scrollHeight - scrollTop - clientHeight) < 50;

  // Capture currently open details before re-render
  const openDetailsSet = new Set();
  if (preserveOpenState) {
    detailBody.querySelectorAll('details[open]').forEach((el) => {
      const summary = el.querySelector('summary');
      if (summary) openDetailsSet.add(summary.textContent.trim());
    });
  }

  const qTokens = data.token_usage?.total?.total_tokens;
  const tokenSuffix = qTokens ? ` — ${fmtTokens(qTokens)} tokens` : '';
  detailHeader.textContent = `${data.question_id || ''} — ${data.video_id || ''}${data.status === 'working' ? ' (Working...)' : ''}${tokenSuffix}`;

  const steps = data.trajectory || [];
  const reasoning = data.reasoning || '';

  // Build preamble: full question + expected answer + options
  const preambleParts = [];
  if (data.question) {
    preambleParts.push(`<div class="detail-question">${escapeHtml(data.question)}</div>`);
  }
  // Show question image if present (e.g. tables, diagrams from VideoMMMU)
  if (data.question_image) {
    preambleParts.push(`<div class="detail-question-image"><img src="data:image/png;base64,${data.question_image}" alt="Question image" style="max-width:100%;max-height:400px;border-radius:6px;margin:8px 0;border:1px solid var(--card-border);" /></div>`);
  }
  // Show answer options if available (list of strings like "A. text" or "B. text")
  if (data.options && Array.isArray(data.options) && data.options.length) {
    const optHtml = data.options.map(opt => {
      const optStr = String(opt);
      const letter = optStr.match(/^([A-Z])/)?.[1] || '';
      const isCorrect = data.expected && letter === data.expected;
      const cls = isCorrect ? ' class="option-correct"' : '';
      return `<span${cls}>${escapeHtml(optStr)}</span>`;
    }).join('');
    preambleParts.push(`<div class="detail-options">${optHtml}</div>`);
  }
  const expectedLabel = data.expected || '—';
  const predictedLabel = data.predicted || null;
  let verdictHtml = `<span class="detail-expected">Expected: <strong>${escapeHtml(expectedLabel)}</strong></span>`;
  if (predictedLabel) {
    const isCorrect = data.correct === true;
    const cls = isCorrect ? 'detail-correct' : 'detail-wrong';
    verdictHtml += ` &nbsp;|&nbsp; <span class="${cls}">Predicted: <strong>${escapeHtml(predictedLabel)}</strong></span>`;
  }
  preambleParts.push(`<div class="detail-verdict">${verdictHtml}</div>`);

  // Show error banner if this question errored
  if (data.error) {
    preambleParts.push(`<div class="error-banner">${escapeHtml(data.error)}</div>`);
  }

  const renderMaybeJson = (label, value, extraClass = '', openDefault = false, plainSummary = false) => {
    if (!value) return '';
    let parsed = null;
    // Handle objects directly (not just strings)
    if (typeof value === 'object' && value !== null) {
      parsed = value;
    } else if (typeof value === 'string') {
      try { parsed = JSON.parse(value); } catch (e) { /* not json */ }
    }
    const summary = label || 'Details';
    let summaryText;
    if (parsed && (typeof parsed === 'object')) {
      const pretty = JSON.stringify(parsed, null, 2);
      const preview = Array.isArray(parsed) ? `array · ${parsed.length} items` : `object · ${Object.keys(parsed).length} keys`;
      summaryText = plainSummary ? summary : `${summary} — ${preview}`;
      // Check if this was previously open
      const wasOpen = openDetailsSet.has(summaryText) || openDefault;
      const openAttr = wasOpen ? ' open' : '';
      return `<div class="block ${extraClass}"><details class="json-block"${openAttr}><summary>${summaryText}</summary><pre>${pretty}</pre></details></div>`;
    }
    const text = typeof value === 'string' ? value : String(value);
    const previewText = text.length > 80 ? `${text.slice(0, 80)}…` : text;
    summaryText = plainSummary ? summary : `${summary} — ${previewText}`;
    // Check if this was previously open
    const wasOpen = openDetailsSet.has(summaryText) || openDefault;
    const openAttr = wasOpen ? ' open' : '';
    return `<div class="block ${extraClass}"><details class="json-block"${openAttr}><summary>${summaryText}</summary><pre>${text}</pre></details></div>`;
  };

  // --- Lightweight Markdown renderer for orchestrated memory ---
  const renderMarkdownMemory = (mdText) => {
    // Parse markdown sections (## headers) into structured HTML
    const lines = mdText.split('\n');
    const sections = [];
    let currentSection = null;
    let preambleLines = [];  // Lines before first ## header (question images, etc.)

    for (const line of lines) {
      const headerMatch = line.match(/^## (.+)/);
      if (headerMatch) {
        if (currentSection) sections.push(currentSection);
        currentSection = { title: headerMatch[1].trim(), lines: [] };
      } else if (currentSection) {
        currentSection.lines.push(line);
      } else {
        preambleLines.push(line);
      }
    }
    if (currentSection) sections.push(currentSection);

    if (sections.length === 0) {
      // Not sectioned markdown — render as simple pre
      return `<div class="block step-memory"><details class="json-block" open><summary>Memory State (orchestrated)</summary><pre>${escapeHtml(mdText)}</pre></details></div>`;
    }

    // Render each section using the existing mem-section styling
    const renderInlineMarkdown = (text) => {
      // Bold: **text** → <strong>
      let html = escapeHtml(text);
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      // Timestamps: [123s-456s] → chip
      html = html.replace(/\[(\d+s(?:-\d+s)?)\]/g, '<span class="mem-chip">$1</span>');
      // Iteration refs: [Iteration N] → tag
      html = html.replace(/\[Iteration (\d+[^[\]]*)\]/g, '<span class="mem-iter">iter $1</span>');
      // Legacy key frame tags: {{KEY_FRAME:/path}} → frame label (deprecated)
      html = html.replace(/\{\{KEY_FRAME:([^}]+)\}\}/g, '<span class="mem-chip" title="$1">📷 frame</span>');
      return html;
    };

    // Render embedded images: {{IMAGE:data:image/jpeg;base64,...}}
    const renderImages = (text) => {
      return text.replace(/\{\{IMAGE:(data:image\/[^}]+)\}\}/g,
        '<div class="mem-key-frame"><img src="$1" style="max-width:320px;max-height:240px;border-radius:6px;margin:4px 0;" /></div>');
    };

    const parts = sections.map(sec => {
      let content = sec.lines.join('\n').trim();
      if (!content) {
        return `<div class="mem-section"><div class="mem-section-title">${escapeHtml(sec.title)}</div><div class="mem-muted">(empty)</div></div>`;
      }

      // Extract and render embedded images before line-by-line processing
      // {{IMAGE:data:image/jpeg;base64,...}} → inline <img> tags
      content = content.replace(/\{\{IMAGE:(data:image\/[^}]+)\}\}/g, '{{IMG_PLACEHOLDER}}');
      const imageMatches = sec.lines.join('\n').match(/\{\{IMAGE:(data:image\/[^}]+)\}\}/g) || [];
      let imageIdx = 0;

      // Render content lines with basic formatting
      const contentLines = content.split('\n');
      let html = '';
      let inList = false;

      for (const line of contentLines) {
        const trimmed = line.trim();
        if (!trimmed) {
          if (inList) { html += '</div>'; inList = false; }
          continue;
        }

        // Numbered list items: 1. **[timestamp]**: ...
        const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        // Bullet list items: - **text**: ...
        const bulletMatch = !numberedMatch && trimmed.match(/^[-*]\s+(.*)/);
        // Indented sub-items: - **text**: ... (with leading spaces)
        const indentedMatch = !numberedMatch && !bulletMatch && line.match(/^\s{2,}[-*]\s+(.*)/);

        // Embedded image placeholder
        if (trimmed === '{{IMG_PLACEHOLDER}}') {
          if (inList) { html += '</div>'; inList = false; }
          if (imageIdx < imageMatches.length) {
            const dataUri = imageMatches[imageIdx].replace(/^\{\{IMAGE:/, '').replace(/\}\}$/, '');
            html += `<div class="mem-key-frame" style="margin:6px 0;"><img src="${dataUri}" style="max-width:320px;max-height:240px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.2);" /></div>`;
            imageIdx++;
          }
          continue;
        }

        if (numberedMatch || bulletMatch || indentedMatch) {
          if (!inList) { html += '<div class="mem-entries">'; inList = true; }
          const text = numberedMatch ? numberedMatch[2] : (bulletMatch ? bulletMatch[1] : indentedMatch[1]);
          const indent = indentedMatch ? ' style="margin-left:16px"' : '';
          html += `<div class="mem-entry"${indent}><div class="mem-entry-header">${renderInlineMarkdown(text)}</div></div>`;
        } else {
          if (inList) { html += '</div>'; inList = false; }
          html += `<div class="mem-finding">${renderInlineMarkdown(trimmed)}</div>`;
        }
      }
      if (inList) html += '</div>';

      // Count items for badge
      const itemCount = contentLines.filter(l => l.trim().match(/^(\d+\.|-|\*)\s/)).length;
      const badge = itemCount > 0 ? ` <span class="mem-count">${itemCount}</span>` : '';

      return `<div class="mem-section"><div class="mem-section-title">${escapeHtml(sec.title)}${badge}</div>${html}</div>`;
    });

    // Render preamble (question images, etc.) before sections
    let preambleHtml = '';
    if (preambleLines.length > 0) {
      const preambleText = preambleLines.join('\n').trim();
      if (preambleText) {
        // Render embedded images from preamble
        const imageMatches = preambleText.match(/\{\{IMAGE:(data:image\/[^}]+)\}\}/g) || [];
        let preambleContent = preambleText.replace(/\{\{IMAGE:(data:image\/[^}]+)\}\}/g, '{{IMG_PLACEHOLDER}}');
        let imgIdx = 0;
        let pHtml = '';
        for (const line of preambleContent.split('\n')) {
          const trimmed = line.trim();
          if (trimmed === '{{IMG_PLACEHOLDER}}' && imgIdx < imageMatches.length) {
            const dataUri = imageMatches[imgIdx].replace(/^\{\{IMAGE:/, '').replace(/\}\}$/, '');
            pHtml += `<div class="mem-key-frame" style="margin:6px 0;"><img src="${dataUri}" style="max-width:400px;max-height:300px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.2);" /></div>`;
            imgIdx++;
          } else if (trimmed) {
            pHtml += `<div class="mem-line">${escapeHtml(trimmed)}</div>`;
          }
        }
        if (pHtml) {
          preambleHtml = `<div class="mem-section"><div class="mem-section-title">Question Image</div>${pHtml}</div>`;
        }
      }
    }

    const label = 'Memory State (orchestrated)';
    const wasOpen = openDetailsSet.has(label) || true;
    const openAttr = wasOpen ? ' open' : '';
    return `<div class="block step-memory"><details class="json-block"${openAttr}><summary>${label}</summary><div class="mem-container">${preambleHtml}${parts.join('')}</div></details></div>`;
  };

  // --- Custom rich renderer for <MEMORY> JSON blocks ---
  // Handles regular in-memory format, file-backed memory format, and orchestrated markdown
  const renderMemoryState = (raw) => {
    // Strip <MEMORY> tags
    let content = typeof raw === 'string' ? raw : String(raw);
    content = content.replace(/<\/?MEMORY>/g, '').trim();

    // Extract <TRANSCRIPT> block — render separately as collapsed
    let transcriptHtml = '';
    const transcriptMatch = content.match(/<TRANSCRIPT>([\s\S]*?)<\/TRANSCRIPT>/);
    if (transcriptMatch) {
      content = content.replace(/<TRANSCRIPT>[\s\S]*?<\/TRANSCRIPT>/, '').trim();
      const lines = transcriptMatch[1].trim().split('\n').length;
      transcriptHtml = `<div class="block" style="margin-top:8px;"><details class="json-block"><summary>Transcript (${lines} lines)</summary><pre style="max-height:300px;overflow-y:auto;font-size:11px;white-space:pre-wrap;">${escapeHtml(transcriptMatch[1].trim())}</pre></details></div>`;
    }

    // Detect orchestrated markdown memory (starts with ## or contains ## headers)
    if (content.startsWith('## ') || /\n## /.test(content)) {
      return renderMarkdownMemory(content) + transcriptHtml;
    }

    // Try JSON parse for legacy formats
    let mem;
    try {
      mem = typeof content === 'object' ? content : JSON.parse(content);
    } catch (e) {
      // Unknown format — raw display
      return `<div class="block step-memory"><details class="json-block" open><summary>Memory State</summary><pre>${escapeHtml(raw)}</pre></details></div>`;
    }

    // Detect format: file-backed has by_level in coverage or work_ledger/pinned_entries/focal_entries
    const isFileBacked = !!(mem.coverage?.by_level || mem.work_ledger || mem.global_results || mem.pinned_entries || mem.focal_entries || mem.regions);

    const parts = [];

    // Video info bar — handle both formats
    if (mem.video) {
      const v = mem.video;
      // Regular format: {duration_s, resolution}
      // File-backed format: {duration_seconds, size_mb, format, video: {codec, width, height, fps}, audio: {...}}
      const dur = v.duration_s ?? v.duration_seconds ?? '?';
      const durStr = typeof dur === 'number' ? `${Math.round(dur)}s` : dur;
      let res = v.resolution || '';
      if (!res && v.video) res = `${v.video.width}x${v.video.height}`;
      const extras = [];
      if (v.size_mb) extras.push(`${v.size_mb}MB`);
      if (v.video?.codec) extras.push(v.video.codec);
      if (v.video?.fps) extras.push(`${v.video.fps}fps`);
      const extrasStr = extras.length ? ` &middot; ${extras.join(' &middot; ')}` : '';
      parts.push(`<div class="mem-bar"><span class="mem-bar-label">Video</span><span>${durStr} &middot; ${res || '?'}${extrasStr}</span></div>`);
    }

    // Scenes (regular format only)
    if (mem.scenes) {
      const sc = mem.scenes;
      const items = (sc.items || []).map(s => `<span class="mem-chip">${s.id}: ${s.start_s}s&ndash;${s.end_s}s</span>`).join('');
      const trunc = sc.truncated ? ` <span class="mem-muted">+${sc.truncated} more</span>` : '';
      parts.push(`<div class="mem-section"><div class="mem-section-title">Scenes <span class="mem-count">${sc.count}</span></div><div class="mem-chips">${items}${trunc}</div></div>`);
    }

    // Coverage — handle both formats
    if (mem.coverage) {
      const c = mem.coverage;
      if (isFileBacked) {
        // File-backed: {total_entries, by_level: {L0, L1, L2, L3}, time_range: [start, end]}
        const range = c.time_range ? `${Math.round(c.time_range[0])}s &ndash; ${Math.round(c.time_range[1])}s` : 'none';
        const levels = c.by_level || {};
        const levelLabels = { L0: 'Global', L1: 'Pinned', L2: 'Temporal', L3: 'Frame' };
        const levelChips = Object.entries(levels)
          .filter(([, v]) => v > 0)
          .map(([k, v]) => `<span class="mem-chip">${levelLabels[k] || k}: ${v}</span>`)
          .join('');
        parts.push(`<div class="mem-bar"><span class="mem-bar-label">Coverage</span><span>${c.total_entries || 0} entries &middot; ${range}</span></div>`);
        if (levelChips) {
          parts.push(`<div class="mem-bar"><span class="mem-bar-label">Levels</span><div class="mem-chips" style="display:inline">${levelChips}</div></div>`);
        }
      } else {
        // Regular: {num_timestamps, range_s: [start, end]}
        const range = c.range_s ? `${c.range_s[0]}s &ndash; ${c.range_s[1]}s` : 'none';
        parts.push(`<div class="mem-bar"><span class="mem-bar-label">Coverage</span><span>${c.num_timestamps} timestamps &middot; ${range}</span></div>`);
      }
    }

    // Focal point (file-backed only)
    if (mem.focal_point) {
      const fp = mem.focal_point;
      const ts = fp.timestamp != null ? `${Math.round(fp.timestamp)}s` : '?';
      const intent = fp.intent ? ` &mdash; ${escapeHtml(fp.intent)}` : '';
      parts.push(`<div class="mem-bar"><span class="mem-bar-label">Focal Point</span><span>${ts}${intent}</span></div>`);
    }

    // Sensory (regular format)
    if (mem.sensory && mem.sensory.length) {
      const rows = mem.sensory.map(s => {
        const resClass = s.resolution === 'coarse' ? 'mem-tag-coarse' : 'mem-tag-fine';
        const interval = s.interval ? `${s.interval.start_s}s&ndash;${s.interval.end_s}s` : '';
        const frames = s.num_frames ? `${s.num_frames}f` : '';
        const summary = s.visual_summary ? `<div class="mem-summary">${escapeHtml(s.visual_summary)}</div>` : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag ${resClass}">${s.resolution}</span>` +
          `<span class="mem-iter">iter ${s.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(s.tool)}</span>` +
          `<span class="mem-muted">${interval} &middot; ${frames}</span>` +
          `</div>${summary}</div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Sensory <span class="mem-count">${mem.sensory.length}</span></div>${rows}</div>`);
    }

    // Regions — legacy backward compat for archived runs
    if (mem.regions && mem.regions.length) {
      const rows = mem.regions.map(r => {
        const interval = r.interval ? `${Math.round(r.interval[0])}s&ndash;${Math.round(r.interval[1])}s` : '';
        const summary = r.summary ? `<div class="mem-summary">${escapeHtml(r.summary)}</div>` : '';
        const facts = (r.key_facts && r.key_facts.length)
          ? `<div class="mem-facts">${r.key_facts.map(f => `<span class="mem-chip">${escapeHtml(f)}</span>`).join('')}</div>`
          : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">L1 region</span>` +
          `<span class="mem-iter">iter ${r.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(r.tool || '')}</span>` +
          (interval ? `<span class="mem-muted">${interval}</span>` : '') +
          `</div>${summary}${facts}</div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Regions <span class="mem-count">${mem.regions.length}</span></div>${rows}</div>`);
    }

    // Focal entries — L1 temporal + L2 frame (handles both new string and old numeric level tags)
    if (mem.focal_entries && mem.focal_entries.length) {
      const rows = mem.focal_entries.map(f => {
        const isFrame = f.level === 'L2' || f.level === 3;
        const levelTag = isFrame ? 'L2 frame' : 'L1 temporal';
        const tagClass = isFrame ? 'mem-tag-fine' : 'mem-tag-coarse';
        const interval = f.interval ? `${Math.round(f.interval[0])}s&ndash;${Math.round(f.interval[1])}s` : '';
        const ts = f.timestamp != null ? `@${Math.round(f.timestamp)}s` : '';
        const summary = f.summary ? `<div class="mem-summary">${escapeHtml(f.summary)}</div>` : '';
        const facts = (f.key_facts && f.key_facts.length)
          ? `<div class="mem-facts">${f.key_facts.map(kf => `<span class="mem-chip">${escapeHtml(kf)}</span>`).join('')}</div>`
          : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag ${tagClass}">${levelTag}</span>` +
          `<span class="mem-iter">iter ${f.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(f.tool || '')}</span>` +
          `<span class="mem-muted">${interval} ${ts}</span>` +
          `</div>${summary}${facts}</div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Focal Entries <span class="mem-count">${mem.focal_entries.length}</span></div>${rows}</div>`);
    }

    // Work Ledger (new format) — structured index of agent work
    if (mem.work_ledger) {
      const wl = mem.work_ledger;
      let rows = '';

      // Frame samples
      if (wl.frame_samples && wl.frame_samples.length) {
        rows += wl.frame_samples.map(fs => {
          const interval = fs.interval ? `${fs.interval[0]}s\u2013${fs.interval[1]}s` : '';
          return `<div class="mem-entry"><div class="mem-entry-header">` +
            `<span class="mem-tag mem-tag-coarse">Frames</span>` +
            `<span class="mem-iter">iter ${fs.iteration}</span>` +
            `<span class="mem-muted">${interval} &middot; step ${fs.step}s &middot; ${fs.count} frames</span>` +
            `</div></div>`;
        }).join('');
      }

      // Analyses
      if (wl.analyses && wl.analyses.length) {
        rows += wl.analyses.map(a => {
          const interval = a.interval ? `${Math.round(a.interval[0])}s\u2013${Math.round(a.interval[1])}s` : '';
          const ts = a.timestamp != null ? `@${Math.round(a.timestamp)}s` : '';
          const frames = a.frames ? ` &middot; ${a.frames}f` : '';
          return `<div class="mem-entry"><div class="mem-entry-header">` +
            `<span class="mem-tag mem-tag-fine">Analysis</span>` +
            `<span class="mem-iter">iter ${a.iteration}</span>` +
            `<span class="mem-tool">${escapeHtml(a.tool || '')}</span>` +
            `<span class="mem-muted">${interval} ${ts}${frames}</span>` +
            `<span class="mem-chip">${escapeHtml(a.id || '')}</span>` +
            `</div></div>`;
        }).join('');
      }

      // Transcript
      if (wl.transcript) {
        rows += `<div class="mem-entry"><div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">Transcript</span>` +
          `<span class="mem-muted">${wl.transcript.status} &middot; ${wl.transcript.segments} segments &middot; ${Math.round(wl.transcript.duration)}s</span>` +
          `</div></div>`;
      }

      // Files created
      if (wl.files_created && wl.files_created.length) {
        rows += wl.files_created.map(f => {
          return `<div class="mem-entry"><div class="mem-entry-header">` +
            `<span class="mem-tag mem-tag-coarse">File</span>` +
            `<span class="mem-iter">iter ${f.iteration}</span>` +
            `<span class="mem-muted">${escapeHtml(f.path || '')}</span>` +
            `</div></div>`;
        }).join('');
      }

      const totalItems = (wl.frame_samples?.length || 0) + (wl.analyses?.length || 0)
        + (wl.transcript ? 1 : 0) + (wl.files_created?.length || 0);
      parts.push(`<div class="mem-section"><div class="mem-section-title">Work Ledger <span class="mem-count">${totalItems}</span></div>${rows}</div>`);
    }

    // Global results (legacy format — backward compat for archived runs)
    if (!mem.work_ledger && mem.global_results && mem.global_results.length) {
      const rows = mem.global_results.map(r => {
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">L0</span>` +
          `<span class="mem-iter">iter ${r.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(r.tool || '')}</span>` +
          `</div><div class="mem-finding">${escapeHtml(r.finding || '')}</div></div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Global Results <span class="mem-count">${mem.global_results.length}</span></div>${rows}</div>`);
    }

    // Narrative thread (file-backed, L0 level)
    if (mem.narrative && mem.narrative.length) {
      const rows = mem.narrative.map(n => {
        const interval = (n.interval && n.interval[0] != null) ? `${n.interval[0]}s\u2013${n.interval[1]}s` : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">L0</span>` +
          `<span class="mem-iter">iter ${n.t}</span>` +
          (interval ? `<span class="mem-chip-focus">${interval}</span>` : '') +
          `</div><div class="mem-facts">${escapeHtml(n.note || '')}</div></div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Narrative Thread <span class="mem-count">${mem.narrative.length}</span></div>${rows}</div>`);
    }

    // Pinned entries (L0_PINNED — always loaded important findings)
    if (mem.pinned_entries && mem.pinned_entries.length) {
      const rows = mem.pinned_entries.map(p => {
        const interval = p.interval ? `${Math.round(p.interval[0])}s\u2013${Math.round(p.interval[1])}s` : '';
        const summary = p.summary ? `<div class="mem-summary">${escapeHtml(p.summary)}</div>` : '';
        const facts = (p.key_facts && p.key_facts.length)
          ? `<div class="mem-facts">${p.key_facts.map(f => `<span class="mem-chip">${escapeHtml(f)}</span>`).join('')}</div>`
          : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">Pinned</span>` +
          `<span class="mem-iter">iter ${p.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(p.tool || '')}</span>` +
          (interval ? `<span class="mem-muted">${interval}</span>` : '') +
          `</div>${summary}${facts}</div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Pinned Entries <span class="mem-count">${mem.pinned_entries.length}</span></div>${rows}</div>`);
    }

    // Results (regular format)
    if (mem.results && mem.results.length) {
      const rows = mem.results.map(r => {
        const interval = r.interval ? `${r.interval.start_s}s&ndash;${r.interval.end_s}s` : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-iter">iter ${r.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(r.tool)}</span>` +
          (interval ? `<span class="mem-muted">${interval}</span>` : '') +
          `</div><div class="mem-finding">${escapeHtml(r.finding || '')}</div></div>`;
      }).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Results <span class="mem-count">${mem.results.length}</span></div>${rows}</div>`);
    }

    // Reasoning trace — handle flat array, structured {summary, recent}, and legacy formats
    if (mem.reasoning_trace) {
      const renderTraceEntry = (t) => {
        const tool = t.tool || t.tool_chosen || '';
        const args = t.args || t.tool_args_summary || '';
        const reasoning = t.reasoning || t.thought || '';
        const reasoningHtml = reasoning ? `<div class="mem-reasoning">${escapeHtml(reasoning)}</div>` : '';
        return `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-iter">iter ${t.iteration}</span>` +
          `<span class="mem-tool">${escapeHtml(tool)}</span>` +
          `<span class="mem-muted">${escapeHtml(args)}</span>` +
          `</div>${reasoningHtml}</div>`;
      };

      let rows = '';
      let count = 0;

      if (Array.isArray(mem.reasoning_trace)) {
        // Flat array (short runs or legacy)
        rows = mem.reasoning_trace.map(renderTraceEntry).join('');
        count = mem.reasoning_trace.length;
      } else if (mem.reasoning_trace.summary) {
        // Structured: summary + recent entries
        const sc = mem.reasoning_trace.summary_covers;
        const coversLabel = sc ? `steps ${sc[0]}\u2013${sc[1]}` : '';
        rows += `<div class="mem-entry">` +
          `<div class="mem-entry-header">` +
          `<span class="mem-tag mem-tag-coarse">Summary</span>` +
          `<span class="mem-muted">${coversLabel}</span>` +
          `</div><div class="mem-finding">${escapeHtml(mem.reasoning_trace.summary)}</div></div>`;
        if (mem.reasoning_trace.recent && mem.reasoning_trace.recent.length) {
          rows += mem.reasoning_trace.recent.map(renderTraceEntry).join('');
        }
        count = (mem.reasoning_trace.recent ? mem.reasoning_trace.recent.length : 0) + 1;
      }

      if (rows) {
        parts.push(`<div class="mem-section"><div class="mem-section-title">Reasoning Trace <span class="mem-count">${count}</span></div>${rows}</div>`);
      }
    }

    // Focus
    if (mem.focus && mem.focus.length) {
      const chips = mem.focus.map(f => `<span class="mem-chip mem-chip-focus">${f.start_s}s&ndash;${f.end_s}s</span>`).join('');
      parts.push(`<div class="mem-section"><div class="mem-section-title">Focus</div><div class="mem-chips">${chips}</div></div>`);
    }

    // Count Ledger (counting skill)
    if (mem.count_ledger) {
      const cl = mem.count_ledger;
      const count = cl.count != null ? cl.count : '?';
      const notes = cl.notes || '';
      const notesHtml = notes ? `<pre class="count-ledger-notes">${escapeHtml(notes)}</pre>` : '';
      parts.push(
        `<div class="mem-section mem-section-count-ledger">` +
        `<div class="mem-section-title">Count Ledger <span class="count-ledger-count">${count}</span></div>` +
        notesHtml +
        `</div>`
      );
    }

    // Visual Planner Todo List
    if (mem.todo_list && mem.todo_list.length) {
      const todos = mem.todo_list;
      const done = todos.filter(t => t.done).length;
      const total = todos.length;
      const progressPct = total > 0 ? Math.round((done / total) * 100) : 0;
      const badgeClass = done === total ? 'todo-badge-done' : 'todo-badge-pending';
      const rows = todos.map(t => {
        const statusIcon = t.done ? '✅' : (t.feedback ? '🔄' : '⬜');
        const itemClass = t.done ? 'todo-item todo-item-done' : (t.feedback ? 'todo-item todo-item-retry' : 'todo-item');
        const tsHtml = t.timestamp_hint ? `<span class="todo-ts">${escapeHtml(t.timestamp_hint)}</span>` : '';
        let detailHtml = '';
        if (t.done && t.finding) {
          detailHtml = `<div class="todo-finding">↳ ${escapeHtml(t.finding)}</div>`;
        } else if (!t.done && t.feedback) {
          detailHtml = `<div class="todo-feedback">✗ ${escapeHtml(t.feedback)}</div>`;
          if (t.last_attempt) {
            detailHtml += `<div class="todo-attempt">tried: ${escapeHtml(t.last_attempt)}</div>`;
          }
        }
        return `<div class="${itemClass}">` +
          `<div class="todo-header"><span class="todo-status">${statusIcon}</span><span class="todo-id">#${t.id}</span>${tsHtml}<span class="todo-task">${escapeHtml(t.task)}</span></div>` +
          detailHtml +
          `</div>`;
      }).join('');
      const progressBar = `<div class="todo-progress-bar"><div class="todo-progress-fill" style="width:${progressPct}%"></div></div>`;
      parts.push(
        `<div class="mem-section mem-section-todo">` +
        `<div class="mem-section-title">Visual Todos <span class="todo-badge ${badgeClass}">${done}/${total}</span></div>` +
        progressBar +
        `<div class="todo-list">${rows}</div>` +
        `</div>`
      );
    }

    // Conversation state (no-memory / LLM-in-Sandbox mode)
    if (mem.conversation) {
      const c = mem.conversation;
      const roles = c.by_role || {};
      const roleChips = Object.entries(roles)
        .map(([role, count]) => `<span class="mem-chip">${role}: ${count}</span>`)
        .join('');
      parts.push(
        `<div class="mem-bar"><span class="mem-bar-label">Messages</span><span>${c.total_messages || 0} total</span></div>` +
        `<div class="mem-bar"><span class="mem-bar-label">Roles</span><div class="mem-chips" style="display:inline">${roleChips}</div></div>` +
        `<div class="mem-bar"><span class="mem-bar-label">Context</span><span>${(c.total_chars || 0).toLocaleString()} chars &middot; ~${(c.est_tokens || 0).toLocaleString()} tokens</span></div>`
      );
    }

    // Mode label (no-memory mode)
    if (mem.mode) {
      parts.push(`<div class="mem-bar"><span class="mem-bar-label">Mode</span><span class="mem-muted">${escapeHtml(mem.mode)}</span></div>`);
    }

    // Stats (for full renders)
    if (mem.stats) {
      parts.push(`<div class="mem-bar"><span class="mem-bar-label">Stats</span><span class="mem-muted">${escapeHtml(mem.stats)}</span></div>`);
    }

    // Wrap in collapsible details (open by default)
    const isNoMemory = !!mem.conversation;
    const label = isFileBacked ? 'Memory State (file-backed)' : (isNoMemory ? 'Conversation State (no memory)' : 'Memory State');
    const wasOpen = openDetailsSet.has('Memory State') || openDetailsSet.has(label) || true;
    const openAttr = wasOpen ? ' open' : '';
    return `<div class="block step-memory"><details class="json-block"${openAttr}><summary>${label}</summary><div class="mem-container">${parts.join('')}</div></details></div>`;
  };

  const parseContent = (raw) => {
    if (!raw) return { raw, parsed: null };
    if (typeof raw === 'string') {
      try { return { raw, parsed: JSON.parse(raw) }; } catch (e) { return { raw, parsed: null }; }
    }
    return { raw: raw.toString(), parsed: raw };
  };

  const parts = [];
  // Question preamble (question text, expected, predicted)
  if (preambleParts.length) {
    parts.push(`<div class="detail-preamble">${preambleParts.join('')}</div>`);
  }
  // Token usage summary for this question
  if (data.token_usage && data.token_usage.by_agent && Object.keys(data.token_usage.by_agent).length > 0) {
    const byAgent = data.token_usage.by_agent;
    const totalTok = data.token_usage.total || {};
    const agentRows = Object.entries(byAgent)
      .sort((a, b) => (b[1].total_tokens || 0) - (a[1].total_tokens || 0))
      .map(([agent, c]) => `<div class="table-row"><span>${escapeHtml(agent)}</span><span>${fmtTokens(c.input_tokens)}</span><span>${fmtTokens(c.output_tokens)}</span><span>${fmtTokens(c.total_tokens)}</span></div>`)
      .join('');
    const agentHeader = '<div class="table-row table-head"><span>Agent</span><span>Input</span><span>Output</span><span>Total</span></div>';
    const agentTotalRow = `<div class="table-row" style="font-weight:600;border-top:1px solid var(--card-border)"><span>Total</span><span>${fmtTokens(totalTok.input_tokens)}</span><span>${fmtTokens(totalTok.output_tokens)}</span><span>${fmtTokens(totalTok.total_tokens)}</span></div>`;
    const callCount = data.token_usage.call_count || 0;

    // Per-iteration breakdown
    const byIter = data.token_usage.by_iteration || {};
    let iterSection = '';
    const iterKeys = Object.keys(byIter).sort((a, b) => Number(a) - Number(b));
    if (iterKeys.length > 0) {
      const iterHeader = '<div class="table-row table-head"><span>Iteration</span><span>Input</span><span>Output</span><span>Total</span></div>';
      const iterRows = iterKeys.map(iterKey => {
        const agents = byIter[iterKey];
        let inp = 0, out = 0;
        Object.values(agents).forEach(c => { inp += c.input_tokens || 0; out += c.output_tokens || 0; });
        const tot = inp + out;
        // Build agent breakdown tooltip
        const agentParts = Object.entries(agents)
          .sort((a, b) => (b[1].total_tokens || 0) - (a[1].total_tokens || 0))
          .map(([a, c]) => `${a}: ${fmtTokens(c.total_tokens)}`)
          .join(', ');
        return `<div class="table-row" title="${escapeHtml(agentParts)}"><span>Iter ${iterKey}</span><span>${fmtTokens(inp)}</span><span>${fmtTokens(out)}</span><span>${fmtTokens(tot)}</span></div>`;
      }).join('');
      iterSection = `<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--card-border)"><div style="font-weight:600;font-size:12px;margin-bottom:6px;opacity:0.7">Per Iteration (hover for agent breakdown)</div><div class="mini-table">${iterHeader}${iterRows}</div></div>`;
    }

    const qCost = calcTotalCostByAgent(byAgent);
    const costText = qCost.total > 0 ? ` — ${fmtCost(qCost.total)}` : '';
    const summaryLabel = `Token Usage — ${fmtTokens(totalTok.total_tokens)} total (${callCount} calls)${costText}`;
    const wasOpen = openDetailsSet.has(summaryLabel);
    const openAttr = wasOpen ? ' open' : '';
    parts.push(`<div class="block step-generic"><details class="json-block"${openAttr}><summary>${summaryLabel}</summary><div class="mini-table">${agentHeader}${agentRows}${agentTotalRow}</div>${iterSection}</details></div>`);
  }
  if (reasoning) {
    parts.push(renderMaybeJson('Reasoning', reasoning, 'step-thought', true, true));
  }
  if (steps.length) {
    const groups = [];
    let currentIteration = null;
    let pendingThought = null;

    steps.forEach((s, idx) => {
      const st = (s.step_type || '').toLowerCase();
      const { parsed } = parseContent(s.content);
      
      // Handle iteration markers - start a new group
      if (st === 'iteration') {
        // Save any pending group
        if (currentIteration && currentIteration.items.length) {
          groups.push(currentIteration);
        }
        const iterNum = parsed?.number || (groups.length + 1);
        const iterMax = parsed?.max || '?';
        currentIteration = { title: `Iteration ${iterNum}/${iterMax}`, items: [] };
        // Add any pending thought to the new iteration
        if (pendingThought) {
          currentIteration.items.push(pendingThought);
          pendingThought = null;
        }
        return; // Don't render the iteration marker itself as a step
      }
      
      const tool = parsed && typeof parsed === 'object' && (parsed.tool || parsed.name || parsed.tool_name || (parsed.tool_call && parsed.tool_call.name));
      const toolLabel = tool ? ` · tool: ${tool}` : '';
      const stepHeader = `Step ${idx + 1}${s.step_type ? ` [${s.step_type}]` : ''}${toolLabel}`;

      // Render keep_count tool calls as compact inline cards
      if (tool === 'keep_count' && st.includes('tool')) {
        const input = parsed.input || parsed.arguments || {};
        const count = input.count != null ? input.count : '?';
        const notes = input.notes || '';
        const preview = notes.length > 120 ? notes.slice(0, 120) + '…' : notes;
        const card = `<div class="step-card step-keep-count">` +
          `<span class="keep-count-badge">${count}</span>` +
          `<span class="keep-count-label">keep_count</span>` +
          (preview ? `<span class="keep-count-notes">${escapeHtml(preview)}</span>` : '') +
          `</div>`;
        if (!currentIteration) currentIteration = { title: 'Iteration 1', items: [] };
        currentIteration.items.push(card);
        return;
      }
      // Render keep_count tool results as compact confirmation
      if (st.includes('result') && typeof s.content === 'string' && s.content.startsWith('Count ledger updated:')) {
        const match = s.content.match(/^Count ledger updated: (\d+)/);
        const count = match ? match[1] : '?';
        const card = `<div class="step-card step-keep-count-result">` +
          `<span class="keep-count-badge keep-count-badge-ok">${count}</span>` +
          `<span class="keep-count-label">ledger saved</span>` +
          `</div>`;
        if (!currentIteration) currentIteration = { title: 'Iteration 1', items: [] };
        currentIteration.items.push(card);
        return;
      }

      let stepClass = 'step-generic';
      if (st === 'skill_selection') {
          // Render skill selection as a highlighted banner before iterations
          const skills = parsed?.skills || [];
          const tools = parsed?.tools || [];
          const label = skills.length
            ? `Skills activated: <strong>${skills.join(', ')}</strong> (tools: ${tools.join(', ')})`
            : 'No skills activated';
          const banner = `<div class="step-card step-skill-selection"><div class="step-header">Skill Selection</div><div class="step-body">${label}</div></div>`;
          if (!currentIteration) {
            currentIteration = { title: 'Setup', items: [] };
          }
          currentIteration.items.push(banner);
          return;
      } else if (st === 'memory') {
          stepClass = 'step-memory';
      } else if (st.includes('result') || st.includes('output')) {
          stepClass = 'step-tool-result';
      } else if (st.includes('answer')) {
          stepClass = 'step-answer';
      } else if (st.includes('tool') || st.includes('action') || st.includes('call')) {
          stepClass = 'step-tool-use';
      } else if (st.includes('vision') || st.includes('frame')) {
          stepClass = 'step-vision';
      } else if (st.includes('transcrib') || st.includes('audio')) {
          stepClass = 'step-audio';
      } else if (st.includes('plan') || st.includes('thought') || st.includes('analysis')) {
          stepClass = 'step-thought';
      }

      const isThought = stepClass === 'step-thought';
      const isMemory = st === 'memory';
      const isAnswer = stepClass === 'step-answer' || st.includes('answer');
      const openDefault = isThought || isMemory;
      // Tool results with visual attachments get a custom renderer
      const isVisualResult = parsed && parsed.attachments && Array.isArray(parsed.attachments);
      let content;
      if (isMemory) {
        content = renderMemoryState(s.content || '');
      } else if (isVisualResult) {
        const attachBadges = parsed.attachments.map(a => {
          if (a.type === 'video') return `<span class="badge badge-video">video${a.fps ? ` ${a.fps}fps` : ''}</span> <code class="oss-url">${escapeHtml(a.url || '')}</code>`;
          if (a.type === 'image') return '<span class="badge badge-image">image</span>';
          return `<span class="badge">${a.type}</span>`;
        }).join(' ');
        const textContent = parsed.text || '';
        const previewText = textContent.length > 80 ? `${textContent.slice(0, 80)}…` : textContent;
        const summaryText = `${stepHeader} — ${previewText}`;
        const wasOpen = openDetailsSet.has(summaryText) || openDefault;
        const openAttr = wasOpen ? ' open' : '';
        content = `<div class="block ${stepClass}"><details class="json-block"${openAttr}><summary>${summaryText}</summary><div class="visual-attachments">${attachBadges}</div><pre>${escapeHtml(textContent)}</pre></details></div>`;
      } else {
        content = renderMaybeJson(stepHeader, s.content || '', stepClass, openDefault, isThought);
      }

      if (isThought) {
        // Hold thoughts to attach to next iteration or current group
        pendingThought = content;
      } else if (isAnswer) {
        // Final answer gets its own group
        if (currentIteration && currentIteration.items.length) {
          groups.push(currentIteration);
        }
        currentIteration = { title: 'Final Answer', items: [] };
        if (pendingThought) {
          currentIteration.items.push(pendingThought);
          pendingThought = null;
        }
        currentIteration.items.push(content);
      } else {
        // Ensure we have a current iteration group
        if (!currentIteration) {
          currentIteration = { title: 'Iteration 1', items: [] };
        }
        // Add any pending thought first
        if (pendingThought) {
          currentIteration.items.push(pendingThought);
          pendingThought = null;
        }
        currentIteration.items.push(content);
      }
    });

    // Don't forget the last group
    if (currentIteration && currentIteration.items.length) {
      groups.push(currentIteration);
    }
    // And any trailing thought
    if (pendingThought && groups.length) {
      groups[groups.length - 1].items.push(pendingThought);
    }

    // --- Performance: lazy trajectory rendering ---
    // For long trajectories, render only the first few iteration groups
    // and add a "Show more" button for the rest. If the user already
    // clicked "Show more" for this question, render everything — otherwise
    // poll-driven re-renders would collapse the trajectory back to the
    // initial 5 groups, dropping the user's scroll position into stale DOM.
    const MAX_INITIAL_GROUPS = 5;
    const alreadyExpanded = data.question_id != null && _expandedQuestionIds.has(data.question_id);
    const hasMore = groups.length > MAX_INITIAL_GROUPS && !alreadyExpanded;
    const initialGroups = hasMore ? groups.slice(0, MAX_INITIAL_GROUPS) : groups;
    const remainingGroups = hasMore ? groups.slice(MAX_INITIAL_GROUPS) : [];

    initialGroups.forEach((g) => {
      parts.push(`<div class="iteration-divider">${escapeHtml(g.title)}</div>`);
      parts.push(g.items.join('\n'));
    });

    if (hasMore) {
      parts.push(`<div class="show-more-steps" id="showMoreSteps">Show ${remainingGroups.length} more iteration${remainingGroups.length > 1 ? 's' : ''}...</div>`);
      // Store for post-render handler attachment
      _pendingRemainingGroups = remainingGroups;
    }
  } else {
    if (data._loading) {
      parts.push('<div class="block" style="color:var(--accent);font-weight:600;">Loading trajectory...</div>');
    } else {
      parts.push('<div class="block">No trajectory captured yet.</div>');
    }
  }
  detailBody.innerHTML = parts.join('\n');

  // Attach show-more handler if needed
  if (_pendingRemainingGroups) {
    const showMoreBtn = detailBody.querySelector('#showMoreSteps');
    const remaining = _pendingRemainingGroups;
    _pendingRemainingGroups = null;
    if (showMoreBtn && remaining.length) {
      showMoreBtn.addEventListener('click', () => {
        // Persist expansion so subsequent polls render everything inline
        // (instead of collapsing back to the initial 5 groups).
        if (data && data.question_id != null) {
          _expandedQuestionIds.add(data.question_id);
        }
        const frag = document.createDocumentFragment();
        remaining.forEach((g) => {
          const divider = document.createElement('div');
          divider.className = 'iteration-divider';
          divider.textContent = g.title;
          frag.appendChild(divider);
          const content = document.createElement('div');
          content.innerHTML = g.items.join('\n');
          while (content.firstChild) frag.appendChild(content.firstChild);
        });
        showMoreBtn.replaceWith(frag);
      });
    }
  }

  // Restore scroll position: stick to bottom if user was there, otherwise preserve position
  if (preserveOpenState) {
    if (wasAtBottom) {
      detailBody.scrollTop = detailBody.scrollHeight;
    } else {
      detailBody.scrollTop = scrollTop;
    }
  }
}

async function selectQuestion(questionId) {
  activeQuestionId = questionId;
  _lastQuestionListFingerprint = '';  // Force re-render for active highlight
  _lastDetailFingerprint = '';  // Force fresh render on manual selection
  renderQuestionList();

  // Show loading indicator immediately
  const cached = (questionsCache || []).find((q) => String(q.question_id) === String(questionId));
  if (cached) {
    displayQuestionDetail({ ...cached, _loading: true });
  }

  try {
    let data = null;
    // Try API first (live mode)
    try {
      const url = `/api/questions/${encodeURIComponent(questionId)}`;
      const res = await fetch(url);
      if (res.ok) {
        data = await res.json();
        console.log(`[selectQuestion] API OK for ${questionId}, trajectory: ${(data.trajectory||[]).length} steps`);
      } else {
        console.warn(`[selectQuestion] API ${res.status} for ${url}`);
      }
    } catch (e) {
      console.warn('[selectQuestion] API fetch failed:', e.message);
    }

    // Fall back to on-demand fetch from run file (static mode)
    if (!data && _currentRunFile) {
      data = await _fetchQuestionFromRunFile(questionId, _currentRunFile);
    }

    // Last resort: show cached data without trajectory
    if (!data && cached) {
      console.warn('[selectQuestion] Using cached data (no trajectory)');
      data = cached;
    }

    if (!data) throw new Error('Question not found');
    displayQuestionDetail(data);
  } catch (err) {
    detailHeader.textContent = 'Error loading question';
    detailBody.textContent = '—';
    console.error('[selectQuestion] Error:', err);
  }
}

// Fetch a single question's full data (including trajectory) from a static run file.
// Uses streaming JSON parse to avoid loading the entire file into memory.
async function _fetchQuestionFromRunFile(questionId, runFile) {
  try {
    const res = await fetch(`./data/runs/${runFile}`);
    if (!res.ok) return null;

    // Stream the response and search for the target question
    const text = await res.text();

    // Find the question in the JSON by searching for its ID
    // We parse the full text but immediately extract just the one question
    // to minimize time the full data is in memory.
    const data = JSON.parse(text);
    const questions = data.questions || [];
    const match = questions.find(q => String(q.question_id) === String(questionId));

    // Return just the matched question (let the rest be GC'd)
    return match || null;
  } catch (e) {
    console.error('Failed to fetch question from run file:', e);
    return null;
  }
}

function renderRunSelector() {
  runSelect.innerHTML = '';
  runIndex.forEach((run) => {
    const opt = document.createElement('option');
    const acc = run.accuracy !== undefined && run.accuracy !== null ? ` ${(run.accuracy * 100).toFixed(1)}%` : '';
    opt.value = run.run_id;
    opt.textContent = `${run.run_id}${acc}`;
    runSelect.appendChild(opt);
  });
}

async function loadRun(runId) {
  const meta = runIndex.find((r) => r.run_id === runId) || runIndex[0];
  if (!meta) {
    workerGrid.textContent = 'No runs found. Add JSON files under dashboard/data/runs.';
    return;
  }
  runSelect.value = meta.run_id;
  const res = await fetch(`./data/runs/${meta.file}`);
  if (!res.ok) throw new Error('Failed to load run file');
  const data = await res.json();
  currentRun = data;
  _currentRunFile = meta.file;  // Track for on-demand trajectory fetch

  // --- Performance: strip trajectories from questionsCache ---
  // Trajectories can be 100s of KB per question and are NOT kept in memory.
  // They are fetched on demand when the user clicks a question.
  const rawQuestions = data.questions || [];
  questionsCache = rawQuestions.map(q => {
    const { trajectory, reasoning, ...light } = q;
    return light;
  });

  // Pass lightweight questions to updateSummary (token_usage is preserved)
  activeQuestionId = null;
  _lastQuestionListFingerprint = '';
  _clearQuestionNodeCaches();
  updateSummary({ ...data, questions: questionsCache });
  // Free original heavy questions array to reclaim memory
  data.questions = null;

  renderWorkerGrid();
  renderQuestionList();
}

async function loadIndex() {
  try {
    const res = await fetch('./data/index.json');
    if (!res.ok) throw new Error('Failed to load index');
    const data = await res.json();
    runIndex = data.runs || [];
    renderRunSelector();
    if (runIndex.length) {
      await loadRun(runIndex[0].run_id);
    } else {
      workerGrid.textContent = 'No runs indexed yet.';
    }
  } catch (err) {
    workerGrid.textContent = 'Unable to load runs (check data/index.json on GitHub Pages).';
    console.error(err);
  }
}

runSelect.addEventListener('change', (e) => {
  loadRun(e.target.value).catch((err) => console.error(err));
});

refreshRun.addEventListener('click', () => {
  const current = runSelect.value || (runIndex[0] && runIndex[0].run_id);
  if (current) {
    loadRun(current).catch((err) => console.error(err));
  }
  // Also refresh live status
  fetchLiveStatus().catch((err) => console.error(err));
});

// ========== LIVE API SUPPORT ==========
let liveMode = false;
let liveInterval = null;
let liveQuestion = null;
let liveWorkers = [];  // For parallel mode

async function fetchLiveStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return null;
    const data = await res.json();
    return data;
  } catch (err) {
    return null;
  }
}

async function fetchLiveQuestions() {
  try {
    const res = await fetch('/api/questions');
    if (!res.ok) return null;
    const data = await res.json();
    return data;
  } catch (err) {
    return null;
  }
}

async function fetchLiveQuestion() {
  try {
    const res = await fetch('/api/live');
    if (!res.ok) return null;
    const data = await res.json();
    return data.active ? data.question : null;
  } catch (err) {
    return null;
  }
}

async function fetchLiveWorkers() {
  try {
    const res = await fetch('/api/live/workers');
    if (!res.ok) return [];
    const data = await res.json();
    console.log('fetchLiveWorkers:', data.workers?.length || 0, 'workers');
    return data.workers || [];
  } catch (err) {
    console.error('fetchLiveWorkers error:', err);
    return [];
  }
}

let _lastGoodQMeta = {};

function updateLiveUI(status, questions) {
  // Prefer questions meta (from dashboard_questions.json) over status (from log parsing)
  // Cache last good meta to prevent flashing when JSON is mid-write.
  // Merge new meta into cached meta so sparse updates don't wipe existing fields.
  if (questions && questions.meta && questions.meta.total_questions) {
    _lastGoodQMeta = { ..._lastGoodQMeta, ...questions.meta };
  }
  const qMeta = _lastGoodQMeta;
  const p = status?.progress || {};
  
  // Use questions meta if available, otherwise fall back to status
  const answered = qMeta.answered ?? p.answered ?? 0;
  const total = qMeta.total_questions ?? p.total ?? p.total_questions ?? 0;
  const accuracy = qMeta.accuracy ?? p.accuracy ?? null;
  const rate = qMeta.rate_s_per_q ?? p.rate_seconds_per_question ?? null;
  const correctCount = qMeta.correct ?? p.estimated_correct ?? (answered && accuracy !== null ? Math.round(answered * accuracy) : null);
  
  progressMetric.textContent = `${answered} / ${total || '--'}`;
  accuracyMetric.textContent = accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : '--%';
  rateMetric.textContent = rate !== null ? `${rate.toFixed(1)} s/q` : '-- s/q';
  
  if (total && answered) {
    const pct = Math.min(100, Math.max(0, (answered / total) * 100));
    progressBar.style.width = `${pct}%`;
  }
  
  totalQuestions.textContent = `Total questions: ${total || '--'}`;
  estimatedCorrect.textContent = correctCount !== null ? `Correct: ${correctCount}` : 'Correct: --';

  // Update token usage card with input/output breakdown
  const tokenTotal = qMeta.token_usage || {};
  const inTok = tokenTotal.input_tokens || 0;
  const outTok = tokenTotal.output_tokens || 0;
  tokensMetric.textContent = fmtTokens(inTok + outTok);
  tokensBreakdown.textContent = `In: ${fmtTokens(inTok)} | Out: ${fmtTokens(outTok)}`;

  // Update cost card (use per-agent rates for accurate total)
  const cost = calcTotalCostByAgent(qMeta.token_usage_by_agent);
  costMetric.textContent = fmtCost(cost.total);
  costBreakdown.textContent = `In: ${fmtCost(cost.inCost)} | Out: ${fmtCost(cost.outCost)}`;

  // Per-agent token & cost breakdown
  renderAgentBreakdowns(qMeta.token_usage_by_agent);

  // Update meta - show model and dataset from questions meta
  const model = qMeta.model || 'Live evaluation';
  const dataset = qMeta.dataset || (status?.log_file?.split('/').pop() || 'eval_live.log');
  const runId = qMeta.run_id || '—';
  runMetric.textContent = runId;
  videoMetric.textContent = model;
  questionMetric.textContent = dataset;

  // Update error banner - only show fatal errors (no answer produced)
  const errorBanner = document.getElementById('errorBanner');
  const errorSummary = qMeta.error_summary || {};
  const failedCount = qMeta.failed || 0;

  if (failedCount > 0) {
    let html = `<div class="error-title">⚠ ${failedCount} question(s) failed</div>`;
    for (const [err, count] of Object.entries(errorSummary)) {
      html += `<div class="error-line">× ${count}: ${escapeHtml(err)}</div>`;
    }
    errorBanner.innerHTML = html;
    errorBanner.style.display = 'block';
  } else {
    errorBanner.style.display = 'none';
  }
  
  // Update questions if available
  if (questions && questions.questions && questions.questions.length) {
    questionsCache = questions.questions;
  }
  
  // Always re-render to show live question at top
  renderQuestionList();
  
  // Update worker status grid
  renderWorkerGrid();
}

// Throttle the heavy /api/questions fetch — only every 5th poll (every 10s).
// Status, live question, and workers still poll every 2s for responsiveness.
let _pollCount = 0;

async function pollLive() {
  _pollCount++;
  const fetchQuestions = _pollCount % 5 === 0;

  const [status, questions, live, workers] = await Promise.all([
    fetchLiveStatus(),
    fetchQuestions ? fetchLiveQuestions() : Promise.resolve(null),
    fetchLiveQuestion(),
    fetchLiveWorkers(),
  ]);

  // Only update state if we got real data — prevents flashing to empty
  // when the JSON file is mid-write
  const hasStatusData = status && status.progress && (status.progress.answered !== null || status.progress.total !== null);
  const hasQuestionsData = questions && questions.meta && questions.meta.total_questions;
  const hasLiveQuestion = live !== null;
  const hasLiveWorkers = workers && workers.length > 0;

  if (hasLiveQuestion || hasLiveWorkers) {
    liveQuestion = live;
    liveWorkers = workers || [];
  }

  if (hasStatusData || hasQuestionsData || hasLiveQuestion || hasLiveWorkers) {
    liveMode = true;
    updateLiveUI(status, questions);
    
    // Auto-update detail view: fetch full detail (with trajectory) for the
    // actively viewed worker.  The lightweight poll data doesn't include
    // trajectories, so we fetch on demand only when the step count changes.
    if (activeQuestionId && displayedQuestionId === activeQuestionId) {
      const activeWorker = liveWorkers.find(w => w.question_id === activeQuestionId);
      if (activeWorker) {
        const newSteps = activeWorker.trajectory_steps ?? 0;
        const oldFp = _lastDetailFingerprint;
        const newFp = `${activeWorker.question_id}:${newSteps}::${activeWorker.predicted || ''}`;
        if (newFp !== oldFp) {
          // Step count changed — fetch full detail
          const wid = activeWorker.worker_id ?? 0;
          fetch(`/api/live/workers/${wid}`)
            .then(r => r.ok ? r.json() : null)
            .then(full => { if (full) displayQuestionDetail(full, true); })
            .catch(() => {});
        }
      }
    }
  }
}

async function initDashboard() {
  // Load pricing config
  try {
    const cfgRes = await fetch('/api/config');
    if (cfgRes.ok) {
      const cfg = await cfgRes.json();
      if (cfg.pricing) {
        PRICE_INPUT_PER_M = cfg.pricing.input_per_million || 0;
        PRICE_OUTPUT_PER_M = cfg.pricing.output_per_million || 0;
        PRICE_CURRENCY = cfg.pricing.currency || 'USD';
        // Load per-agent pricing overrides
        if (cfg.pricing.vlm) AGENT_PRICING.vlm = cfg.pricing.vlm;
        if (cfg.pricing.summarizer) AGENT_PRICING.summarizer = cfg.pricing.summarizer;
      }
    }
  } catch (e) { /* use defaults */ }

  // Try live API first
  const status = await fetchLiveStatus();
  const questions = await fetchLiveQuestions();
  const live = await fetchLiveQuestion();
  const workers = await fetchLiveWorkers();
  
  // Check if we have any live data
  const hasStatusData = status && status.progress && (status.progress.answered !== null || status.progress.total !== null);
  const hasQuestionsData = questions && questions.meta && questions.meta.total_questions;
  const hasLiveQuestion = live !== null;
  const hasLiveWorkers = workers && workers.length > 0;
  
  if (hasStatusData || hasQuestionsData || hasLiveQuestion || hasLiveWorkers) {
    liveMode = true;
    liveQuestion = live;
    liveWorkers = workers || [];
    console.log(`Live evaluation detected (${liveWorkers.length} workers), starting live updates`);
    updateLiveUI(status, questions);
    // Poll every 2 seconds for more responsive live updates
    liveInterval = setInterval(pollLive, 2000);
    // Don't load static index when in live mode
    return;
  }
  
  // Only load static index if not in live mode
  try {
    await loadIndex();
  } catch (err) {
    workerGrid.textContent = 'No data available. Start an evaluation or add archived runs.';
  }
}

// Start dashboard
initDashboard();
