// -- Config -----------------------------------------------------------
// All endpoints are same-origin: admin proxies /jobs to the generator API.
const ADMIN_URL = '';

let currentTask       = null;
let studentId         = 'demo';
let taskNumber        = 10;
let sessionSubmissions = [];   // accumulated for POST /jobs

// Map submission_id -> index in sessionSubmissions so we can patch tags later.
const submissionIndex = {};

// -- Init -------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    const idInput   = document.getElementById('studentId');
    const taskSel   = document.getElementById('taskNumber');
    const newBtn    = document.getElementById('newTaskBtn');
    const submitBtn = document.getElementById('submitBtn');
    const genBtn    = document.getElementById('genConspectBtn');
    const answer    = document.getElementById('answerInput');

    studentId  = idInput.value.trim() || 'demo';
    taskNumber = Number(taskSel.value || 10);

    idInput.addEventListener('change', e => {
        studentId = e.target.value || 'demo';
        sessionSubmissions = [];
    });
    taskSel.addEventListener('change', e => {
        taskNumber = Number(e.target.value || 10);
        currentTask = null;
        sessionSubmissions = [];
        document.getElementById('result').innerHTML = '';
        loadNewTask();
    });

    newBtn.addEventListener('click', loadNewTask);
    submitBtn.addEventListener('click', submitAnswer);
    genBtn.addEventListener('click', generateConspect);
    answer.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); submitAnswer(); } });

    loadNewTask();
    startMonitor();
    checkHealth();
});

// -- Task -------------------------------------------------------------

async function loadNewTask() {
    const body = document.getElementById('taskText');
    const res  = document.getElementById('result');
    const inp  = document.getElementById('answerInput');
    body.innerHTML = '<span class="loading">Загрузка задачи...</span>';
    res.innerHTML = ''; res.className = '';
    inp.value = '';
    try {
        const r = await fetch(`/task/${taskNumber}/new?student_id=${studentId}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        currentTask = d;
        body.innerHTML = renderMarkdown(sanitizeTaskText(d.text || ''));
        typeset(body);
    } catch (e) {
        body.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
    }
}

// -- Submit ------------------------------------------------------------

async function submitAnswer() {
    const inp = document.getElementById('answerInput');
    const res = document.getElementById('result');
    const val = inp.value.trim();
    if (!val || !currentTask) return;

    res.innerHTML = '<span class="loading">Проверка...</span>';
    res.className = '';
    try {
        const r = await fetch(`/task/${taskNumber}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                student_id: studentId,
                task_number: taskNumber,
                task_id: currentTask.task_id,
                answer: val,
            }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();

        let h = d.is_correct
            ? `<div class="answer-line correct-line">+ Правильно! Ответ: <code>${esc(d.correct_answer)}</code></div>`
            : `<div class="answer-line wrong-line">X Неправильно. Правильный ответ: <code>${esc(d.correct_answer)}</code></div>`;

        if (d.queued) {
            h += `<div class="queue-notice">> Диагноз поставлен в очередь (ID: <code>${esc(d.submission_id.slice(0,8))}...</code>)</div>`;
        } else {
            h += `<div class="queue-notice warn">! Kafka недоступна -- диагноз не будет получен</div>`;
        }
        res.innerHTML = h;
        res.className = d.is_correct ? 'correct' : 'wrong';

        // Track submission for conspect generation
        const sub = {
            submission_id: d.submission_id,
            task_id: currentTask.task_id,
            task_number: taskNumber,
            task_text: currentTask.text || '',
            is_correct: d.is_correct,
            student_answer: val,
            correct_answer: d.correct_answer || null,
            diagnostic_tags: [],   // will be filled in when eval-worker finishes
            submitted_at: new Date().toISOString(),
        };
        submissionIndex[d.submission_id] = sessionSubmissions.length;
        sessionSubmissions.push(sub);
    } catch (e) {
        res.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
        res.className = '';
    }
}

// -- Conspect ----------------------------------------------------------

async function generateConspect() {
    const el = document.getElementById('conspectContent');

    // Refresh diagnostic_tags from monitoring before submitting job
    await _patchTagsFromMonitor();

    const jobId = crypto.randomUUID();
    const nSubs = sessionSubmissions.length;
    el.innerHTML = `<span class="loading">Создание задания${nSubs === 0 ? ' (нет ответов -- базовый конспект)' : ` (${nSubs} ответ${nSubs === 1 ? '' : 'ов'})`}...</span>`;

    try {
        const r = await fetch('/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ui-local' },
            body: JSON.stringify({ job_id: jobId, user_id: studentId, submissions: sessionSubmissions }),
        });
        if (!r.ok) {
            const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
            throw new Error(err.detail || `HTTP ${r.status}`);
        }
        el.innerHTML = '<span class="loading">Генерация конспекта...</span>';
        const result = await _pollJob(jobId, el);
        el.innerHTML = renderMarkdown(sanitizeConspectText(result.text || ''));
        typeset(el);
    } catch (e) {
        el.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
    }
}

async function _patchTagsFromMonitor() {
    try {
        const r = await fetch(`${ADMIN_URL}/api/submissions?limit=200`);
        if (r.ok) _patchSessionTags(await r.json());
    } catch (_) { /* best-effort */ }
}

function _patchSessionTags(rows) {
    for (const row of rows) {
        const idx = submissionIndex[row.submission_id];
        if (idx !== undefined && row.tags && row.tags.length) {
            sessionSubmissions[idx].diagnostic_tags = row.tags;
        }
    }
}

async function _pollJob(jobId, statusEl, maxMs = 180_000, intervalMs = 2_000) {
    const deadline = Date.now() + maxMs;
    while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, intervalMs));
        const r = await fetch(`/jobs/${jobId}`, {
            headers: { 'Authorization': 'Bearer ui-local' },
        });
        if (!r.ok) throw new Error(`Poll failed: HTTP ${r.status}`);
        const job = await r.json();
        if (job.status === 'done') return job.result;
        if (job.status === 'failed') throw new Error(job.error_message || 'Генерация не удалась');
        if (statusEl) statusEl.innerHTML = `<span class="loading">Генерация конспекта (${esc(job.status)})...</span>`;
    }
    throw new Error('Timeout: конспект не был создан за 3 минуты. Убедитесь, что conspect-worker запущен.');
}

// -- Monitor -----------------------------------------------------------

const POLL_MS = 3000;

function startMonitor() {
    refreshMonitor();
    setInterval(refreshMonitor, POLL_MS);
}

async function refreshMonitor() {
    await Promise.all([refreshSubmissions(), refreshJobs(), refreshStats(), refreshJobStats()]);
    document.getElementById('lastRefresh').textContent = 'обновлено ' + new Date().toLocaleTimeString('ru');
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? '--';
}

async function refreshStats() {
    try {
        const r = await fetch(`${ADMIN_URL}/api/stats`);
        if (!r.ok) return;
        const s = await r.json(), c = s.counts || {}, lat = s.latency_ms || {};
        setText('statPending', c.pending);
        setText('statDone', c.done);
        setText('statAvg', fmtMs(lat.avg));
        setText('statP95', fmtMs(lat.p95));
        setText('statThroughput', s.throughput_per_min ?? 0);
        setText('statOldest', s.oldest_pending_s != null ? fmtSec(s.oldest_pending_s) : '--');
        setText('statLag', s.kafka_lag);
        setText('hdrSubCount', c.total);
        document.getElementById('statLag')?.parentElement
            ?.classList.toggle('warn', (s.kafka_lag ?? 0) > 10);
    } catch (_) { /* silent */ }
}

async function refreshJobStats() {
    try {
        const r = await fetch(`${ADMIN_URL}/api/job-stats`);
        if (!r.ok) return;
        const s = await r.json(), c = s.counts || {}, lat = s.latency_ms || {};
        setText('jobStatQueued', c.queued);
        setText('jobStatRunning', c.running);
        setText('jobStatDone', c.done);
        setText('jobStatFailed', c.failed);
        setText('jobStatAvg', fmtMs(lat.avg));
        setText('jobStatP95', fmtMs(lat.p95));
        setText('jobStatOldest', s.oldest_queued_s != null ? fmtSec(s.oldest_queued_s) : '--');
        setText('hdrJobCount', c.total);
    } catch (_) { /* silent */ }
}

function fmtMs(ms) {
    if (ms == null) return '--';
    if (ms < 1000) return Math.round(ms) + 'мс';
    if (ms < 60_000) return (ms / 1000).toFixed(1) + 'с';
    return Math.floor(ms / 60_000) + 'м ' + Math.round((ms % 60_000) / 1000) + 'с';
}

function fmtSec(s) {
    if (s == null) return '--';
    if (s < 60) return Math.round(s) + 'с';
    if (s < 3600) return Math.floor(s / 60) + 'м ' + Math.round(s % 60) + 'с';
    return Math.floor(s / 3600) + 'ч';
}

async function refreshSubmissions() {
    try {
        const r = await fetch(`${ADMIN_URL}/api/submissions`);
        if (!r.ok) return;
        const rows = await r.json();
        document.getElementById('subCount').textContent = rows.length;
        const tbody = document.getElementById('subBody');
        tbody.innerHTML = rows.map(row => {
            const payload = row.payload || {};
            const answer  = payload.student_answer ?? '--';
            const correct = row.status === 'done' || row.status === 'failed'
                ? (payload.is_correct ? '+' : 'X') : '--';
            const tags    = (row.tags || []).join(', ') || '--';
            const err     = row.last_error ? truncate(row.last_error, 40) : '--';
            const queued  = fmtQueueTime(row.created_at, row.diagnosed_at);
            return `<tr class="status-${row.status}">
                <td class="mono">${fmtTime(row.created_at)}</td>
                <td>${esc(row.user_id)}</td>
                <td>${esc(String(row.task_number))}</td>
                <td class="mono">${esc(truncate(answer, 20))}</td>
                <td class="center">${correct}</td>
                <td><span class="status-pill status-${row.status}">${esc(row.status)}</span></td>
                <td class="mono">${queued}</td>
                <td class="tags-cell">${esc(tags)}</td>
                <td class="err-cell" title="${esc(row.last_error || '')}">${esc(err)}</td>
            </tr>`;
        }).join('');

        _patchSessionTags(rows);
    } catch (_) { /* silent */ }
}

async function refreshJobs() {
    try {
        const r = await fetch(`${ADMIN_URL}/api/jobs`);
        if (!r.ok) return;
        const rows = await r.json();
        document.getElementById('jobCount').textContent = rows.length;
        const tbody = document.getElementById('jobBody');
        tbody.innerHTML = rows.map(row => {
            const err = row.error_message ? truncate(row.error_message, 50) : '--';
            return `<tr class="status-${row.status}">
                <td class="mono">${fmtTime(row.created_at)}</td>
                <td>${esc(row.user_id)}</td>
                <td><span class="status-pill status-${row.status}">${esc(row.status)}</span></td>
                <td class="err-cell" title="${esc(row.error_message || '')}">${esc(err)}</td>
                <td class="center">${row.retry_count}</td>
            </tr>`;
        }).join('');
    } catch (_) { /* silent */ }
}

// -- Health check ------------------------------------------------------

async function checkHealth() {
    try {
        const r = await fetch('/readyz');
        const d = r.ok ? await r.json() : {};
        const dbOk    = d.db    ?? r.ok;
        const kafkaOk = d.kafka ?? r.ok;
        _setBadge('dbStatus',    dbOk    ? 'ok' : 'err', dbOk    ? 'ok' : 'err');
        _setBadge('kafkaStatus', kafkaOk ? 'ok' : 'err', kafkaOk ? 'ok' : 'err');
    } catch (_) {
        _setBadge('dbStatus',    'err', 'err');
        _setBadge('kafkaStatus', 'err', 'err');
    }
    setTimeout(checkHealth, 10_000);
}

function _setBadge(id, text, state) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = `status-badge status-${state}`;
}

// -- Helpers -----------------------------------------------------------

function fmtTime(iso) {
    if (!iso) return '--';
    return new Date(iso).toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtQueueTime(createdAt, diagnosedAt) {
    if (!createdAt) return '--';
    const end = diagnosedAt ? new Date(diagnosedAt) : new Date();
    const sec = Math.round((end - new Date(createdAt)) / 1000);
    if (sec < 60) return sec + 'с';
    if (sec < 3600) return Math.floor(sec / 60) + 'м ' + (sec % 60) + 'с';
    return Math.floor(sec / 3600) + 'ч ' + Math.floor((sec % 3600) / 60) + 'м';
}

function truncate(s, n) {
    if (!s) return '';
    return String(s).length > n ? String(s).slice(0, n) + '...' : String(s);
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function renderMarkdown(text) {
    if (!text) return '<p class="muted">Пусто</p>';
    const normalized = normalizeMathDelimiters(String(text));
    if (typeof marked !== 'undefined') {
        try { marked.setOptions({ breaks: true, gfm: true }); return marked.parse(normalized); } catch (_) {}
    }
    return `<p>${esc(normalized)}</p>`;
}

function typeset(el) {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([el]).catch(() => {});
}

function normalizeMathDelimiters(text) {
    let t = text;
    t = t.replace(/\\\[(.*?)\\\]/gs, (_, expr) => `\n$$${expr.trim()}$$\n`);
    t = t.replace(/\\\((.*?)\\\)/gs, (_, expr) => `$${expr.trim()}$`);
    t = t.replace(/\[\s*(\\(?:frac|sqrt|cdot|Delta|alpha|beta|gamma|pi|sin|cos|tan|ln|log)[\s\S]*?)\s*\]/g, (_, expr) => `\n$$${expr.trim()}$$\n`);
    return t;
}

function sanitizeConspectText(text) {
    let t = String(text || '');
    t = t.replace(/^\s*фаза\s*\d+.*$/gim, '');
    t = t.replace(/^\s*.*(?:scaffold(?:ing)?|misconception-targeted|stepwise|phase)\s*.*$/gim, '');
    t = t.replace(/\bt(?:06|1[02])_mm\d+\b/gim, '');
    t = t.replace(/\b(?:tag|id|json|prereq|subtype)\b/gim, '');
    t = t.replace(/^\s*(?:В конспекте обнаружены ошибки|Ниже приведены исправленные версии|Исправленный конспект готов)[^\n]*\n?/gim, '');
    t = t.replace(/~\s*([+-]?\d+[.,]\d+)/g, '= $1');
    t = t.replace(/примерно\s*=\s*/gi, '= ');
    t = t.replace(/\n{3,}/g, '\n\n').trim();
    return t;
}

function sanitizeTaskText(text) {
    let t = String(text || '');
    t = t.replace(/(?:\s*[.;,:]){2,}\s*$/g, '');
    t = t.replace(/\s*;\s*;\s*$/g, '');
    t = t.replace(/\s+/g, ' ');
    t = t.replace(/\(\s+/g, '(').replace(/\s+\)/g, ')');
    t = t.replace(/\s+([,.:;!?])/g, '$1');
    return t.trim();
}
