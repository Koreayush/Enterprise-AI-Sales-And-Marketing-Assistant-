/* ================================================================
   Enterprise AI Assistant — Application Logic
   ================================================================ */

const API_BASE = 'http://localhost:8000';

// ---- Navigation ----
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const page = btn.dataset.page;
        // Update nav
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // Update pages
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${page}`).classList.add('active');
    });
});

// ---- Health Check ----
async function checkHealth() {
    const badge = document.getElementById('health-badge');
    const text = badge.querySelector('.health-text');
    try {
        const res = await fetch(`${API_BASE}/health/`);
        const data = await res.json();
        if (data.status === 'ok' && data.pipeline === 'Ready') {
            badge.className = 'health-badge online';
            text.textContent = 'Pipeline Ready';
        } else {
            badge.className = 'health-badge offline';
            text.textContent = 'Pipeline Not Ready';
        }
    } catch {
        badge.className = 'health-badge offline';
        text.textContent = 'Disconnected';
    }
}

checkHealth();
setInterval(checkHealth, 15000);

// ================================================================
// CHAT
// ================================================================

let isChatLoading = false;

function askSuggestion(query) {
    document.getElementById('chat-input').value = query;
    sendChat();
}

function toggleContext() {
    const field = document.getElementById('context-field');
    field.classList.toggle('hidden');
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

document.getElementById('chat-input').addEventListener('input', function () {
    autoResize(this);
});

async function sendChat() {
    if (isChatLoading) return;

    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;

    const context = document.getElementById('customer-context')?.value?.trim() || '';

    // Remove welcome message
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage('user', query);
    input.value = '';
    autoResize(input);

    // Show typing indicator
    const typingId = showTyping();
    isChatLoading = true;
    document.getElementById('send-btn').disabled = true;

    try {
        const res = await fetch(`${API_BASE}/chat/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, customer_context: context }),
        });

        removeTyping(typingId);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        addAssistantMessage(data);
    } catch (err) {
        removeTyping(typingId);
        addMessage('assistant', `⚠️ Error: ${err.message}`);
        showToast(err.message, 'error');
    } finally {
        isChatLoading = false;
        document.getElementById('send-btn').disabled = false;
    }
}

function addMessage(role, text) {
    const container = document.getElementById('chat-messages');
    const avatar = role === 'user' ? 'You' : '✦';

    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">
            <div class="message-content">${escapeHtml(text)}</div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function addAssistantMessage(data) {
    const container = document.getElementById('chat-messages');

    const evalData = data.evaluation?.generation;
    let evalHtml = '';
    let sourcesHtml = '';

    if (evalData) {
        const faithClass = evalData.faithfulness >= 0.8 ? 'good' : evalData.faithfulness >= 0.5 ? 'warn' : 'bad';
        const relClass = evalData.answer_relevance >= 0.8 ? 'good' : evalData.answer_relevance >= 0.5 ? 'warn' : 'bad';
        const qualClass = evalData.overall_quality >= 0.7 ? 'good' : evalData.overall_quality >= 0.5 ? 'warn' : 'bad';

        evalHtml = `
            <div class="eval-badge">
                <span class="eval-pill ${faithClass}">Faithfulness ${(evalData.faithfulness * 100).toFixed(0)}%</span>
                <span class="eval-pill ${relClass}">Relevance ${(evalData.answer_relevance * 100).toFixed(0)}%</span>
                <span class="eval-pill ${qualClass}">Quality ${(evalData.overall_quality * 100).toFixed(0)}%</span>
                ${evalData.is_hallucination ? '<span class="eval-pill bad">⚠ Hallucination Detected</span>' : ''}
            </div>
        `;
    }

    if (data.sources && data.sources.length > 0) {
        sourcesHtml = `<div class="sources-tag">Sources: <span>${data.sources.join(', ')}</span></div>`;
    }

    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">✦</div>
        <div class="message-body">
            <div class="message-content">${escapeHtml(data.response)}</div>
            ${evalHtml}
            ${sourcesHtml}
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('chat-messages');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">✦</div>
        <div class="message-body">
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ================================================================
// EMAIL GENERATOR
// ================================================================

async function generateEmail() {
    const btn = document.getElementById('generate-email-btn');
    const name = document.getElementById('email-name').value.trim();
    const company = document.getElementById('email-company').value.trim();
    const painPoint = document.getElementById('email-pain').value.trim();
    const emailType = document.getElementById('email-type').value;
    const context = document.getElementById('email-context').value.trim();

    if (!name || !company || !painPoint) {
        showToast('Please fill in Customer Name, Company, and Pain Point', 'error');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Generating...';

    try {
        const res = await fetch(`${API_BASE}/generate-email/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                customer_name: name,
                company,
                pain_point: painPoint,
                email_type: emailType,
                context,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        const preview = document.getElementById('email-preview');
        preview.textContent = data.content;
        preview.classList.remove('empty-state');

        document.getElementById('copy-email-btn').style.display = 'flex';
        showToast('Email generated successfully!', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            Generate Email
        `;
    }
}

function copyEmail() {
    const content = document.getElementById('email-preview').textContent;
    navigator.clipboard.writeText(content).then(() => {
        showToast('Email copied to clipboard!', 'success');
    });
}

// ================================================================
// DOCUMENT UPLOAD
// ================================================================

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) uploadFile(files[0]);
    e.target.value = '';
}

async function uploadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'csv'].includes(ext)) {
        showToast('Unsupported file type. Use PDF, TXT, or CSV.', 'error');
        return;
    }

    const progressDiv = document.getElementById('upload-progress');
    const bar = document.getElementById('upload-bar');
    const status = document.getElementById('upload-status');
    const filename = document.getElementById('upload-filename');

    progressDiv.classList.remove('hidden');
    filename.textContent = file.name;
    bar.style.width = '20%';
    status.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        bar.style.width = '50%';
        status.textContent = 'Indexing document...';

        const res = await fetch(`${API_BASE}/upload-doc/`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${res.status})`);
        }

        const data = await res.json();
        bar.style.width = '100%';
        status.textContent = `✓ ${data.message}`;

        addUploadItem(file.name, data.chunks_created);
        showToast(`${file.name} uploaded and indexed!`, 'success');

        setTimeout(() => {
            progressDiv.classList.add('hidden');
            bar.style.width = '0%';
        }, 3000);
    } catch (err) {
        bar.style.width = '100%';
        bar.style.background = 'var(--danger)';
        status.textContent = `✗ ${err.message}`;
        showToast(err.message, 'error');

        setTimeout(() => {
            progressDiv.classList.add('hidden');
            bar.style.width = '0%';
            bar.style.background = '';
        }, 4000);
    }
}

function addUploadItem(name, chunks) {
    const list = document.getElementById('upload-list');
    const emptyHint = list.querySelector('.empty-hint');
    if (emptyHint) emptyHint.remove();

    const item = document.createElement('div');
    item.className = 'upload-item';
    item.innerHTML = `
        <div class="upload-item-info">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            <span>${escapeHtml(name)}</span>
        </div>
        <span class="upload-item-meta">${chunks} chunks</span>
    `;
    list.prepend(item);
}

// ================================================================
// EVALUATION
// ================================================================

async function runBenchmark() {
    const btn = document.getElementById('benchmark-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Running...';

    try {
        const res = await fetch(`${API_BASE}/evaluate/benchmark/`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Benchmark failed (${res.status})`);
        }

        const data = await res.json();
        displayBenchmarkResults(data);
        showToast(`Benchmark ${data.status}: Quality ${(data.avg_overall_quality * 100).toFixed(0)}%`, data.status === 'PASS' ? 'success' : 'error');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Run Benchmark
        `;
    }
}

function displayBenchmarkResults(data) {
    // Update metric cards
    updateMetric('faithfulness', data.avg_faithfulness, 0.9);
    updateMetric('relevance', data.avg_answer_relevance, 0.85);
    updateMetric('quality', data.avg_overall_quality, 0.7);
    updateMetric('hallucination', data.hallucination_rate, 0.05, true);

    // Status badge
    const statusDiv = document.getElementById('benchmark-status');
    statusDiv.className = `benchmark-status ${data.status.toLowerCase()}`;
    statusDiv.textContent = data.status === 'PASS'
        ? `✓ PASS — ${data.num_test_cases} test cases`
        : `✗ FAIL — ${data.num_test_cases} test cases`;

    // Table
    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = '';
    data.results.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="max-width:260px;">${escapeHtml(r.query)}</td>
            <td><span class="eval-pill ${r.faithfulness >= 0.8 ? 'good' : r.faithfulness >= 0.5 ? 'warn' : 'bad'}">${(r.faithfulness * 100).toFixed(0)}%</span></td>
            <td><span class="eval-pill ${r.answer_relevance >= 0.8 ? 'good' : r.answer_relevance >= 0.5 ? 'warn' : 'bad'}">${(r.answer_relevance * 100).toFixed(0)}%</span></td>
            <td><span class="eval-pill ${r.overall_quality >= 0.7 ? 'good' : r.overall_quality >= 0.5 ? 'warn' : 'bad'}">${(r.overall_quality * 100).toFixed(0)}%</span></td>
            <td>${r.is_hallucination ? '<span class="eval-pill bad">Yes</span>' : '<span class="eval-pill good">No</span>'}</td>
        `;
        tbody.appendChild(tr);
    });

    document.getElementById('benchmark-results').classList.remove('hidden');
}

function updateMetric(name, value, target, inverted = false) {
    const valueEl = document.getElementById(`metric-${name}`);
    const barEl = document.getElementById(`bar-${name}`);

    valueEl.textContent = (value * 100).toFixed(1) + '%';
    const barPercent = inverted ? Math.min(value / 0.2, 1) * 100 : value * 100;
    setTimeout(() => { barEl.style.width = barPercent + '%'; }, 100);
}

async function saveReport() {
    try {
        const res = await fetch(`${API_BASE}/evaluate/save/`, { method: 'POST' });
        if (!res.ok) throw new Error('Save failed');
        const data = await res.json();
        showToast(`Report saved to ${data.report_path}`, 'success');

        // Fetch and show report
        const reportRes = await fetch(`${API_BASE}/evaluate/report/`);
        if (reportRes.ok) {
            const reportData = await reportRes.json();
            document.getElementById('report-content').textContent = reportData.report;
            document.getElementById('report-card').classList.remove('hidden');
        }
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ================================================================
// TOAST NOTIFICATIONS
// ================================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    };

    toast.innerHTML = `${icons[type] || icons.info}<span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s var(--ease-out) forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ================================================================
// UTILITIES
// ================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
