// Helios Dashboard — LM Studio-style model discovery + unified model picker
// ============================================================

// ---------- Global state ----------

var ollamaModels = [];    // [{name, size}]
var localGgufModels = []; // [{path, filename, repo_id, size_display}]
var downloadedSet = {};   // filename -> path
var trendingLoaded = false;
var lastAssignedModel = null; // for role modal

// Quants considered "recommended" (best quality/size balance)
var RECOMMENDED_QUANTS = ['Q4_K_M', 'Q4_K_L', 'Q5_K_M'];


// ---------- Bootstrap ----------

(async function boot() {
    await refreshLocalModels();
    populateAllModelSelects();
})();


// ============================================================
//  DATA FETCHING
// ============================================================

async function refreshLocalModels() {
    var ollamaP = fetch('/api/ollama/models').then(function(r){return r.json();}).catch(function(){return {models:[]};});
    var localP  = fetch('/api/hub/local').then(function(r){return r.json();}).catch(function(){return {models:[]};});
    var results = await Promise.all([ollamaP, localP]);

    ollamaModels = (results[0].models || []).filter(function(m){ return m.name; });
    localGgufModels = results[1].models || [];

    downloadedSet = {};
    localGgufModels.forEach(function(m) { downloadedSet[m.filename] = m.path; });
}


// ============================================================
//  UNIFIED MODEL SELECTOR  (Run view)
// ============================================================

function populateAllModelSelects() {
    document.querySelectorAll('.model-select').forEach(function(sel) {
        populateModelSelect(sel);
    });
    var hint = document.getElementById('no-models-hint');
    if (ollamaModels.length === 0 && localGgufModels.length === 0) {
        hint.classList.remove('hidden');
    } else {
        hint.classList.add('hidden');
    }
}

function populateModelSelect(sel) {
    var prevVal = sel.value;
    sel.textContent = '';

    if (ollamaModels.length > 0) {
        var ollamaGroup = document.createElement('optgroup');
        ollamaGroup.label = 'Ollama';
        ollamaModels.forEach(function(m) {
            var opt = document.createElement('option');
            opt.value = 'ollama::' + m.name;
            opt.textContent = m.name + (m.size ? '  (' + formatBytes(m.size) + ')' : '');
            ollamaGroup.appendChild(opt);
        });
        sel.appendChild(ollamaGroup);
    }

    if (localGgufModels.length > 0) {
        var ggufGroup = document.createElement('optgroup');
        ggufGroup.label = 'Downloaded GGUF';
        localGgufModels.forEach(function(m) {
            var opt = document.createElement('option');
            opt.value = 'gguf::' + m.path;
            opt.textContent = m.filename + '  (' + m.size_display + ')';
            ggufGroup.appendChild(opt);
        });
        sel.appendChild(ggufGroup);
    }

    if (ollamaModels.length === 0 && localGgufModels.length === 0) {
        var none = document.createElement('option');
        none.value = '';
        none.textContent = 'No models available';
        none.disabled = true;
        none.selected = true;
        sel.appendChild(none);
    }

    if (prevVal) {
        var exists = Array.from(sel.options).some(function(o){ return o.value === prevVal; });
        if (exists) sel.value = prevVal;
    }
}

function getRoleConfig(prefix) {
    var sel = document.querySelector('.model-select[data-role="' + prefix + '"]');
    var val = sel.value || '';
    if (val.startsWith('gguf::')) {
        return { provider: 'llama_cpp', model: null, model_path: val.slice(6) };
    }
    if (val.startsWith('ollama::')) {
        return { provider: 'ollama', model: val.slice(8), model_path: null };
    }
    return { provider: 'ollama', model: val || null, model_path: null };
}

function refreshRoleSelectors() {
    populateAllModelSelects();
}


// ============================================================
//  MODEL CONFIG TOGGLE
// ============================================================

function toggleModelConfig() {
    var body = document.getElementById('model-config-body');
    var toggle = document.getElementById('model-config-toggle');
    if (body.classList.contains('hidden')) {
        body.classList.remove('hidden');
        toggle.textContent = '\u2212';
    } else {
        body.classList.add('hidden');
        toggle.textContent = '+';
    }
}


// ============================================================
//  DISCOVER — Trending + Search
// ============================================================

document.getElementById('hub-search-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') searchHub();
});

// Auto-load trending when Discover tab is opened
function loadTrending() {
    if (trendingLoaded) return;
    trendingLoaded = true;
    var container = document.getElementById('trending-results');
    container.innerHTML = '<div class="loading-state">Loading trending models from HuggingFace...</div>';

    fetch('/api/hub/trending?limit=15')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            container.textContent = '';
            if (data.error) { container.innerHTML = '<div class="empty-state">Error: ' + data.error + '</div>'; return; }
            var models = data.models || [];
            if (!models.length) { container.innerHTML = '<div class="empty-state">No trending models found.</div>'; return; }
            models.forEach(function(m) { container.appendChild(buildModelCard(m)); });
        })
        .catch(function(err) {
            container.innerHTML = '<div class="empty-state">Error: ' + err.message + '</div>';
        });
}

async function searchHub() {
    var query = document.getElementById('hub-search-input').value.trim();
    if (!query) return;

    // Show search section, hide trending
    document.getElementById('trending-section').classList.add('hidden');
    document.getElementById('search-section').classList.remove('hidden');
    document.getElementById('search-title').textContent = 'Results for "' + query + '"';

    var container = document.getElementById('hub-results');
    container.innerHTML = '<div class="loading-state">Searching HuggingFace Hub...</div>';

    try {
        var res = await fetch('/api/hub/search?q=' + encodeURIComponent(query) + '&limit=25');
        var data = await res.json();
        container.textContent = '';
        if (data.error) { container.innerHTML = '<div class="empty-state">Error: ' + data.error + '</div>'; return; }

        var models = data.models || [];
        if (!models.length) {
            container.innerHTML = '<div class="empty-state">No GGUF models found for "' + query + '"</div>';
            return;
        }
        models.forEach(function(m) { container.appendChild(buildModelCard(m)); });
    } catch (err) {
        container.innerHTML = '<div class="empty-state">Error: ' + err.message + '</div>';
    }
}

function clearSearch() {
    document.getElementById('hub-search-input').value = '';
    document.getElementById('search-section').classList.add('hidden');
    document.getElementById('trending-section').classList.remove('hidden');
}


// ============================================================
//  MODEL CARD — shared between trending and search
// ============================================================

function buildModelCard(m) {
    var card = document.createElement('div');
    card.className = 'discover-card';

    var head = document.createElement('div');
    head.className = 'discover-card-head';

    var left = document.createElement('div');
    left.className = 'discover-card-left';

    var nameEl = document.createElement('div');
    nameEl.className = 'discover-card-name';
    nameEl.textContent = m.repo_id;
    left.appendChild(nameEl);

    var metaEl = document.createElement('div');
    metaEl.className = 'discover-card-meta';
    metaEl.innerHTML = '<span class="meta-downloads">' + formatNumber(m.downloads) + ' downloads</span>'
        + '<span class="meta-sep">&middot;</span>'
        + '<span class="meta-likes">' + m.likes + ' likes</span>';
    left.appendChild(metaEl);

    var arrow = document.createElement('span');
    arrow.className = 'discover-arrow';
    arrow.textContent = '\u25B6';

    head.appendChild(left);
    head.appendChild(arrow);
    card.appendChild(head);

    var body = document.createElement('div');
    body.className = 'discover-card-body hidden';
    card.appendChild(body);

    head.addEventListener('click', function() {
        var isOpen = !body.classList.contains('hidden');
        if (isOpen) {
            body.classList.add('hidden');
            arrow.textContent = '\u25B6';
        } else {
            arrow.textContent = '\u25BC';
            body.classList.remove('hidden');
            if (!body.dataset.loaded) loadRepoFiles(m.repo_id, body);
        }
    });

    return card;
}


// ============================================================
//  QUANT FILE LIST (inside expanded card)
// ============================================================

async function loadRepoFiles(repoId, bodyEl) {
    bodyEl.innerHTML = '<div class="loading-state">Loading quantisations...</div>';
    bodyEl.dataset.loaded = '1';

    try {
        var res = await fetch('/api/hub/files/' + encodeURIComponent(repoId));
        var data = await res.json();
        if (data.error) { bodyEl.innerHTML = '<div class="empty-state">Error: ' + data.error + '</div>'; return; }

        bodyEl.textContent = '';
        var files = data.files || [];
        if (!files.length) { bodyEl.innerHTML = '<div class="empty-state">No GGUF files in this repository.</div>'; return; }

        files.forEach(function(f) {
            var row = document.createElement('div');
            row.className = 'quant-row';

            var info = document.createElement('div');
            info.className = 'quant-info';

            // Quant label + recommended badge
            var labelRow = document.createElement('div');
            labelRow.className = 'quant-label-row';
            var qLabel = document.createElement('span');
            qLabel.className = 'quant-label';
            qLabel.textContent = f.quantisation || f.filename;
            labelRow.appendChild(qLabel);

            if (f.quantisation && RECOMMENDED_QUANTS.indexOf(f.quantisation) !== -1) {
                var badge = document.createElement('span');
                badge.className = 'recommended-badge';
                badge.textContent = 'Recommended';
                labelRow.appendChild(badge);
            }
            info.appendChild(labelRow);

            // Size + RAM estimate
            var metaRow = document.createElement('div');
            metaRow.className = 'quant-meta-row';
            var qSize = document.createElement('span');
            qSize.className = 'quant-meta';
            qSize.textContent = f.size_display;
            metaRow.appendChild(qSize);

            if (f.size_bytes > 0) {
                var ramEst = document.createElement('span');
                ramEst.className = 'quant-ram';
                var ramGb = (f.size_bytes * 1.15) / (1024 * 1024 * 1024);
                ramEst.textContent = '~' + ramGb.toFixed(1) + ' GB RAM';
                metaRow.appendChild(ramEst);
            }
            info.appendChild(metaRow);

            // Filename
            var fname = document.createElement('div');
            fname.className = 'quant-filename';
            fname.textContent = f.filename;
            info.appendChild(fname);
            row.appendChild(info);

            // Action area
            var actions = document.createElement('div');
            actions.className = 'quant-actions';

            if (downloadedSet[f.filename]) {
                // Already downloaded — show badge + use button
                var badge = document.createElement('span');
                badge.className = 'downloaded-badge';
                badge.textContent = 'Downloaded';
                actions.appendChild(badge);

                var useBtn = document.createElement('button');
                useBtn.className = 'use-model-btn';
                useBtn.textContent = 'Use';
                useBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    showRoleModal(f.filename, downloadedSet[f.filename]);
                });
                actions.appendChild(useBtn);
            } else {
                var btn = document.createElement('button');
                btn.className = 'download-btn';
                btn.textContent = 'Download';
                btn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    doDownload(repoId, f.filename, btn, actions);
                });
                actions.appendChild(btn);
            }

            row.appendChild(actions);
            bodyEl.appendChild(row);
        });
    } catch (err) {
        bodyEl.innerHTML = '<div class="empty-state">Error: ' + err.message + '</div>';
    }
}

async function doDownload(repoId, filename, btn, actionsEl) {
    btn.disabled = true;
    btn.textContent = 'Downloading...';
    btn.className = 'download-btn downloading';

    try {
        var res = await fetch('/api/hub/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, filename: filename }),
        });
        var data = await res.json();
        if (data.error) { btn.textContent = 'Error'; btn.className = 'download-btn error'; return; }

        // Replace button with downloaded badge + use button
        actionsEl.textContent = '';
        var badge = document.createElement('span');
        badge.className = 'downloaded-badge';
        badge.textContent = 'Downloaded';
        actionsEl.appendChild(badge);

        var useBtn = document.createElement('button');
        useBtn.className = 'use-model-btn';
        useBtn.textContent = 'Use';
        useBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            showRoleModal(filename, data.path);
        });
        actionsEl.appendChild(useBtn);

        downloadedSet[filename] = data.path;
        await refreshLocalModels();
        refreshRoleSelectors();

        // Auto-show role picker
        showRoleModal(filename, data.path);
    } catch (err) { btn.textContent = 'Error'; btn.className = 'download-btn error'; }
}


// ============================================================
//  ROLE PICKER MODAL (assign model to a role after download)
// ============================================================

function showRoleModal(filename, path) {
    lastAssignedModel = 'gguf::' + path;
    document.getElementById('modal-model-name').textContent = filename;
    document.getElementById('role-modal').classList.remove('hidden');
}

function closeRoleModal() {
    document.getElementById('role-modal').classList.add('hidden');
    lastAssignedModel = null;
}

function assignRole(role) {
    if (!lastAssignedModel) return;
    if (role === 'all') {
        ['orch', 'sub', 'ref'].forEach(function(r) { setRoleValue(r, lastAssignedModel); });
    } else {
        setRoleValue(role, lastAssignedModel);
    }
    closeRoleModal();

    // Auto-expand model config and switch to Run tab
    var body = document.getElementById('model-config-body');
    if (body.classList.contains('hidden')) toggleModelConfig();
    navTo('run');
}

function setRoleValue(role, val) {
    var sel = document.querySelector('.model-select[data-role="' + role + '"]');
    if (sel) {
        var exists = Array.from(sel.options).some(function(o) { return o.value === val; });
        if (exists) sel.value = val;
    }
}


// ============================================================
//  MY MODELS
// ============================================================

async function loadMyModelsDownloaded() {
    var list = document.getElementById('local-models-list');
    list.textContent = 'Loading...';
    await refreshLocalModels();
    list.textContent = '';

    if (!localGgufModels.length) {
        list.innerHTML = '<div class="empty-state">No downloaded GGUF models yet.<br>Go to <a href="#" onclick="navTo(\'discover\'); return false;">Discover</a> to search and download models.</div>';
        return;
    }

    localGgufModels.forEach(function(m) {
        var item = document.createElement('div');
        item.className = 'local-model-item';

        var info = document.createElement('div');
        info.className = 'local-model-info';
        var name = document.createElement('div');
        name.className = 'local-model-name';
        name.textContent = m.filename;
        info.appendChild(name);
        var meta = document.createElement('div');
        meta.className = 'local-model-meta';
        meta.textContent = m.repo_id + ' \u00b7 ' + m.size_display;
        info.appendChild(meta);
        var path = document.createElement('div');
        path.className = 'local-model-path';
        path.textContent = m.path;
        path.title = 'Click to copy path';
        path.addEventListener('click', function() {
            navigator.clipboard.writeText(m.path);
            path.textContent = 'Copied!';
            setTimeout(function() { path.textContent = m.path; }, 1500);
        });
        info.appendChild(path);

        var actions = document.createElement('div');
        actions.className = 'local-model-actions';

        var useBtn = document.createElement('button');
        useBtn.className = 'use-model-btn';
        useBtn.textContent = 'Use';
        useBtn.addEventListener('click', function() {
            showRoleModal(m.filename, m.path);
        });
        actions.appendChild(useBtn);

        var delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', function() {
            if (!confirm('Delete ' + m.filename + '?')) return;
            fetch('/api/hub/local?path=' + encodeURIComponent(m.path), { method: 'DELETE' })
                .then(function() {
                    item.remove();
                    delete downloadedSet[m.filename];
                    refreshLocalModels().then(refreshRoleSelectors);
                });
        });
        actions.appendChild(delBtn);

        item.appendChild(info);
        item.appendChild(actions);
        list.appendChild(item);
    });
}

async function loadMyModelsOllama() {
    var list = document.getElementById('ollama-models-list');
    list.textContent = 'Loading Ollama models...';
    try {
        var res = await fetch('/api/ollama/models');
        var data = await res.json();
        list.textContent = '';

        if (data.error) {
            list.innerHTML = '<div class="empty-state">Could not connect to Ollama.<br>Make sure it is running: <code>ollama serve</code></div>';
            return;
        }
        var models = data.models || [];
        if (!models.length) {
            list.innerHTML = '<div class="empty-state">No models pulled in Ollama yet.<br>Run: <code>ollama pull llama3</code></div>';
            return;
        }
        models.forEach(function(m) {
            var item = document.createElement('div');
            item.className = 'local-model-item';
            var info = document.createElement('div');
            info.className = 'local-model-info';
            var name = document.createElement('div');
            name.className = 'local-model-name';
            name.textContent = m.name;
            info.appendChild(name);
            if (m.size) {
                var meta = document.createElement('div');
                meta.className = 'local-model-meta';
                meta.textContent = formatBytes(m.size);
                info.appendChild(meta);
            }
            item.appendChild(info);

            var actions = document.createElement('div');
            actions.className = 'local-model-actions';
            var useBtn = document.createElement('button');
            useBtn.className = 'use-model-btn';
            useBtn.textContent = 'Use';
            useBtn.addEventListener('click', function() {
                lastAssignedModel = 'ollama::' + m.name;
                document.getElementById('modal-model-name').textContent = m.name;
                document.getElementById('role-modal').classList.remove('hidden');
            });
            actions.appendChild(useBtn);
            item.appendChild(actions);

            list.appendChild(item);
        });
    } catch (err) { list.textContent = 'Error: ' + err.message; }
}


// ============================================================
//  NAVIGATION
// ============================================================

document.querySelectorAll('.nav-btn').forEach(function(btn) {
    btn.addEventListener('click', function() { navTo(btn.dataset.view); });
});

function navTo(viewName) {
    document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.view').forEach(function(v) { v.classList.remove('active'); });
    var btn = document.querySelector('.nav-btn[data-view="' + viewName + '"]');
    if (btn) btn.classList.add('active');
    document.getElementById('view-' + viewName).classList.add('active');

    if (viewName === 'discover') loadTrending();
    if (viewName === 'my-models') loadMyModelsDownloaded();
    if (viewName === 'sessions') loadSessions();
    if (viewName === 'config') loadConfig();
}

document.querySelectorAll('.my-models-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
        document.querySelectorAll('.my-models-tab').forEach(function(t) { t.classList.remove('active'); });
        document.querySelectorAll('.my-models-content').forEach(function(c) { c.classList.remove('active'); });
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        if (tab.dataset.tab === 'downloaded') loadMyModelsDownloaded();
        if (tab.dataset.tab === 'ollama-tab') loadMyModelsOllama();
    });
});


// ============================================================
//  RUN
// ============================================================

async function startRun() {
    var objective = document.getElementById('objective').value.trim();
    if (!objective) return;
    var output = document.getElementById('run-output');
    var btn = document.getElementById('btn-run');
    var useWs = document.getElementById('use-ws').checked;
    output.textContent = '';
    btn.disabled = true;
    btn.textContent = 'Running...';
    if (useWs) { await startWebSocketRun(objective, output); }
    else       { await startHttpRun(objective, output); }
    btn.disabled = false;
    btn.textContent = 'Run';
}

async function startHttpRun(objective, output) {
    addEvent(output, 'session', 'Starting: ' + objective);
    addEvent(output, 'cost', 'Processing with local models... this may take a minute. Enable "Stream live" for real-time updates.');
    try {
        var body = {
            objective: objective,
            orchestrator: getRoleConfig('orch'),
            sub_agent: getRoleConfig('sub'),
            refiner: getRoleConfig('ref'),
        };
        var res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        var data = await res.json();
        if (data.error) { addEvent(output, 'error', 'Error: ' + data.error); return; }
        addEvent(output, 'complete', 'Session complete!');
        if (data.exchanges) {
            data.exchanges.forEach(function(ex, i) {
                addEvent(output, 'subtask', 'Task ' + (i+1) + ': ' + (ex.subtask ? ex.subtask.description : ex.prompt));
                addEvent(output, 'result', cleanResult(ex.result || ''));
            });
        }
        if (data.synthesis) addEvent(output, 'complete', cleanResult(data.synthesis));
    } catch (err) { addEvent(output, 'error', 'Error: ' + err.message); }
}

async function startWebSocketRun(objective, output) {
    var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var ws;
    try {
        ws = new WebSocket(protocol + '//' + location.host + '/ws/run');
    } catch (e) {
        addEvent(output, 'cost', 'WebSocket not available, falling back to HTTP...');
        await startHttpRun(objective, output);
        return;
    }

    var connected = false;
    var done = false;

    ws.onopen = function() {
        connected = true;
        ws.send(JSON.stringify({
            objective: objective,
            orchestrator: getRoleConfig('orch'),
            sub_agent: getRoleConfig('sub'),
            refiner: getRoleConfig('ref'),
        }));
        addEvent(output, 'session', 'Connected. Running: ' + objective);
    };
    ws.onmessage = function(event) {
        var msg;
        try { msg = JSON.parse(event.data); } catch(e) { return; }
        switch (msg.type) {
            case 'session_started':   addEvent(output, 'session', 'Session started: orchestrating...'); break;
            case 'subtask_created':   addEvent(output, 'subtask', 'Sub-task: ' + msg.data.subtask); break;
            case 'subtask_completed': addEvent(output, 'result', cleanResult(msg.data.result_preview)); break;
            case 'cost_incurred':     addEvent(output, 'cost', msg.data.phase + ' (' + msg.data.model + ') ' + msg.data.input_tokens + '/' + msg.data.output_tokens + ' tokens'); break;
            case 'objective_completed': addEvent(output, 'complete', 'Objective complete'); break;
            case 'output_generated':  addEvent(output, 'session', 'Refiner output generated'); break;
            case 'session_completed': addEvent(output, 'complete', 'Done!'); done = true; break;
            case 'complete':
                done = true;
                addEvent(output, 'complete', 'Session complete!');
                if (msg.data && msg.data.synthesis) addEvent(output, 'complete', cleanResult(msg.data.synthesis));
                break;
            case 'error': addEvent(output, 'error', msg.data.error); done = true; break;
        }
    };
    ws.onerror = function() {
        if (!connected) {
            // Connection never opened — fall back to HTTP
            addEvent(output, 'cost', 'WebSocket connection failed, falling back to HTTP...');
            startHttpRun(objective, output);
        } else if (!done) {
            addEvent(output, 'error', 'WebSocket error');
        }
    };
    ws.onclose = function() {
        if (!done && connected) addEvent(output, 'session', 'Connection closed');
    };
}

function addEvent(container, type, text) {
    var div = document.createElement('div');
    div.className = 'event event-' + type;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}


// ============================================================
//  SESSIONS + CONFIG
// ============================================================

async function loadSessions() {
    var list = document.getElementById('sessions-list');
    list.textContent = 'Loading...';
    try {
        var res = await fetch('/api/sessions');
        var sessions = await res.json();
        list.textContent = '';
        if (!sessions.length) { list.textContent = 'No sessions yet.'; return; }
        sessions.forEach(function(s) {
            var item = document.createElement('div');
            item.className = 'session-item';
            item.addEventListener('click', function() { showSession(s.id); });
            var info = document.createElement('div');
            info.className = 'session-info';
            var obj = document.createElement('div');
            obj.className = 'session-objective';
            obj.textContent = s.objective.substring(0, 80);
            info.appendChild(obj);
            var meta = document.createElement('div');
            meta.className = 'session-meta';
            var status = document.createElement('span');
            status.className = 'status-' + s.status;
            status.textContent = s.status;
            meta.appendChild(status);
            meta.appendChild(document.createTextNode(' \u00b7 ' + s.exchange_count + ' tasks \u00b7 ' + new Date(s.created_at).toLocaleString()));
            info.appendChild(meta);
            item.appendChild(info);
            list.appendChild(item);
        });
    } catch (err) { list.textContent = 'Error: ' + err.message; }
}

async function showSession(id) {
    var detail = document.getElementById('session-detail');
    detail.classList.remove('hidden');
    detail.textContent = 'Loading...';
    try {
        var res = await fetch('/api/sessions/' + id);
        var s = await res.json();
        detail.textContent = '';
        var title = document.createElement('h3'); title.textContent = s.objective; detail.appendChild(title);
        (s.exchanges || []).forEach(function(ex, i) {
            var t = document.createElement('div'); t.className = 'event event-subtask';
            t.textContent = 'Task ' + (i+1) + ': ' + (ex.subtask ? ex.subtask.description : ex.prompt);
            detail.appendChild(t);
            var r = document.createElement('div'); r.className = 'event event-result';
            r.textContent = (ex.result || '').substring(0, 500);
            detail.appendChild(r);
        });
        if (s.synthesis) {
            var syn = document.createElement('div'); syn.className = 'event event-complete';
            syn.textContent = s.synthesis.substring(0, 1000);
            detail.appendChild(syn);
        }
    } catch (err) { detail.textContent = 'Error: ' + err.message; }
}

async function searchSessions() {
    var query = document.getElementById('search-input').value.trim();
    if (!query) return loadSessions();
    var list = document.getElementById('sessions-list');
    list.textContent = 'Searching...';
    try {
        var res = await fetch('/api/sessions/search/' + encodeURIComponent(query));
        var sessions = await res.json();
        list.textContent = '';
        if (!sessions.length) { list.textContent = 'No results.'; return; }
        sessions.forEach(function(s) {
            var item = document.createElement('div'); item.className = 'session-item';
            item.addEventListener('click', function() { showSession(s.id); });
            var info = document.createElement('div'); info.className = 'session-info';
            var obj = document.createElement('div'); obj.className = 'session-objective';
            obj.textContent = s.objective.substring(0, 80); info.appendChild(obj);
            item.appendChild(info); list.appendChild(item);
        });
    } catch (err) { list.textContent = 'Error: ' + err.message; }
}

async function loadConfig() {
    try {
        var res = await fetch('/api/config');
        var config = await res.json();
        document.getElementById('config-output').textContent = JSON.stringify(config, null, 2);
    } catch (err) { document.getElementById('config-output').textContent = 'Error: ' + err.message; }
}


// ============================================================
//  HELPERS
// ============================================================

function cleanResult(text) {
    // If the text is raw JSON from a tool call, extract the content
    if (!text) return '';
    var trimmed = text.trim();
    // Try to extract content from JSON tool calls like {"name":"output_synthesis","parameters":{"content":"..."}}
    if (trimmed.indexOf('"name"') !== -1 && trimmed.indexOf('"parameters"') !== -1) {
        try {
            var obj = JSON.parse(trimmed);
            if (obj && obj.parameters && obj.parameters.content) return obj.parameters.content;
        } catch(e) {}
        // Try to find JSON embedded in text
        var jsonStart = trimmed.indexOf('{"name"');
        if (jsonStart > 0) {
            var before = trimmed.substring(0, jsonStart).trim();
            try {
                var obj2 = JSON.parse(trimmed.substring(jsonStart));
                if (obj2 && obj2.parameters && obj2.parameters.content) return obj2.parameters.content;
            } catch(e2) {}
            return before; // Return the text before the JSON
        }
    }
    return text;
}

function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

function formatBytes(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + ' GB';
    if (bytes >= 1048576)    return (bytes / 1048576).toFixed(0) + ' MB';
    return bytes + ' B';
}
