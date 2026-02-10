// State
let authHeader = '';
let ws = null;
let modelCatalogLoaded = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check for existing auth
    const savedAuth = localStorage.getItem('ccnim_admin_auth');
    if (savedAuth) {
        authHeader = savedAuth;
    } else {
        authHeader = '';
    }
    // Backend currently allows open admin routes; don't block UI behind login.
    showApp();
});

function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    authHeader = username || password ? ('Basic ' + btoa(username + ':' + password)) : '';
    localStorage.setItem('ccnim_admin_auth', authHeader);
    showApp();
}

function showApp() {
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('app').style.display = 'flex';
    
    // Connect WebSocket
    connectWebSocket();
    
    // Initial data fetch
    loadDashboard();
    loadLogs();
    loadKeys();
    refreshModelCatalog();
    
    // Refresh interval
    setInterval(loadDashboard, 5000);
    setInterval(loadLogs, 5000);
    setInterval(loadKeys, 5000);
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/admin/ws`;
    
    // Get auth from current connection
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log' && data.data) {
            addLogEntry(data.data);
            updateDashboardStats();
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Dashboard
function loadDashboard() {
    fetch('/admin/status', { headers: { 'Authorization': authHeader } })
        .then(res => {
            if (!res.ok) {
                throw new Error(`status ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            updateDashboard(data);
        })
        .catch(err => {
            console.error('Error loading dashboard:', err);
            const statusEl = document.getElementById('model-update-status');
            if (statusEl) {
                statusEl.textContent = 'Dashboard data unavailable. Check server logs.';
                statusEl.className = 'model-status error';
            }
        });
}

function updateDashboard(data) {
    // Update header stats
    const agg = data.aggregate || {};
    document.getElementById('rpm-stat').textContent = `RPM: ${agg.requests_per_minute || 0}`;
    document.getElementById('success-rate-stat').textContent = `Success: ${agg.success_rate || 100}%`;
    document.getElementById('total-requests-stat').textContent = `Total: ${agg.total_requests || 0}`;
    
    // Active model
    const modelView = document.getElementById('active-model-view');
    const activeModel = data.active_model || 'unknown';
    const defaultModel = data.default_model || 'unknown';
    const hasOverride = !!data.has_runtime_model_override;
    modelView.innerHTML = `
        <div class="model-item primary">
            <span class="model-name">${activeModel}</span>
            <span class="model-pos">${hasOverride ? 'runtime override' : 'from .env MODEL'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Default</span>
            <span class="settings-value">${defaultModel}</span>
        </div>
    `;
    const modelInput = document.getElementById('active-model-input');
    if (modelInput && document.activeElement !== modelInput) {
        modelInput.value = activeModel;
    }
    const statusEl = document.getElementById('model-update-status');
    if (statusEl && data.provider_error) {
        statusEl.textContent = `Provider unavailable: ${data.provider_error}`;
        statusEl.className = 'model-status error';
    }
    if (!modelCatalogLoaded) {
        refreshModelCatalog();
    }
    
    // Fallback summary
    const fallbackInfo = document.getElementById('fallback-info');
    const blockedKeys = (data.keys || []).filter(k => k.blocked).length;
    fallbackInfo.innerHTML = `
        <div class="settings-item">
            <span class="settings-label">Strategy</span>
            <span class="settings-value">API key rotation</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Total Keys</span>
            <span class="settings-value">${(data.keys || []).length}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Blocked Keys</span>
            <span class="settings-value">${blockedKeys}</span>
        </div>
    `;
    
    // Keys Summary
    const keysSummary = document.getElementById('keys-summary');
    keysSummary.innerHTML = '';
    data.keys.forEach(key => {
        const div = document.createElement('div');
        div.className = 'keys-summary-item';
        div.innerHTML = `
            <span class="key-badge">${key.key_masked || ('***' + key.key_suffix)}</span>
            <span class="${key.blocked ? 'status-blocked' : 'status-active'}">
                ${key.blocked ? `🔴 Blocked (${key.cooldown_remaining_seconds || 0}s)` : `🟢 ${key.remaining_requests || 0} left`}
            </span>
        `;
        keysSummary.appendChild(div);
    });
    
    // Settings
    const settingsInfo = document.getElementById('settings-info');
    settingsInfo.innerHTML = `
        <div class="settings-item">
            <span class="settings-label">Rate Limit</span>
            <span class="settings-value">${data.settings.rate_limit} req/${data.settings.rate_window}s</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Cooldown</span>
            <span class="settings-value">${data.settings.cooldown_seconds}s</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Max In Flight</span>
            <span class="settings-value">${data.settings.max_in_flight || '-'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Active Requests</span>
            <span class="settings-value">${(data.runtime && data.runtime.active_requests) || 0}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">HTTP Latency</span>
            <span class="settings-value">avg ${agg.avg_http_latency_ms || 0}ms / p95 ${agg.p95_http_latency_ms || 0}ms</span>
        </div>
    `;
}

function updateDashboardStats() {
    // Small update when new log comes in
    const rpmStat = document.getElementById('rpm-stat');
    const currentRPM = parseInt(rpmStat.textContent.split(':')[1]) || 0;
    rpmStat.textContent = `RPM: ${currentRPM + 1}`;
}

// Logs
function loadLogs() {
    fetch('/admin/logs?limit=200', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('logs-body');
            tbody.innerHTML = '';
            data.logs.forEach(log => addLogEntry(log));
            updateLogCount();
        })
        .catch(err => console.error('Error loading logs:', err));
}

function addLogEntry(log) {
    const tbody = document.getElementById('logs-body');
    const tr = document.createElement('tr');
    
    const time = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '';
    const statusClass = log.status === 'success' ? 'status-success' : 'status-error';
    
    tr.innerHTML = `
        <td class="log-time">${time}</td>
        <td><code>${log.model}</code></td>
        <td><code>***${log.key_suffix}</code></td>
        <td class="${statusClass}">${log.status}</td>
        <td>${log.response_time_ms ? log.response_time_ms.toFixed(0) + 'ms' : '-'}</td>
    `;
    
    tbody.insertBefore(tr, tbody.firstChild);
    
    // Keep only last 200 in DOM
    while (tbody.children.length > 200) {
        tbody.removeChild(tbody.lastChild);
    }
}

function filterLogs() {
    const modelFilter = document.getElementById('log-filter-model').value.toLowerCase();
    const keyFilter = document.getElementById('log-filter-key').value.toLowerCase();
    const statusFilter = document.getElementById('log-filter-status').value;
    
    const rows = document.querySelectorAll('#logs-body tr');
    rows.forEach(row => {
        const model = row.cells[1].textContent.toLowerCase();
        const key = row.cells[2].textContent.toLowerCase();
        const status = row.querySelector('.status-success, .status-error').textContent.toLowerCase();
        
        const matchModel = !modelFilter || model.includes(modelFilter);
        const matchKey = !keyFilter || key.includes(keyFilter);
        const matchStatus = !statusFilter || status.includes(statusFilter);
        
        row.style.display = matchModel && matchKey && matchStatus ? '' : 'none';
    });
    
    updateLogCount();
}

function updateLogCount() {
    const visible = document.querySelectorAll('#logs-body tr:not([style*="display: none"])').length;
    document.getElementById('log-count').textContent = `${visible} logs`;
}

// Keys
function loadKeys() {
    fetch('/admin/keys', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('keys-body');
            tbody.innerHTML = '';
            data.keys.forEach(key => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${key.key_masked}</code></td>
                    <td class="${key.blocked ? 'status-blocked' : 'status-active'}">
                        ${key.blocked ? 'Blocked' : 'Active'}
                    </td>
                    <td>${key.usage_count}/${key.in_flight + key.usage_count}</td>
                    <td>${key.in_flight}</td>
                    <td>
                        ${key.blocked ? 
                            `<button class="action-btn unblock" onclick="unblockKey('${key.key_suffix}')">Unblock</button>` :
                            `<button class="action-btn block" onclick="blockKey('${key.key_suffix}')">Block</button>`
                        }
                    </td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error('Error loading keys:', err));
}

function blockKey(keySuffix) {
    if (!confirm(`Block key ending in ${keySuffix}?`)) return;
    
    fetch(`/admin/keys/${keySuffix}/block`, {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => loadKeys())
    .catch(err => console.error('Error blocking key:', err));
}

function unblockKey(keySuffix) {
    fetch(`/admin/keys/${keySuffix}/unblock`, {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => loadKeys())
    .catch(err => console.error('Error unblocking key:', err));
}

function updateActiveModel() {
    const model = document.getElementById('active-model-input').value.trim();
    const persist = !!document.getElementById('persist-model-checkbox').checked;
    const statusEl = document.getElementById('model-update-status');
    if (!model) {
        statusEl.textContent = 'Model is required';
        statusEl.className = 'model-status error';
        return;
    }

    fetch('/admin/model', {
        method: 'POST',
        headers: {
            'Authorization': authHeader,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model, persist })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'updated') {
            statusEl.textContent = data.persisted ? 'Model updated and persisted to .env' : 'Runtime model updated';
            statusEl.className = 'model-status success';
            loadDashboard();
        } else {
            statusEl.textContent = data.detail || 'Failed to update model';
            statusEl.className = 'model-status error';
        }
    })
    .catch(err => {
        console.error('Error updating model:', err);
        statusEl.textContent = 'Error updating model';
        statusEl.className = 'model-status error';
    });
}

function resetActiveModel() {
    const statusEl = document.getElementById('model-update-status');
    fetch('/admin/model/reset', {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => {
        statusEl.textContent = 'Runtime override cleared';
        statusEl.className = 'model-status success';
        loadDashboard();
    })
    .catch(err => {
        console.error('Error resetting model:', err);
        statusEl.textContent = 'Error resetting model';
        statusEl.className = 'model-status error';
    });
}

function refreshModelCatalog() {
    fetch('/admin/models?limit=500', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const datalist = document.getElementById('nim-model-options');
            if (!datalist) return;
            datalist.innerHTML = '';
            (data.models || []).forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                datalist.appendChild(option);
            });
            modelCatalogLoaded = true;
        })
        .catch(err => console.error('Error loading model catalog:', err));
}
