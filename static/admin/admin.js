// State
let authHeader = '';
let ws = null;
let currentChain = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check for existing auth
    const savedAuth = localStorage.getItem('ccnim_admin_auth');
    if (savedAuth) {
        authHeader = savedAuth;
        showApp();
    }
});

function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    authHeader = 'Basic ' + btoa(username + ':' + password);
    
    // Test credentials
    fetch('/admin/status', {
        headers: { 'Authorization': authHeader }
    })
    .then(res => {
        if (res.ok) {
            localStorage.setItem('ccnim_admin_auth', authHeader);
            showApp();
        } else {
            document.getElementById('login-error').style.display = 'block';
        }
    })
    .catch(() => {
        document.getElementById('login-error').style.display = 'block';
    });
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
    loadModels();
    
    // Refresh interval
    setInterval(loadDashboard, 5000);
    setInterval(loadLogs, 5000);
    setInterval(loadKeys, 5000);
    setInterval(loadModels, 5000);
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
        .then(res => res.json())
        .then(data => {
            updateDashboard(data);
        })
        .catch(err => console.error('Error loading dashboard:', err));
}

function updateDashboard(data) {
    // Update header stats
    const agg = data.aggregate || {};
    document.getElementById('rpm-stat').textContent = `RPM: ${agg.requests_per_minute || 0}`;
    document.getElementById('success-rate-stat').textContent = `Success: ${agg.success_rate || 100}%`;
    document.getElementById('total-requests-stat').textContent = `Total: ${agg.total_requests || 0}`;
    
    // Model Chain
    const chainList = document.getElementById('model-chain-list');
    chainList.innerHTML = '';
    data.model_chain.forEach((model, index) => {
        const isPrimary = index === 0;
        const div = document.createElement('div');
        div.className = `model-item ${isPrimary ? 'primary' : 'fallback'}`;
        div.innerHTML = `
            <span class="model-name">${isPrimary ? '🔴 ' : ''}${model}</span>
            <span class="model-pos">#${index + 1}</span>
        `;
        chainList.appendChild(div);
    });
    
    // Model Health
    const healthList = document.getElementById('model-health-list');
    healthList.innerHTML = '';
    for (const [model, health] of Object.entries(data.health || {})) {
        const div = document.createElement('div');
        div.className = 'health-item';
        
        const rate = health.success_rate || 0;
        const healthClass = rate > 90 ? '' : rate > 70 ? 'warning' : 'danger';
        
        div.innerHTML = `
            <div class="health-item-header">
                <span class="model-name">${model}</span>
                <span class="health-stats">${health.success}/${health.total} (${rate}%)</span>
            </div>
            <div class="health-bar">
                <div class="health-fill ${healthClass}" style="width: ${rate}%"></div>
            </div>
            ${health.recent_errors && health.recent_errors.length ? 
                `<div class="error-list">${health.recent_errors.slice(0, 2).map(e => `• ${e.substring(0, 50)}...`).join('<br>')}</div>` : ''}
        `;
        healthList.appendChild(div);
    }
    
    // Keys Summary
    const keysSummary = document.getElementById('keys-summary');
    keysSummary.innerHTML = '';
    data.keys.forEach(key => {
        const div = document.createElement('div');
        div.className = 'keys-summary-item';
        div.innerHTML = `
            <span class="key-badge">${key.key}</span>
            <span class="${key.blocked ? 'status-blocked' : 'status-active'}">
                ${key.blocked ? '🔴 Blocked' : `🟢 ${key.remaining_requests} left`}
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

// Models
function loadModels() {
    fetch('/admin/chain', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            currentChain = data.chain || [];
            renderChainList(currentChain);
        })
        .catch(err => console.error('Error loading chain:', err));
}

function renderChainList(chain) {
    const list = document.getElementById('chain-list');
    list.innerHTML = '';
    
    chain.forEach((model, index) => {
        const li = document.createElement('li');
        li.draggable = true;
        li.dataset.model = model;
        li.dataset.index = index;
        li.innerHTML = `
            <span class="chain-number">${index + 1}</span>
            <span class="chain-model-name">${model}</span>
        `;
        
        li.addEventListener('dragstart', handleDragStart);
        li.addEventListener('dragover', handleDragOver);
        li.addEventListener('drop', handleDrop);
        li.addEventListener('dragenter', handleDragEnter);
        li.addEventListener('dragleave', handleDragLeave);
        
        list.appendChild(li);
    });
}

let draggedItem = null;

function handleDragStart(e) {
    draggedItem = this;
    e.dataTransfer.effectAllowed = 'move';
    this.style.opacity = '0.5';
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleDragEnter(e) {
    this.style.border = '2px dashed #e94560';
}

function handleDragLeave(e) {
    this.style.border = '';
}

function handleDrop(e) {
    e.preventDefault();
    this.style.border = '';
    
    if (draggedItem !== this) {
        const fromIndex = parseInt(draggedItem.dataset.index);
        const toIndex = parseInt(this.dataset.index);
        
        // Reorder array
        const item = currentChain.splice(fromIndex, 1)[0];
        currentChain.splice(toIndex, 0, item);
        
        renderChainList(currentChain);
    }
    
    draggedItem.style.opacity = '';
    draggedItem = null;
}

function saveChainOrder() {
    fetch('/admin/chain/reorder', {
        method: 'POST',
        headers: {
            'Authorization': authHeader,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(currentChain)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'reordered') {
            alert('Model chain updated successfully!');
        } else {
            alert('Failed to update chain');
        }
    })
    .catch(err => {
        console.error('Error saving chain:', err);
        alert('Error saving chain order');
    });
}
