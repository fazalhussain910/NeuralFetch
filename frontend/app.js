/**
 * NeuralFetch — Frontend Application Logic
 * Supports standalone deployment on Netlify / Vercel with configurable API endpoint.
 */

// State
const state = {
  apiBaseUrl: localStorage.getItem('neuralfetch_api_url') || detectDefaultApiUrl(),
  health: null,
  currentQueryResponse: null,
  timerInterval: null,
  startTime: null,
};

function detectDefaultApiUrl() {
  // If served from FastAPI (e.g. localhost:8000), use origin; else fallback to http://localhost:8000
  if (window.location.port === '8000') {
    return window.location.origin;
  }
  return 'http://localhost:8000';
}

// DOM Elements
const elements = {
  healthIndicator: document.getElementById('health-indicator'),
  healthStatusText: document.getElementById('health-status-text'),
  btnHealthCheck: document.getElementById('btn-health-check'),
  modelBadge: document.getElementById('model-badge'),
  activeModelText: document.getElementById('active-model-text'),
  
  searchForm: document.getElementById('search-form'),
  queryInput: document.getElementById('query-input'),
  targetUrlInput: document.getElementById('target-url-input'),
  submitBtn: document.getElementById('submit-btn'),
  promptChips: document.querySelectorAll('.prompt-chip'),
  
  workflowSection: document.getElementById('workflow-section'),
  workflowTimer: document.getElementById('workflow-timer'),
  stepQA: document.getElementById('step-node-qa'),
  stepWN: document.getElementById('step-node-wn'),
  stepRI: document.getElementById('step-node-ri'),
  stepSYN: document.getElementById('step-node-syn'),
  
  resultsSection: document.getElementById('results-section'),
  answerBody: document.getElementById('answer-body'),
  metaTime: document.getElementById('meta-time'),
  metaChunks: document.getElementById('meta-chunks'),
  metaDevice: document.getElementById('meta-device'),
  btnCopyAnswer: document.getElementById('btn-copy-answer'),
  copyBtnText: document.getElementById('copy-btn-text'),
  btnShowChunks: document.getElementById('btn-show-chunks'),
  retrievedCount: document.getElementById('retrieved-count'),
  citationsContainer: document.getElementById('citations-container'),
  citationsCount: document.getElementById('citations-count'),
  scrapedUrlsList: document.getElementById('scraped-urls-list'),
  
  // Modals
  toolsModal: document.getElementById('tools-modal'),
  btnOpenTools: document.getElementById('btn-open-tools'),
  btnCloseTools: document.getElementById('btn-close-tools'),
  tabDirectIndex: document.getElementById('tab-direct-index'),
  tabTestLLM: document.getElementById('tab-test-llm'),
  panelDirectIndex: document.getElementById('panel-direct-index'),
  panelTestLLM: document.getElementById('panel-test-llm'),
  toolIndexUrl: document.getElementById('tool-index-url'),
  btnRunIndexUrl: document.getElementById('btn-run-index-url'),
  toolIndexResult: document.getElementById('tool-index-result'),
  toolLlmProvider: document.getElementById('tool-llm-provider'),
  toolLlmPrompt: document.getElementById('tool-llm-prompt'),
  btnRunTestLlm: document.getElementById('btn-run-test-llm'),
  toolLlmResult: document.getElementById('tool-llm-result'),
  
  settingsModal: document.getElementById('settings-modal'),
  btnOpenSettings: document.getElementById('btn-open-settings'),
  btnCloseSettings: document.getElementById('btn-close-settings'),
  settingsApiUrl: document.getElementById('settings-api-url'),
  btnSaveSettings: document.getElementById('btn-save-settings'),
  btnResetApiUrl: document.getElementById('btn-reset-api-url'),
  
  chunksModal: document.getElementById('chunks-modal'),
  btnCloseChunks: document.getElementById('btn-close-chunks'),
  chunksList: document.getElementById('chunks-list'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  
  setupEventListeners();
  checkBackendHealth();
  elements.settingsApiUrl.value = state.apiBaseUrl;
});

// Event Listeners
function setupEventListeners() {
  // Query Submission
  elements.searchForm.addEventListener('submit', handleQuerySubmit);

  // Quick Prompt Chips
  elements.promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.queryInput.value = chip.textContent.trim();
      elements.queryInput.focus();
      handleQuerySubmit(new Event('submit'));
    });
  });

  // Health check button
  elements.btnHealthCheck.addEventListener('click', checkBackendHealth);

  // Copy Answer
  elements.btnCopyAnswer.addEventListener('click', copyAnswerToClipboard);

  // Show Raw Vectors Modal
  elements.btnShowChunks.addEventListener('click', openChunksModal);
  elements.btnCloseChunks.addEventListener('click', () => elements.chunksModal.classList.add('hidden'));

  // Tools Modal
  elements.btnOpenTools.addEventListener('click', () => elements.toolsModal.classList.remove('hidden'));
  elements.btnCloseTools.addEventListener('click', () => elements.toolsModal.classList.add('hidden'));
  
  elements.tabDirectIndex.addEventListener('click', () => switchToolsTab('directIndex'));
  elements.tabTestLLM.addEventListener('click', () => switchToolsTab('testLLM'));
  elements.btnRunIndexUrl.addEventListener('click', runDirectIndexTool);
  elements.btnRunTestLlm.addEventListener('click', runTestLlmTool);

  // Settings Modal
  elements.btnOpenSettings.addEventListener('click', () => elements.settingsModal.classList.remove('hidden'));
  elements.btnCloseSettings.addEventListener('click', () => elements.settingsModal.classList.add('hidden'));
  elements.btnSaveSettings.addEventListener('click', saveSettings);
  elements.btnResetApiUrl.addEventListener('click', resetApiUrl);

  // Close modals on Backdrop or Escape key
  [elements.toolsModal, elements.settingsModal, elements.chunksModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      elements.toolsModal.classList.add('hidden');
      elements.settingsModal.classList.add('hidden');
      elements.chunksModal.classList.add('hidden');
    }
  });
}

// Check Backend Health
async function checkBackendHealth() {
  elements.healthIndicator.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
  elements.healthStatusText.textContent = 'Connecting...';

  try {
    const res = await fetch(`${state.apiBaseUrl}/api/v1/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const data = await res.json();
    state.health = data;

    elements.healthIndicator.className = 'w-2 h-2 rounded-full bg-emerald-400';
    elements.healthStatusText.textContent = 'Online';

    // Update Model Badge
    const model = data.llm_model || 'Unknown';
    const provider = (data.llm_provider || 'ollama').toUpperCase();
    elements.activeModelText.textContent = `${provider} • ${model}`;
    elements.modelBadge.classList.remove('hidden');

    if (window.lucide) window.lucide.createIcons();
  } catch (err) {
    console.warn('Backend connection error:', err);
    elements.healthIndicator.className = 'w-2 h-2 rounded-full bg-rose-500';
    elements.healthStatusText.textContent = 'Offline (Check URL)';
  }
}

// Handle Query Submission
async function handleQuerySubmit(e) {
  e.preventDefault();
  const question = elements.queryInput.value.trim();
  if (!question) return;

  const targetUrl = elements.targetUrlInput.value.trim();
  const targetUrls = targetUrl ? [targetUrl] : undefined;

  // UI state: Start loading
  setLoadingState(true);

  try {
    const payload = { question, target_urls: targetUrls };
    const response = await fetch(`${state.apiBaseUrl}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    state.currentQueryResponse = data;

    // Display Results
    renderResults(data);

    // Celebration
    if (window.confetti) {
      window.confetti({
        particleCount: 40,
        spread: 60,
        origin: { y: 0.75 },
        colors: ['#06b6d4', '#8b5cf6', '#10b981']
      });
    }
  } catch (err) {
    alert(`Agent Error: ${err.message}\n\nPlease verify that your NeuralFetch backend server is running at:\n${state.apiBaseUrl}`);
  } finally {
    setLoadingState(false);
  }
}

// Loading and Workflow Animation States
function setLoadingState(isLoading) {
  elements.submitBtn.disabled = isLoading;
  
  if (isLoading) {
    elements.submitBtn.innerHTML = `
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
      </svg>
      <span>Researching...</span>
    `;

    elements.workflowSection.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');

    // Start Workflow Timer
    state.startTime = Date.now();
    elements.workflowTimer.textContent = '0.0s';
    clearInterval(state.timerInterval);
    state.timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - state.startTime) / 1000).toFixed(1);
      elements.workflowTimer.textContent = `${elapsed}s`;
    }, 100);

    // Simulate animated node progression while request is processing
    animateWorkflowSteps();
  } else {
    elements.submitBtn.innerHTML = `
      <i data-lucide="sparkles" class="w-4 h-4"></i>
      <span>Launch Agent</span>
    `;
    clearInterval(state.timerInterval);
    if (window.lucide) window.lucide.createIcons();
  }
}

function animateWorkflowSteps() {
  const steps = [elements.stepQA, elements.stepWN, elements.stepRI, elements.stepSYN];
  steps.forEach(s => s.className = 'step-card bg-brand-surface p-3.5 rounded-xl border border-brand-border/60 transition-all flex flex-col space-y-1.5');

  // Step 1: QA active
  elements.stepQA.classList.add('active');

  setTimeout(() => {
    if (!elements.submitBtn.disabled) return;
    elements.stepQA.classList.remove('active');
    elements.stepQA.classList.add('completed');
    elements.stepWN.classList.add('active');
  }, 1200);

  setTimeout(() => {
    if (!elements.submitBtn.disabled) return;
    elements.stepWN.classList.remove('active');
    elements.stepWN.classList.add('completed');
    elements.stepRI.classList.add('active');
  }, 3500);

  setTimeout(() => {
    if (!elements.submitBtn.disabled) return;
    elements.stepRI.classList.remove('active');
    elements.stepRI.classList.add('completed');
    elements.stepSYN.classList.add('active');
  }, 5500);
}

// Render Results
function renderResults(data) {
  // Mark all steps completed
  [elements.stepQA, elements.stepWN, elements.stepRI, elements.stepSYN].forEach(s => {
    s.classList.remove('active');
    s.classList.add('completed');
  });

  // Meta stats
  elements.metaTime.textContent = `${data.execution_time_seconds}s`;
  elements.metaChunks.textContent = `${data.indexed_chunks_count} chunks indexed`;
  elements.metaDevice.textContent = data.device_used || 'ROCm / CPU';
  elements.retrievedCount.textContent = (data.retrieved_chunks || []).length;

  // Render Markdown Answer
  const rawAnswer = data.answer || 'No response generated.';
  const htmlContent = marked.parse(rawAnswer);
  const cleanHtml = DOMPurify.sanitize(htmlContent);
  elements.answerBody.innerHTML = cleanHtml;

  // Render Citations
  renderCitations(data.citations || []);

  // Render Scraped Trajectory
  renderScrapedTrajectory(data.scraped_urls || []);

  // Show results
  elements.resultsSection.classList.remove('hidden');
  elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  if (window.lucide) window.lucide.createIcons();
}

function renderCitations(citations) {
  elements.citationsCount.textContent = `${citations.length} sources cited`;
  elements.citationsContainer.innerHTML = '';

  if (citations.length === 0) {
    elements.citationsContainer.innerHTML = `
      <div class="col-span-full p-4 rounded-xl bg-brand-surface border border-brand-border text-center text-xs text-slate-400">
        No external web citations were returned for this query.
      </div>
    `;
    return;
  }

  citations.forEach((cit, idx) => {
    const card = document.createElement('div');
    card.className = 'citation-card p-4 rounded-xl flex flex-col justify-between space-y-3';
    
    // Extract clean domain
    let domain = 'Source';
    try {
      domain = new URL(cit.url).hostname.replace('www.', '');
    } catch {}

    const scorePercent = Math.min(Math.round((cit.score || 0.7) * 100), 100);

    card.innerHTML = `
      <div class="space-y-1.5">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-semibold text-brand-cyan uppercase tracking-wider flex items-center space-x-1">
            <span>[${idx + 1}]</span>
            <span class="truncate max-w-[160px]">${domain}</span>
          </span>
          <span class="text-[11px] font-mono text-slate-400">${scorePercent}% Match</span>
        </div>
        <p class="text-xs text-slate-300 line-clamp-3 leading-relaxed">
          "${cit.snippet}"
        </p>
      </div>

      <div class="pt-2 border-t border-brand-border/60 flex items-center justify-between text-xs">
        <div class="w-24 bg-brand-border/50 h-1.5 rounded-full overflow-hidden">
          <div class="bg-gradient-to-r from-brand-cyan to-brand-500 h-full rounded-full" style="width: ${scorePercent}%"></div>
        </div>
        <a href="${cit.url}" target="_blank" rel="noopener noreferrer" class="text-brand-400 hover:text-white flex items-center space-x-1 transition font-medium">
          <span>Open Link</span>
          <i data-lucide="external-link" class="w-3 h-3"></i>
        </a>
      </div>
    `;
    elements.citationsContainer.appendChild(card);
  });
}

function renderScrapedTrajectory(urls) {
  elements.scrapedUrlsList.innerHTML = '';
  if (urls.length === 0) {
    elements.scrapedUrlsList.innerHTML = `<li>No external URLs scraped.</li>`;
    return;
  }

  urls.forEach(u => {
    const li = document.createElement('li');
    li.className = 'truncate hover:text-brand-cyan transition cursor-pointer';
    li.title = u;
    li.innerHTML = `<a href="${u}" target="_blank" rel="noopener noreferrer">${u}</a>`;
    elements.scrapedUrlsList.appendChild(li);
  });
}

// Chunks / Vectors Modal
function openChunksModal() {
  const chunks = state.currentQueryResponse?.retrieved_chunks || [];
  elements.chunksList.innerHTML = '';

  if (chunks.length === 0) {
    elements.chunksList.innerHTML = `<p class="text-slate-400 p-4 text-center">No retrieved chunks stored in current response state.</p>`;
  } else {
    chunks.forEach((c, i) => {
      const item = document.createElement('div');
      item.className = 'p-4 rounded-xl bg-brand-surface border border-brand-border space-y-2';
      item.innerHTML = `
        <div class="flex items-center justify-between text-slate-400">
          <span class="font-bold text-white">Chunk #${i + 1} • ${c.title || 'Untitled'}</span>
          <span class="font-mono text-brand-cyan">Score: ${Number(c.score).toFixed(4)}</span>
        </div>
        <p class="text-slate-300 font-mono text-[11px] whitespace-pre-wrap leading-relaxed">${c.text}</p>
        <div class="text-[10px] text-slate-500 truncate">URL: ${c.url}</div>
      `;
      elements.chunksList.appendChild(item);
    });
  }

  elements.chunksModal.classList.remove('hidden');
}

// Copy Answer
async function copyAnswerToClipboard() {
  const answerText = state.currentQueryResponse?.answer || elements.answerBody.innerText;
  try {
    await navigator.clipboard.writeText(answerText);
    elements.copyBtnText.textContent = 'Copied!';
    setTimeout(() => { elements.copyBtnText.textContent = 'Copy'; }, 2000);
  } catch (e) {
    console.error('Clipboard copy failed:', e);
  }
}

// Tools Modal Tab Switching
function switchToolsTab(tab) {
  if (tab === 'directIndex') {
    elements.tabDirectIndex.className = 'pb-2 text-brand-400 border-b-2 border-brand-400';
    elements.tabTestLLM.className = 'pb-2 text-slate-400 hover:text-slate-200';
    elements.panelDirectIndex.classList.remove('hidden');
    elements.panelTestLLM.classList.add('hidden');
  } else {
    elements.tabTestLLM.className = 'pb-2 text-brand-400 border-b-2 border-brand-400';
    elements.tabDirectIndex.className = 'pb-2 text-slate-400 hover:text-slate-200';
    elements.panelTestLLM.classList.remove('hidden');
    elements.panelDirectIndex.classList.add('hidden');
  }
}

// Direct URL Indexing Tool
async function runDirectIndexTool() {
  const url = elements.toolIndexUrl.value.trim();
  if (!url) {
    alert('Please provide a URL to scrape and index.');
    return;
  }

  elements.btnRunIndexUrl.disabled = true;
  elements.btnRunIndexUrl.textContent = 'Scraping and Chunking...';
  elements.toolIndexResult.classList.add('hidden');

  try {
    const res = await fetch(`${state.apiBaseUrl}/api/v1/index-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    elements.toolIndexResult.classList.remove('hidden');
    elements.toolIndexResult.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    elements.toolIndexResult.classList.remove('hidden');
    elements.toolIndexResult.textContent = `Error: ${err.message}`;
  } finally {
    elements.btnRunIndexUrl.disabled = false;
    elements.btnRunIndexUrl.innerHTML = `<i data-lucide="upload-cloud" class="w-4 h-4"></i><span>Scrape & Index into Qdrant</span>`;
    if (window.lucide) window.lucide.createIcons();
  }
}

// Test LLM Diagnostic Tool
async function runTestLlmTool() {
  const prompt = elements.toolLlmPrompt.value.trim();
  const provider = elements.toolLlmProvider.value || undefined;

  elements.btnRunTestLlm.disabled = true;
  elements.btnRunTestLlm.textContent = 'Executing test inference...';
  elements.toolLlmResult.classList.add('hidden');

  try {
    const res = await fetch(`${state.apiBaseUrl}/api/v1/test-llm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, provider })
    });
    const data = await res.json();
    elements.toolLlmResult.classList.remove('hidden');
    elements.toolLlmResult.textContent = `[${data.provider} • ${data.model} (${data.execution_time_seconds}s)]:\n\n${data.response}`;
  } catch (err) {
    elements.toolLlmResult.classList.remove('hidden');
    elements.toolLlmResult.textContent = `Diagnostic Error: ${err.message}`;
  } finally {
    elements.btnRunTestLlm.disabled = false;
    elements.btnRunTestLlm.innerHTML = `<i data-lucide="zap" class="w-4 h-4"></i><span>Send Diagnostic Request</span>`;
    if (window.lucide) window.lucide.createIcons();
  }
}

// Settings
function saveSettings() {
  const newUrl = elements.settingsApiUrl.value.trim().replace(/\/+$/, '');
  if (!newUrl) {
    alert('Please enter a valid API URL.');
    return;
  }
  state.apiBaseUrl = newUrl;
  localStorage.setItem('neuralfetch_api_url', newUrl);
  elements.settingsModal.classList.add('hidden');
  checkBackendHealth();
}

function resetApiUrl() {
  const def = detectDefaultApiUrl();
  elements.settingsApiUrl.value = def;
  saveSettings();
}
