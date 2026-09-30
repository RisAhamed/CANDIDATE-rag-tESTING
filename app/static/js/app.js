// ===== Candidate Search RAG - Frontend Application =====

(function() {
    'use strict';

    // ===== State ===== {
    const state = {
        currentTab: 'search',
        candidates: [],
        searchResults: [],
        debugMode: false,
        expandedCards: new Set()
    };

    // ===== DOM Elements =====
    const elements = {
        // Tabs
        tabButtons: document.querySelectorAll('.tab-btn'),
        tabPanels: document.querySelectorAll('.tab-panel'),

        // Search
        searchForm: document.getElementById('search-form'),
        searchInput: document.getElementById('search-input'),
        searchBtn: document.getElementById('search-btn'),
        topKSelect: document.getElementById('top-k-select'),
        debugToggle: document.getElementById('debug-toggle'),
        debugPanel: document.getElementById('debug-panel'),
        debugContent: document.getElementById('debug-content'),
        debugClose: document.getElementById('debug-close'),
        searchResults: document.getElementById('search-results'),

        // Upload - Profile
        profileUploadForm: document.getElementById('profile-upload-form'),
        profileCandidateId: document.getElementById('profile-candidate-id'),
        profileFile: document.getElementById('profile-file'),
        profileDropZone: document.getElementById('profile-drop-zone'),
        profileFileName: document.getElementById('profile-file-name'),
        profileUploadBtn: document.getElementById('profile-upload-btn'),
        profileUploadStatus: document.getElementById('profile-upload-status'),

        // Upload - Interview
        interviewUploadForm: document.getElementById('interview-upload-form'),
        interviewCandidateId: document.getElementById('interview-candidate-id'),
        interviewId: document.getElementById('interview-id'),
        interviewDate: document.getElementById('interview-date'),
        interviewStage: document.getElementById('interview-stage'),
        interviewer: document.getElementById('interviewer'),
        interviewFile: document.getElementById('interview-file'),
        interviewDropZone: document.getElementById('interview-drop-zone'),
        interviewFileName: document.getElementById('interview-file-name'),
        interviewUploadBtn: document.getElementById('interview-upload-btn'),
        interviewUploadStatus: document.getElementById('interview-upload-status'),

        // Candidates
        candidatesList: document.getElementById('candidates-list'),
        refreshCandidatesBtn: document.getElementById('refresh-candidates-btn'),

        // Modal
        candidateModal: document.getElementById('candidate-modal'),
        modalOverlay: document.getElementById('modal-overlay'),
        modalClose: document.getElementById('modal-close'),
        modalTitle: document.getElementById('modal-title'),
        modalBody: document.getElementById('modal-body'),

        // Toast
        toastContainer: document.getElementById('toast-container')
    };

    // ===== Utility Functions =====

    function formatScore(score) {
        if (score === null || score === undefined) return 'N/A';
        return (score * 100).toFixed(0) + '%';
    }

    function getScoreClass(score) {
        if (score === null || score === undefined) return 'low';
        if (score > 0.7) return 'high';
        if (score > 0.4) return 'medium';
        return 'low';
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function setLoading(button, loading) {
        const btnText = button.querySelector('.btn-text');
        const spinner = button.querySelector('.spinner');
        button.disabled = loading;
        if (loading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    function showStatus(element, message, isError = false) {
        element.textContent = message;
        element.className = `upload-status ${isError ? 'error' : 'success'}`;
        element.classList.remove('hidden');
    }

    function hideStatus(element) {
        element.classList.add('hidden');
    }

    // ===== Tab Switching =====

    function switchTab(tabName) {
        state.currentTab = tabName;

        elements.tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        elements.tabPanels.forEach(panel => {
            panel.classList.toggle('active', panel.id === `tab-${tabName}`);
        });

        if (tabName === 'candidates') {
            loadCandidates();
        }
    }

    // ===== Search =====

    async function performSearch(event) {
        event.preventDefault();

        const query = elements.searchInput.value.trim();
        if (!query) {
            showToast('Please enter a search query', 'error');
            return;
        }

        const topK = parseInt(elements.topKSelect.value, 10);
        const debug = elements.debugToggle.checked;

        setLoading(elements.searchBtn, true);
        elements.searchResults.innerHTML = '<div class="candidates-loading"><div class="spinner large"></div><p>Searching candidates...</p></div>';

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK, debug })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Search failed with status ${response.status}`);
            }

            const data = await response.json();
            state.searchResults = data.results || [];

            if (debug) {
                renderDebugPanel(data.interpreted_requirements);
            } else {
                elements.debugPanel.classList.add('hidden');
            }

            renderSearchResults(data);
        } catch (error) {
            console.error('Search error:', error);
            elements.searchResults.innerHTML = `
                <div class="candidates-empty">
                    <p>Error: ${escapeHtml(error.message)}</p>
                </div>
            `;
            showToast(error.message, 'error');
        } finally {
            setLoading(elements.searchBtn, false);
        }
    }

    function renderDebugPanel(requirements) {
        if (!requirements) {
            elements.debugPanel.classList.add('hidden');
            return;
        }

        const formatValue = (val) => {
            if (Array.isArray(val)) return val.length ? val.join(', ') : 'None';
            if (val === null || val === undefined) return 'None';
            return String(val);
        };

        const sections = [
            ['Job Titles', requirements.job_titles],
            ['Required Skills', requirements.required_skills],
            ['Experience Years', requirements.experience_years],
            ['Experience Range', requirements.experience_min || requirements.experience_max
                ? `${requirements.experience_min || '?'} - ${requirements.experience_max || '?'} years`
                : null],
            ['Location', requirements.location],
            ['Interview Required Skills', requirements.interview_required_skills],
            ['Employment Type', requirements.employment_type],
            ['Hard Constraints', requirements.hard_constraints],
            ['Soft Preferences', requirements.soft_preferences],
            ['Semantic Query', requirements.semantic_query]
        ];

        let html = '<pre>';
        sections.forEach(([label, value]) => {
            if (value !== null && value !== undefined && value !== '' &&
                (!Array.isArray(value) || value.length > 0)) {
                html += `${label}: ${formatValue(value)}\n`;
            }
        });
        html += '</pre>';

        elements.debugContent.innerHTML = html;
        elements.debugPanel.classList.remove('hidden');
    }

    function renderSearchResults(data) {
        const results = data.results || [];

        if (results.length === 0) {
            elements.searchResults.innerHTML = `
                <div class="candidates-empty">
                    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="28" cy="28" r="16" stroke="#cbd5e1" stroke-width="3"/>
                        <path d="M40 40l12 12" stroke="#cbd5e1" stroke-width="3" stroke-linecap="round"/>
                        <path d="M22 28h12" stroke="#cbd5e1" stroke-width="3" stroke-linecap="round"/>
                    </svg>
                    <p>No candidates found matching your criteria</p>
                </div>
            `;
            return;
        }

        const html = results.map((candidate, index) => renderCandidateCard(candidate, index)).join('');
        elements.searchResults.innerHTML = html;

        // Add event listeners for expand buttons
        elements.searchResults.querySelectorAll('.expand-btn').forEach(btn => {
            btn.addEventListener('click', () => toggleCardExpansion(btn));
        });

        // Add event listeners for card header clicks
        elements.searchResults.querySelectorAll('.candidate-card-header').forEach(header => {
            header.addEventListener('click', (e) => {
                if (e.target.closest('.expand-btn')) return;
                const card = header.closest('.candidate-card');
                const expandBtn = card.querySelector('.expand-btn');
                toggleCardExpansion(expandBtn);
            });
        });
    }

    function renderCandidateCard(candidate, index) {
        const scoreClass = getScoreClass(candidate.relevance_score);
        const isExpanded = state.expandedCards.has(candidate.candidate_id);

        const matchedSkills = (candidate.matched_skills || []).map(skill =>
            `<span class="skill-badge matched">${escapeHtml(skill)}</span>`
        ).join('');

        const missingReqs = (candidate.missing_requirements || []).map(req =>
            `<span class="skill-badge missing">${escapeHtml(req)}</span>`
        ).join('');

        const profileEvidenceCount = (candidate.profile_evidence || []).length;
        const interviewEvidenceCount = (candidate.interview_evidence || []).length;

        const profileEvidenceHtml = renderEvidenceList(candidate.profile_evidence || [], 'profile');
        const interviewEvidenceHtml = renderEvidenceList(candidate.interview_evidence || [], 'interview');

        const subScores = renderSubScores(candidate);

        return `
            <div class="candidate-card" data-candidate-id="${escapeHtml(candidate.candidate_id)}">
                <div class="candidate-card-header">
                    <div class="candidate-rank">${candidate.rank || index + 1}</div>
                    <div class="candidate-info">
                        <div class="candidate-name">${escapeHtml(candidate.name || 'Unknown')}</div>
                        <div class="candidate-title">${escapeHtml(candidate.title || '')}</div>
                        <div class="candidate-meta">
                            ${candidate.location ? `<span>📍 ${escapeHtml(candidate.location)}</span>` : ''}
                            ${candidate.experience ? `<span>📅 ${candidate.experience} years exp</span>` : ''}
                            ${candidate.interview_date ? `<span>🎤 ${escapeHtml(candidate.interview_date)}</span>` : ''}
                            ${candidate.interview_stage ? `<span>📋 ${escapeHtml(candidate.interview_stage)}</span>` : ''}
                        </div>
                    </div>
                    <div class="candidate-score-section">
                        <div class="relevance-score ${scoreClass}">${formatScore(candidate.relevance_score)}</div>
                        <div class="score-bar">
                            <div class="score-bar-fill ${scoreClass}" style="width: ${((candidate.relevance_score || 0) * 100).toFixed(0)}%"></div>
                        </div>
                        <div class="score-label">Relevance</div>
                    </div>
                </div>

                ${matchedSkills || missingReqs ? `
                <div class="skills-section">
                    ${matchedSkills ? `<div class="skills-label">Matched Skills</div><div class="skills-list">${matchedSkills}</div>` : ''}
                    ${missingReqs ? `<div class="skills-label" style="margin-top:8px;">Missing Requirements</div><div class="skills-list">${missingReqs}</div>` : ''}
                </div>
                ` : ''}

                ${profileEvidenceCount > 0 || interviewEvidenceCount > 0 ? `
                <div class="evidence-indicators">
                    ${profileEvidenceCount > 0 ? `
                    <div class="evidence-indicator profile">
                        <div class="evidence-dot profile"></div>
                        ${profileEvidenceCount} profile evidence
                    </div>
                    ` : ''}
                    ${interviewEvidenceCount > 0 ? `
                    <div class="evidence-indicator interview">
                        <div class="evidence-dot interview"></div>
                        ${interviewEvidenceCount} interview evidence
                    </div>
                    ` : ''}
                </div>
                ` : ''}

                <div class="candidate-details ${isExpanded ? 'expanded' : ''}">
                    ${subScores}
                    ${candidate.summary ? `
                    <div class="detail-item" style="margin-bottom:16px;">
                        <div class="detail-label">Summary</div>
                        <div class="detail-value" style="font-weight:400;line-height:1.6;">${escapeHtml(candidate.summary)}</div>
                    </div>
                    ` : ''}
                    ${profileEvidenceHtml}
                    ${interviewEvidenceHtml}
                </div>

                <button class="expand-btn ${isExpanded ? 'expanded' : ''}">
                    <svg class="chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    ${isExpanded ? 'Show Less' : 'Show More'}
                </button>
            </div>
        `;
    }

    function renderSubScores(candidate) {
        const scores = [
            { label: 'Profile Score', value: candidate.profile_score },
            { label: 'Interview Score', value: candidate.interview_score },
            { label: 'Structured Score', value: candidate.structured_score }
        ].filter(s => s.value !== null && s.value !== undefined);

        if (scores.length === 0) return '';

        return `
            <div class="details-grid">
                ${scores.map(s => `
                    <div class="detail-item">
                        <div class="detail-label">${s.label}</div>
                        <div class="detail-value">${formatScore(s.value)}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    function renderEvidenceList(evidenceList, source) {
        if (!evidenceList || evidenceList.length === 0) return '';

        const items = evidenceList.map(item => {
            const typeClass = item.evidence_type || 'mentioned';
            const meta = [
                item.source_file ? `File: ${escapeHtml(item.source_file)}` : null,
                item.timestamp ? `Date: ${escapeHtml(item.timestamp)}` : null,
                item.section ? `Section: ${escapeHtml(item.section)}` : null
            ].filter(Boolean).join(' | ');

            return `
                <div class="evidence-item ${source}">
                    <div class="evidence-text">${escapeHtml(item.text)}</div>
                    <div class="evidence-meta">
                        <span class="evidence-type-badge ${typeClass}">${escapeHtml(typeClass)}</span>
                        ${meta ? `<span>${meta}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        const title = source === 'profile' ? 'Profile Evidence' : 'Interview Evidence';
        const dotClass = source === 'profile' ? 'profile' : 'interview';

        return `
            <div class="evidence-section">
                <div class="evidence-section-title">
                    <div class="evidence-dot ${dotClass}"></div>
                    ${title}
                </div>
                <div class="evidence-list">${items}</div>
            </div>
        `;
    }

    function toggleCardExpansion(button) {
        const card = button.closest('.candidate-card');
        const details = card.querySelector('.candidate-details');
        const candidateId = card.dataset.candidateId;

        details.classList.toggle('expanded');
        button.classList.toggle('expanded');

        if (details.classList.contains('expanded')) {
            state.expandedCards.add(candidateId);
            button.innerHTML = `
                <svg class="chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Show Less
            `;
        } else {
            state.expandedCards.delete(candidateId);
            button.innerHTML = `
                <svg class="chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Show More
            `;
        }
    }

    // ===== Upload =====

    async function uploadProfile(event) {
        event.preventDefault();

        const candidateId = elements.profileCandidateId.value.trim();
        const file = elements.profileFile.files[0];

        if (!candidateId) {
            showToast('Please enter a Candidate ID', 'error');
            return;
        }
        if (!file) {
            showToast('Please select a file to upload', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('candidate_id', candidateId);

        setLoading(elements.profileUploadBtn, true);
        hideStatus(elements.profileUploadStatus);

        try {
            const response = await fetch('/api/candidates/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
            }

            const data = await response.json();
            showStatus(elements.profileUploadStatus, `Profile uploaded successfully for ${candidateId}`, false);
            showToast('Profile uploaded successfully', 'success');
            elements.profileUploadForm.reset();
            elements.profileFileName.textContent = '';
        } catch (error) {
            console.error('Profile upload error:', error);
            showStatus(elements.profileUploadStatus, `Error: ${error.message}`, true);
            showToast(error.message, 'error');
        } finally {
            setLoading(elements.profileUploadBtn, false);
        }
    }

    async function uploadInterview(event) {
        event.preventDefault();

        const candidateId = elements.interviewCandidateId.value.trim();
        const file = elements.interviewFile.files[0];

        if (!candidateId) {
            showToast('Please enter a Candidate ID', 'error');
            return;
        }
        if (!file) {
            showToast('Please select a file to upload', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('candidate_id', candidateId);

        if (elements.interviewId.value.trim()) {
            formData.append('interview_id', elements.interviewId.value.trim());
        }
        if (elements.interviewDate.value) {
            formData.append('interview_date', elements.interviewDate.value);
        }
        if (elements.interviewStage.value) {
            formData.append('interview_stage', elements.interviewStage.value);
        }
        if (elements.interviewer.value.trim()) {
            formData.append('interviewer', elements.interviewer.value.trim());
        }

        setLoading(elements.interviewUploadBtn, true);
        hideStatus(elements.interviewUploadStatus);

        try {
            const response = await fetch('/api/interviews/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
            }

            const data = await response.json();
            showStatus(elements.interviewUploadStatus, `Interview uploaded successfully for ${candidateId}`, false);
            showToast('Interview uploaded successfully', 'success');
            elements.interviewUploadForm.reset();
            elements.interviewFileName.textContent = '';
        } catch (error) {
            console.error('Interview upload error:', error);
            showStatus(elements.interviewUploadStatus, `Error: ${error.message}`, true);
            showToast(error.message, 'error');
        } finally {
            setLoading(elements.interviewUploadBtn, false);
        }
    }

    // ===== File Drop Zone =====

    function setupFileDropZone(dropZone, fileInput, fileNameDisplay) {
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileNameDisplay.textContent = e.dataTransfer.files[0].name;
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
            } else {
                fileNameDisplay.textContent = '';
            }
        });
    }

    // ===== Candidates List =====

    async function loadCandidates() {
        elements.candidatesList.innerHTML = `
            <div class="candidates-loading">
                <div class="spinner large"></div>
                <p>Loading candidates...</p>
            </div>
        `;

        try {
            const response = await fetch('/api/candidates');

            if (!response.ok) {
                throw new Error(`Failed to load candidates: ${response.status}`);
            }

            const data = await response.json();
            state.candidates = data.candidates || data || [];
            renderCandidatesList(state.candidates);
        } catch (error) {
            console.error('Load candidates error:', error);
            elements.candidatesList.innerHTML = `
                <div class="candidates-empty">
                    <p>Error loading candidates: ${escapeHtml(error.message)}</p>
                </div>
            `;
        }
    }

    function renderCandidatesList(candidates) {
        if (!candidates || candidates.length === 0) {
            elements.candidatesList.innerHTML = `
                <div class="candidates-empty">
                    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="28" cy="28" r="16" stroke="#cbd5e1" stroke-width="3"/>
                        <path d="M40 40l12 12" stroke="#cbd5e1" stroke-width="3" stroke-linecap="round"/>
                    </svg>
                    <p>No candidates found. Upload profiles to get started.</p>
                </div>
            `;
            return;
        }

        const rows = candidates.map(candidate => `
            <tr>
                <td><span class="candidate-table-name">${escapeHtml(candidate.name || 'Unknown')}</span></td>
                <td><span class="candidate-table-title">${escapeHtml(candidate.title || '')}</span></td>
                <td>${escapeHtml(candidate.candidate_id || '')}</td>
                <td>${candidate.location ? escapeHtml(candidate.location) : '-'}</td>
                <td>${candidate.experience ? candidate.experience + ' yrs' : '-'}</td>
                <td>
                    <button class="view-btn" onclick="window.viewCandidateDetail('${escapeHtml(candidate.candidate_id)}')">View</button>
                </td>
            </tr>
        `).join('');

        elements.candidatesList.innerHTML = `
            <div class="candidates-table">
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Title</th>
                            <th>ID</th>
                            <th>Location</th>
                            <th>Experience</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    // Expose viewCandidateDetail globally for onclick handler
    window.viewCandidateDetail = async function(candidateId) {
        try {
            const response = await fetch(`/api/candidates/${encodeURIComponent(candidateId)}`);

            if (!response.ok) {
                throw new Error(`Failed to load candidate: ${response.status}`);
            }

            const candidate = await response.json();
            showCandidateModal(candidate);
        } catch (error) {
            console.error('View candidate error:', error);
            showToast(error.message, 'error');
        }
    };

    function showCandidateModal(candidate) {
        elements.modalTitle.textContent = candidate.name || 'Candidate Details';

        const profileEvidence = (candidate.profile_evidence || []).map(item => `
            <div class="evidence-item profile">
                <div class="evidence-text">${escapeHtml(item.text)}</div>
                <div class="evidence-meta">
                    <span class="evidence-type-badge ${item.evidence_type || 'mentioned'}">${escapeHtml(item.evidence_type || 'mentioned')}</span>
                    ${item.source_file ? `<span>File: ${escapeHtml(item.source_file)}</span>` : ''}
                    ${item.section ? `<span>Section: ${escapeHtml(item.section)}</span>` : ''}
                </div>
            </div>
        `).join('');

        const interviewEvidence = (candidate.interview_evidence || []).map(item => `
            <div class="evidence-item interview">
                <div class="evidence-text">${escapeHtml(item.text)}</div>
                <div class="evidence-meta">
                    <span class="evidence-type-badge ${item.evidence_type || 'explained'}">${escapeHtml(item.evidence_type || 'explained')}</span>
                    ${item.source_file ? `<span>File: ${escapeHtml(item.source_file)}</span>` : ''}
                    ${item.section ? `<span>Section: ${escapeHtml(item.section)}</span>` : ''}
                </div>
            </div>
        `).join('');

        elements.modalBody.innerHTML = `
            <div class="details-grid">
                <div class="detail-item">
                    <div class="detail-label">Candidate ID</div>
                    <div class="detail-value">${escapeHtml(candidate.candidate_id || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Title</div>
                    <div class="detail-value">${escapeHtml(candidate.title || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Location</div>
                    <div class="detail-value">${escapeHtml(candidate.location || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Experience</div>
                    <div class="detail-value">${candidate.experience ? candidate.experience + ' years' : '-'}</div>
                </div>
            </div>
            ${candidate.summary ? `
            <div class="detail-item" style="margin-bottom:20px;">
                <div class="detail-label">Summary</div>
                <div class="detail-value" style="font-weight:400;line-height:1.6;">${escapeHtml(candidate.summary)}</div>
            </div>
            ` : ''}
            ${profileEvidence ? `
            <div class="evidence-section">
                <div class="evidence-section-title">
                    <div class="evidence-dot profile"></div>
                    Profile Evidence
                </div>
                <div class="evidence-list">${profileEvidence}</div>
            </div>
            ` : ''}
            ${interviewEvidence ? `
            <div class="evidence-section">
                <div class="evidence-section-title">
                    <div class="evidence-dot interview"></div>
                    Interview Evidence
                </div>
                <div class="evidence-list">${interviewEvidence}</div>
            </div>
            ` : ''}
        `;

        elements.candidateModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        elements.candidateModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    // ===== Event Listeners =====

    function initEventListeners() {
        // Tab switching
        elements.tabButtons.forEach(btn => {
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        });

        // Search
        elements.searchForm.addEventListener('submit', performSearch);
        elements.debugClose.addEventListener('click', () => {
            elements.debugPanel.classList.add('hidden');
        });

        // Upload
        elements.profileUploadForm.addEventListener('submit', uploadProfile);
        elements.interviewUploadForm.addEventListener('submit', uploadInterview);

        // File drop zones
        setupFileDropZone(elements.profileDropZone, elements.profileFile, elements.profileFileName);
        setupFileDropZone(elements.interviewDropZone, elements.interviewFile, elements.interviewFileName);

        // Candidates
        elements.refreshCandidatesBtn.addEventListener('click', loadCandidates);

        // Modal
        elements.modalClose.addEventListener('click', closeModal);
        elements.modalOverlay.addEventListener('click', closeModal);

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !elements.candidateModal.classList.contains('hidden')) {
                closeModal();
            }
        });
    }

    // ===== Initialize =====

    function init() {
        initEventListeners();
        console.log('Candidate Search RAG app initialized');
    }

    // Start the app when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
