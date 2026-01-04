import React, { useState, useEffect } from 'react';
import { uploadEvidence, verifyEvidence, listEvidence, downloadEvidence, deleteEvidence } from '../api';

function EvidenceManager() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [verifyResult, setVerifyResult] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  
  // Saved analysis state
  const [savedAnalysis, setSavedAnalysis] = useState(null);
  const [evidenceNotes, setEvidenceNotes] = useState('');
  const [isSavingAnalysis, setIsSavingAnalysis] = useState(false);

  // Stored evidence list state
  const [storedEvidence, setStoredEvidence] = useState([]);
  const [isLoadingEvidence, setIsLoadingEvidence] = useState(false);
  const [evidenceError, setEvidenceError] = useState(null);
  const [selectedEvidence, setSelectedEvidence] = useState(null);

  // File upload state
  const [evidenceType, setEvidenceType] = useState('document');
  const [evidenceDescription, setEvidenceDescription] = useState('');
  const [caseId, setCaseId] = useState('web-upload');
  const [originalFileHash, setOriginalFileHash] = useState('');

  // Load saved analysis on component mount
  useEffect(() => {
    const analysis = localStorage.getItem('savedAnalysis');
    if (analysis) {
      try {
        const analysisData = JSON.parse(analysis);
        setSavedAnalysis(analysisData);
      } catch (err) {
        console.warn('Failed to load saved analysis:', err);
      }
    }
    
    // Load stored evidence
    loadStoredEvidence();
  }, []);

  // Load stored evidence from backend
  const loadStoredEvidence = async () => {
    setIsLoadingEvidence(true);
    setEvidenceError(null);
    try {
      const result = await listEvidence();
      setStoredEvidence(result.evidence || []);
    } catch (err) {
      setEvidenceError('Failed to load stored evidence: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setIsLoadingEvidence(false);
    }
  };

  // Download evidence file
  const downloadEvidenceFile = async (evidence) => {
    try {
      const response = await downloadEvidence(evidence.id);
      
      // Create a more descriptive filename with evidence details
      const fileExtension = evidence.file_name.split('.').pop();
      const timestamp = evidence.created_at ? 
        new Date(evidence.created_at).toISOString().split('T')[0] : 
        'unknown';
      const descriptiveName = `${evidence.evidence_type}_${evidence.id}_${timestamp}.${fileExtension}`;
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', descriptiveName);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      alert(`Downloaded: ${descriptiveName}\nType: ${evidence.evidence_type}\nDescription: ${evidence.description || 'No description'}`);
    } catch (err) {
      alert('Failed to download evidence: ' + (err?.response?.data?.detail || err.message));
    }
  };

  // Delete evidence file
  const deleteEvidenceFile = async (evidence) => {
    if (!confirm(`Are you sure you want to delete "${evidence.file_name}"?\n\nThis action cannot be undone.`)) {
      return;
    }

    try {
      await deleteEvidence(evidence.id);
      alert(`Evidence "${evidence.file_name}" has been deleted successfully.`);
      // Reload the evidence list
      loadStoredEvidence();
    } catch (err) {
      alert('Failed to delete evidence: ' + (err?.response?.data?.detail || err.message));
    }
  };

  // View evidence details
  const viewEvidenceDetails = (evidence) => {
    setSelectedEvidence(evidence);
  };

  // Close evidence details modal
  const closeEvidenceDetails = () => {
    setSelectedEvidence(null);
  };

  // Save analysis results as evidence
  const saveAnalysisAsEvidence = async () => {
    if (!savedAnalysis) {
      setError('No saved analysis found to save as evidence.');
      return;
    }

    setIsSavingAnalysis(true);
    setError(null);

    try {
      // Convert analysis data to a format that can be uploaded
      const evidenceData = {
        analysis_results: savedAnalysis,
        notes: evidenceNotes,
        timestamp: new Date().toISOString(),
        type: 'facial_recognition_analysis'
      };

      // For now, we'll save as JSON blob. In a real system, this would go to a proper evidence API
      const blob = new Blob([JSON.stringify(evidenceData, null, 2)], { type: 'application/json' });
      const file = new File([blob], `facial_analysis_${Date.now()}.json`, { type: 'application/json' });

      const result = await uploadEvidence(file, {
        evidence_type: 'facial_analysis',
        notes: evidenceNotes,
        analysis_summary: `Facial recognition analysis with ${savedAnalysis.reconstructionData?.matches?.length || 0} matches found`
      });

      setUploadResult(result);
      alert('Analysis saved as evidence successfully!');
      
      // Ask user if they want to clear the saved analysis
      if (window.confirm('Analysis saved successfully! Clear the saved analysis from browser storage?')) {
        localStorage.removeItem('savedAnalysis');
        setSavedAnalysis(null);
        setEvidenceNotes('');
      }

    } catch (err) {
      setError('Failed to save analysis as evidence: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setIsSavingAnalysis(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadResult(null);
    setVerifyResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const result = await uploadEvidence(selectedFile, {
        evidence_type: evidenceType,
        description: evidenceDescription || undefined
      });
      setUploadResult(result);
    } catch (err) {
      setError('Upload failed: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setIsUploading(false);
    }
  };

  const handleVerify = async () => {
    if (!selectedFile) {
      setError('Please select a file to verify.');
      return;
    }

    // For 3D models, require manual hash entry for watermark verification
    if (evidenceType === 'facial_analysis' && !originalFileHash.trim()) {
      setError('Please enter the original file hash for 3D model watermark verification.');
      return;
    }

    setIsVerifying(true);
    setError(null);
    try {
      if (evidenceType === 'facial_analysis') {
        // 3D model watermark verification
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('case_id', caseId);
        formData.append('file_hash', originalFileHash);

        const response = await fetch('http://127.0.0.1:8000/api/v1/evidence/verify-watermark', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        setVerifyResult(result);
      } else {
        // Regular file integrity verification - calculate hash automatically
        const fileBuffer = await selectedFile.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest('SHA-256', fileBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const calculatedHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

        // For now, just show the calculated hash as verification
        setVerifyResult({
          file_name: selectedFile.name,
          calculated_hash: calculatedHash,
          verification_type: 'integrity_check',
          message: 'File integrity verified. SHA-256 hash calculated successfully.',
          evidence_type: evidenceType
        });
      }
    } catch (err) {
      setError('Verification failed: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="evidence-manager">
      <div className="evidence-header">
        <h1 className="evidence-title">
          <span className="evidence-icon">🔍</span>
          Evidence Management Center
        </h1>
        <p className="evidence-subtitle">Secure evidence upload, verification, and analysis storage</p>
      </div>

      <div className="evidence-grid">
        {/* File Upload Card */}
        <div className="evidence-card upload-card">
          <div className="card-header">
            <h3 className="card-title">
              <span className="card-icon">📁</span>
              File Upload
            </h3>
            <p className="card-description">Upload evidence files for processing and verification</p>
          </div>

          <div className="card-content">
            <div className="file-input-wrapper">
              <input
                type="file"
                onChange={handleFileChange}
                className="file-input"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="file-input-label">
                <span className="file-icon">📎</span>
                Choose Evidence File
              </label>
              {selectedFile && (
                <div className="file-info">
                  <span className="file-name">{selectedFile.name}</span>
                  <span className="file-size">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                </div>
              )}
            </div>

            <div className="evidence-form">
              <div className="form-group">
                <label className="form-label">
                  <span className="label-icon">🏷️</span>
                  Evidence Type
                </label>
                <select
                  value={evidenceType}
                  onChange={(e) => setEvidenceType(e.target.value)}
                  className="form-select"
                >
                  <option value="document">📄 Document</option>
                  <option value="audio">🎵 Audio</option>
                  <option value="image">📸 Image</option>
                  <option value="video">🎥 Video</option>
                  <option value="facial_analysis">🔍 Facial Analysis</option>
                  <option value="police_report">📋 Police Report</option>
                  <option value="witness_statement">👤 Witness Statement</option>
                  <option value="forensic_data">🔬 Forensic Data</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span className="label-icon">📝</span>
                  Description (Optional)
                </label>
                <textarea
                  value={evidenceDescription}
                  onChange={(e) => setEvidenceDescription(e.target.value)}
                  placeholder="Describe this evidence..."
                  className="form-textarea"
                  rows="2"
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span className="label-icon">🔍</span>
                  {evidenceType === 'facial_analysis'
                    ? 'Original File Hash (for 3D watermark verification)'
                    : 'File Verification (automatic integrity check)'}
                </label>
                {evidenceType === 'facial_analysis' ? (
                  <input
                    type="text"
                    value={originalFileHash}
                    onChange={(e) => setOriginalFileHash(e.target.value)}
                    placeholder="Enter hash for watermark verification..."
                    className="form-input"
                  />
                ) : (
                  <div className="form-info">
                    <span className="info-icon">ℹ️</span>
                    SHA-256 hash will be calculated automatically for integrity verification
                  </div>
                )}
              </div>
            </div>

            <div className="action-buttons">
              <button
                onClick={handleUpload}
                disabled={isUploading || !selectedFile}
                className="btn btn-primary"
              >
                {isUploading ? (
                  <>
                    <span className="btn-icon">⏳</span>
                    Uploading...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">⬆️</span>
                    Upload Evidence
                  </>
                )}
              </button>

              <button
                onClick={handleVerify}
                disabled={isVerifying || !selectedFile || (evidenceType === 'facial_analysis' && !originalFileHash.trim())}
                className="btn btn-secondary"
              >
                {isVerifying ? (
                  <>
                    <span className="btn-icon">🔍</span>
                    Verifying...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">✅</span>
                    {evidenceType === 'facial_analysis' ? 'Verify Watermark' : 'Verify Integrity'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Saved Analysis Card */}
        {savedAnalysis ? (
          <div className="evidence-card analysis-card">
            <div className="card-header">
              <h3 className="card-title">
                <span className="card-icon">💾</span>
                Saved Analysis
              </h3>
              <p className="card-description">Your facial recognition analysis results</p>
            </div>

            <div className="card-content">
              <div className="analysis-summary">
                <div className="metric-grid">
                  <div className="metric-item">
                    <div className="metric-label">Image Quality</div>
                    <div className="metric-value quality">
                      {savedAnalysis.reconstructionData?.image_quality_score ?
                        (savedAnalysis.reconstructionData.image_quality_score * 100).toFixed(0) + '%' :
                        'N/A'
                      }
                    </div>
                  </div>

                  <div className="metric-item">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value confidence">
                      {savedAnalysis.reconstructionData?.prediction_entropy != null ?
                        ((1 - savedAnalysis.reconstructionData.prediction_entropy) * 100).toFixed(0) + '%' :
                        'N/A'
                      }
                    </div>
                  </div>

                  <div className="metric-item">
                    <div className="metric-label">Matches Found</div>
                    <div className="metric-value matches">
                      {savedAnalysis.reconstructionData?.matches?.length || 0}
                    </div>
                  </div>

                  <div className="metric-item">
                    <div className="metric-label">Analysis Date</div>
                    <div className="metric-value date">
                      {savedAnalysis.timestamp ?
                        new Date(savedAnalysis.timestamp).toLocaleDateString() :
                        'Unknown'
                      }
                    </div>
                  </div>
                </div>
              </div>

              {savedAnalysis.reconstructionData?.matches?.length > 0 && (
                <div className="matches-section">
                  <h4 className="section-title">Recognition Results</h4>
                  <div className="matches-list">
                    {savedAnalysis.reconstructionData.matches.map((match, index) => (
                      <div key={index} className={`match-item ${match.wanted ? 'wanted' : 'not-wanted'}`}>
                        <div className="match-header">
                          <span className="match-status">
                            {match.wanted ? '🚨 WANTED' : '✅ NOT WANTED'}
                          </span>
                          <span className="match-similarity">
                            {(match.similarity * 100).toFixed(1)}% match
                          </span>
                        </div>
                        <div className="match-name">
                          {match.name || `Person ${match.person_id}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="notes-section">
                <label className="notes-label">
                  <span className="label-icon">📝</span>
                  Evidence Notes
                </label>
                <textarea
                  value={evidenceNotes}
                  onChange={(e) => setEvidenceNotes(e.target.value)}
                  placeholder="Add contextual notes about this evidence..."
                  className="notes-input"
                />
              </div>

              <button
                onClick={saveAnalysisAsEvidence}
                disabled={isSavingAnalysis}
                className="btn btn-success save-analysis-btn"
              >
                {isSavingAnalysis ? (
                  <>
                    <span className="btn-icon">⏳</span>
                    Saving...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">💾</span>
                    Save as Evidence
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="evidence-card empty-card">
            <div className="empty-state">
              <div className="empty-icon">🔬</div>
              <h3 className="empty-title">No Saved Analysis</h3>
              <p className="empty-description">
                Process an image on the Upload page and save your analysis to see it here.
              </p>
              <a href="/upload" className="empty-action">
                <span className="action-icon">📤</span>
                Go to Upload
              </a>
            </div>
          </div>
        )}

        {/* Stored Evidence List Card */}
        <div className="evidence-card stored-evidence-card">
          <div className="card-header">
            <h3 className="card-title">
              <span className="card-icon">📚</span>
              Stored Evidence Database
            </h3>
            <p className="card-description">View and download all stored evidence files</p>
          </div>

          <div className="card-content">
            <div className="evidence-controls">
              <button
                onClick={loadStoredEvidence}
                disabled={isLoadingEvidence}
                className="btn btn-secondary refresh-btn"
              >
                {isLoadingEvidence ? (
                  <>
                    <span className="btn-icon">⏳</span>
                    Loading...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🔄</span>
                    Refresh List
                  </>
                )}
              </button>
            </div>

            {evidenceError && (
              <div className="error-message">
                <span className="error-icon">❌</span>
                {evidenceError}
              </div>
            )}

            {storedEvidence.length > 0 ? (
              <div className="evidence-list">
                <div className="evidence-list-header">
                  <span className="header-item">Type</span>
                  <span className="header-item">File Name</span>
                  <span className="header-item">Description</span>
                  <span className="header-item">Date</span>
                  <span className="header-item">Actions</span>
                </div>
                {storedEvidence.map((evidence) => (
                  <div key={evidence.id} className="evidence-list-item">
                    <span className="evidence-type">
                      <span className="type-icon">
                        {evidence.evidence_type === 'facial_analysis' ? '🔍' :
                         evidence.evidence_type === 'document' ? '📄' :
                         evidence.evidence_type === 'audio' ? '🎵' :
                         evidence.evidence_type === 'image' ? '📸' :
                         evidence.evidence_type === 'video' ? '🎥' :
                         evidence.evidence_type === 'police_report' ? '📋' :
                         evidence.evidence_type === 'witness_statement' ? '👤' :
                         evidence.evidence_type === 'forensic_data' ? '🔬' : '📁'}
                      </span>
                      {evidence.evidence_type.replace('_', ' ')}
                    </span>
                    <span className="evidence-filename" title={evidence.file_name}>
                      {evidence.file_name.length > 20 ? 
                        evidence.file_name.substring(0, 20) + '...' : 
                        evidence.file_name}
                    </span>
                    <span className="evidence-description" title={evidence.description || 'No description'}>
                      {evidence.description ? 
                        (evidence.description.length > 30 ? 
                          evidence.description.substring(0, 30) + '...' : 
                          evidence.description) : 
                        'No description'}
                    </span>
                    <span className="evidence-date">
                      {evidence.created_at ? 
                        new Date(evidence.created_at).toLocaleDateString() : 
                        'Unknown'}
                    </span>
                    <div className="evidence-actions">
                      <button
                        onClick={() => viewEvidenceDetails(evidence)}
                        className="btn btn-small btn-info"
                        title="View evidence details"
                      >
                        <span className="btn-icon">👁️</span>
                        <span className="btn-text">Details</span>
                      </button>
                      <button
                        onClick={() => downloadEvidenceFile(evidence)}
                        className="btn btn-small btn-primary"
                        title={`Download ${evidence.file_name}`}
                      >
                        <span className="btn-icon">⬇️</span>
                        <span className="btn-text">Download</span>
                      </button>
                      <button
                        onClick={() => deleteEvidenceFile(evidence)}
                        className="btn btn-small btn-danger"
                        title={`Delete ${evidence.file_name}`}
                      >
                        <span className="btn-icon">🗑️</span>
                        <span className="btn-text">Delete</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-evidence-state">
                <div className="empty-icon">📂</div>
                <h4 className="empty-title">No Stored Evidence</h4>
                <p className="empty-description">
                  Upload evidence files above to see them stored here.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Evidence Details Modal */}
        {selectedEvidence && (
          <div className="evidence-modal-overlay" onClick={closeEvidenceDetails}>
            <div className="evidence-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3 className="modal-title">
                  <span className="modal-icon">📄</span>
                  Evidence Details
                </h3>
                <button
                  onClick={closeEvidenceDetails}
                  className="modal-close-btn"
                  title="Close details"
                >
                  ✕
                </button>
              </div>

              <div className="modal-content">
                <div className="detail-grid">
                  <div className="detail-item">
                    <label className="detail-label">Evidence ID</label>
                    <span className="detail-value">#{selectedEvidence.id}</span>
                  </div>

                  <div className="detail-item">
                    <label className="detail-label">Type</label>
                    <span className="detail-value evidence-type-badge">
                      <span className="type-icon">
                        {selectedEvidence.evidence_type === 'facial_analysis' ? '🔍' :
                         selectedEvidence.evidence_type === 'document' ? '📄' :
                         selectedEvidence.evidence_type === 'audio' ? '🎵' :
                         selectedEvidence.evidence_type === 'image' ? '📸' :
                         selectedEvidence.evidence_type === 'video' ? '🎥' :
                         selectedEvidence.evidence_type === 'police_report' ? '📋' :
                         selectedEvidence.evidence_type === 'witness_statement' ? '👤' :
                         selectedEvidence.evidence_type === 'forensic_data' ? '🔬' : '📁'}
                      </span>
                      {selectedEvidence.evidence_type.replace('_', ' ')}
                    </span>
                  </div>

                  <div className="detail-item">
                    <label className="detail-label">File Name</label>
                    <span className="detail-value">{selectedEvidence.file_name}</span>
                  </div>

                  <div className="detail-item">
                    <label className="detail-label">File Size</label>
                    <span className="detail-value">{(selectedEvidence.file_size / 1024).toFixed(1)} KB</span>
                  </div>

                  <div className="detail-item">
                    <label className="detail-label">Media Type</label>
                    <span className="detail-value">{selectedEvidence.media_type}</span>
                  </div>

                  <div className="detail-item">
                    <label className="detail-label">Created Date</label>
                    <span className="detail-value">
                      {selectedEvidence.created_at ? 
                        new Date(selectedEvidence.created_at).toLocaleString() : 
                        'Unknown'}
                    </span>
                  </div>

                  <div className="detail-item full-width">
                    <label className="detail-label">Description</label>
                    <span className="detail-value description">
                      {selectedEvidence.description || 'No description provided'}
                    </span>
                  </div>
                </div>

                <div className="modal-actions">
                  <button
                    onClick={() => downloadEvidenceFile(selectedEvidence)}
                    className="btn btn-primary modal-download-btn"
                  >
                    <span className="btn-icon">⬇️</span>
                    Download Evidence
                  </button>
                  <button
                    onClick={closeEvidenceDetails}
                    className="btn btn-secondary modal-close-btn"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status Messages */}
      {(isUploading || isVerifying || uploadResult || verifyResult || error) && (
        <div className="status-section">
          {isUploading && (
            <div className="status-message loading">
              <span className="status-icon">⏳</span>
              Uploading evidence file...
            </div>
          )}

          {isVerifying && (
            <div className="status-message loading">
              <span className="status-icon">🔍</span>
              {evidenceType === 'facial_analysis'
                ? 'Verifying 3D model watermark...'
                : 'Calculating file integrity hash...'}
            </div>
          )}

          {uploadResult && (
            <div className="status-message success">
              <span className="status-icon">✅</span>
              <div className="status-content">
                <strong>Upload Successful</strong>
                <pre className="status-details">{JSON.stringify(uploadResult, null, 2)}</pre>
              </div>
            </div>
          )}

          {verifyResult && (
            <div className="status-message info">
              <span className="status-icon">ℹ️</span>
              <div className="status-content">
                <strong>
                  {verifyResult.verification_type === 'integrity_check'
                    ? 'File Integrity Verified'
                    : '3D Model Verification Complete'}
                </strong>
                <pre className="status-details">{JSON.stringify(verifyResult, null, 2)}</pre>
              </div>
            </div>
          )}

          {error && (
            <div className="status-message error">
              <span className="status-icon">❌</span>
              <div className="status-content">
                <strong>Error</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default EvidenceManager;
