import React, { useState, useRef, useContext, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { recognizeFace, reconstructFace, generateReport, logout, uploadEvidence } from '../api';
import FaceViewer from '../components/FaceViewer';
import { AuthContext } from '../context/AuthProvider';
import './Upload.css';

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [reconstructionData, setReconstructionData] = useState(null);
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [report, setReport] = useState(null);
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState('');
  const [showSaliency, setShowSaliency] = useState(false);
  const [analysisLoaded, setAnalysisLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState('Loading...');
  const [yoloDetections, setYoloDetections] = useState(null);

  // Format file size with appropriate unit (KB or MB)
  const formatFileSize = (bytes) => {
    if (bytes < 1024 * 1024) {
      // Show in KB for files smaller than 1MB
      return `${(bytes / 1024).toFixed(2)} KB`;
    } else {
      // Show in MB for larger files
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }
  };

  // Load saved analysis on component mount
  useEffect(() => {
    const loadSavedAnalysis = () => {
      try {
        const savedData = localStorage.getItem('savedAnalysis');
        if (savedData) {
          const analysisData = JSON.parse(savedData);
          console.log('Loading saved analysis:', analysisData);
          
          // Restore the saved state
          setReconstructionData(analysisData.reconstructionData);
          setRecognitionResult(analysisData.recognitionResult);
          setReport(analysisData.report);
          setImagePreview(analysisData.imagePreview);
          setShowPreview(analysisData.showPreview || false);
          setYoloDetections(analysisData.yoloDetections);
          
          // Note: selectedFile cannot be fully restored from localStorage
          // as File objects can't be serialized, but we can show the metadata
          if (analysisData.selectedFile) {
            setSelectedFile({
              name: analysisData.selectedFile.name,
              type: analysisData.selectedFile.type,
              size: analysisData.selectedFile.size,
              // This is a placeholder - the actual file data is lost
              _restored: true
            });
          }
          
          setAnalysisLoaded(true);
          console.log('✅ Saved analysis loaded successfully');
        }
      } catch (err) {
        console.error('Failed to load saved analysis:', err);
        // Clear corrupted data
        localStorage.removeItem('savedAnalysis');
      }
    };
    
    loadSavedAnalysis();
  }, []);

  // Save current analysis to localStorage
  const saveAnalysis = () => {
    const analysisData = {
      reconstructionData,
      recognitionResult,
      report,
      imagePreview,
      showPreview,
      yoloDetections,
      selectedFile: selectedFile ? {
        name: selectedFile.name,
        type: selectedFile.type,
        size: selectedFile.size
      } : null,
      timestamp: new Date().toISOString()
    };
    
    console.log('Saving analysis data:', analysisData);
    console.log('Recognition result matches:', recognitionResult?.matches);
    console.log('Reconstruction matches:', reconstructionData?.matches);
    
    try {
      localStorage.setItem('savedAnalysis', JSON.stringify(analysisData));
      // Set flag for dashboard intelligence unlocking
      localStorage.setItem('hasProcessedEvidence', 'true');
      alert('Analysis saved successfully! Dashboard intelligence is now unlocked.');
    } catch (err) {
      console.error('Failed to save analysis:', err);
      alert('Failed to save analysis. Storage might be full.');
    }
  };

  // Save evidence after processing is complete
  const saveEvidenceAfterProcessing = async (recognitionData) => {
    try {
      console.log('Saving processed evidence to database...');
      await uploadEvidence(selectedFile, {
        evidence_type: 'image',
        description: `Processed facial recognition evidence - ${recognitionData.verdict} (confidence: ${(recognitionData.cosine_score * 100).toFixed(1)}%)`
      });
      console.log('✅ Evidence saved successfully');
    } catch (evidenceError) {
      console.warn('⚠️ Could not save evidence to database:', evidenceError);
      // Don't fail the whole process if evidence saving fails
    }
  };

  // Clear saved analysis
  const clearSavedAnalysis = () => {
    localStorage.removeItem('savedAnalysis');
    localStorage.removeItem('hasProcessedEvidence');
    setAnalysisLoaded(false);
    alert('Saved analysis cleared.');
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        handleFileChange({ target: { files: [file] } });
      } else {
        setError('Please upload a valid image file.');
      }
    }
  };

  const validateFile = (file) => {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    
    if (!allowedTypes.includes(file.type)) {
      return 'Please upload a JPEG, PNG, or WebP image.';
    }
    
    if (file.size > maxSize) {
      return 'File size must be less than 10MB.';
    }
    
    return null;
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      
      setSelectedFile(file);
      setImageDimensions('Loading...');
      
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result;
        setImagePreview(dataUrl);
        setShowPreview(true);
        
        // Calculate image dimensions
        const img = new Image();
        img.onload = () => {
          setImageDimensions(`${img.naturalWidth}×${img.naturalHeight}`);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
      setReconstructionData(null);
      setRecognitionResult(null);
      setError(null);
      setReport(null);
      setReportError(null);
      setAnalysisLoaded(false); // Reset loaded state when selecting new file
      // Clear any saved analysis since we're starting fresh
      localStorage.removeItem('savedAnalysis');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image to upload.');
      return;
    }

    setIsLoading(true);
    setUploadProgress(0);
    setProcessingStep('Initializing...');
    setError(null);
    setReconstructionData(null);
    setRecognitionResult(null);
    setReport(null);
    setReportError(null);
    setYoloDetections(null);

    try {
      // Step 1: Start 3D reconstruction
      setProcessingStep('Starting 3D reconstruction...');
      setUploadProgress(10);
      
      const reconResponse = await reconstructFace(selectedFile);
      setUploadProgress(40);
      
      // Set reconstruction data directly
      setReconstructionData(reconResponse.data);
      setUploadProgress(80);

      setProcessingStep('Analyzing facial features...');
      setUploadProgress(80);
      
      // Step 2: Recognize the face (can run in parallel)
      const recognitionResponse = await recognizeFace(selectedFile);
      setRecognitionResult(recognitionResponse.data);
      
      // Extract YOLO results from recognition response
      if (recognitionResponse.data.yolo_results) {
        setYoloDetections(recognitionResponse.data.yolo_results);
      }
      
      setUploadProgress(100);
      setProcessingStep('Analysis complete!');

      // For synchronous reconstruction, save evidence after recognition
      if (reconResponse.data.vertices && reconResponse.data.faces) {
        await saveEvidenceAfterProcessing(recognitionResponse.data);
      }
      // For async reconstruction, evidence is saved in the polling completion handler

    } catch (err) {
      console.error('Upload error:', err);
      
      // Handle validation errors (422) properly
      if (err.response?.status === 422 && err.response?.data?.detail) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          // Validation errors are an array of objects
          const errorMessages = detail.map(e => `${e.loc?.join('.') || 'Field'}: ${e.msg}`).join('; ');
          setError(`Validation error: ${errorMessages}`);
        } else if (typeof detail === 'string') {
          setError(detail);
        } else {
          setError('Validation error occurred');
        }
      } else {
        setError(err.response?.data?.detail || err.message || 'An error occurred during upload.');
      }
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        setProcessingStep('');
        setUploadProgress(0);
      }, 2000);
    }
  };

  const handleGenerateReport = async () => {
    if (!recognitionResult || !reconstructionData) {
      setReportError('Cannot generate a report without recognition and reconstruction results.');
      return;
    }
    setIsReportLoading(true);
    setReportError('');
    setReport(null);
    try {
      // Pass proper payload matching ReportRequest schema
      const payload = {
        case_id: 'web-upload',  // Use the same case_id as recognition
        case_context: `Recognition result: ${recognitionResult.verdict} with score ${recognitionResult.cosine_score}. Reconstruction completed with ${reconstructionData.vertices ? reconstructionData.vertices.length : 0} vertices.`,
        matches: reconstructionData.matches || [],
        vertices: reconstructionData.vertices || [],
        faces: reconstructionData.faces || [],
        include_risk_assessment: true,
        include_predicted_hotspots: true,
        include_ethical_concerns: true
      };
      const response = await generateReport(payload);
      const file = new Blob([response.data], { type: 'application/pdf' });
      const fileURL = URL.createObjectURL(file);
      // Open in new tab
      window.open(fileURL);
      setReport("Report generated and opened in a new tab.");
    } catch (err) {
      console.error('Report generation error:', err);
      
      // Handle validation errors properly
      if (err.response?.status === 422 && err.response?.data?.detail) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          const errorMessages = detail.map(e => `${e.loc?.join('.') || 'Field'}: ${e.msg}`).join('; ');
          setReportError(`Validation error: ${errorMessages}`);
        } else {
          setReportError(typeof detail === 'string' ? detail : 'Validation error occurred');
        }
      } else {
        const errorMsg = err.response?.data?.detail || err.message || 'An unknown error occurred during report generation.';
        setReportError(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
      }
    } finally {
      setIsReportLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    setAuth({});
    navigate('/login');
  };

  return (
    <div className="upload-container">
      <div className="upload-form">
        <h2>AI-Powered Facial Recognition and Reconstruction</h2>
        <div className="page-navigation">
          <Link to="/">Dashboard</Link> | 
          <Link to="/evidence">Evidence</Link> | 
          <Link to="/analytics">Analytics</Link>
        </div>
        <div className="form-actions">
          <div 
            className={`file-drop-area ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <div className="drop-content">
              {selectedFile ? (
                <>
                  <div className="file-info">
                    <strong>{selectedFile.name}</strong>
                    <br />
                    <small>{formatFileSize(selectedFile.size)}</small>
                  </div>
                  <button 
                    className="change-file-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current.click();
                    }}
                  >
                    Change File
                  </button>
                </>
              ) : (
                <>
                  <div className="drop-text">
                    <strong>Drop your CCTV image here</strong>
                    <br />
                    or click to browse files
                  </div>
                  <div className="file-types">Supports: JPEG, PNG, WebP (max 10MB)</div>
                  <div className="upload-options">
                    <button 
                      className="camera-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        // Camera functionality would go here
                        alert('Camera capture coming soon!');
                      }}
                    >
                      Use Camera
                    </button>
                  </div>
                </>
              )}
            </div>
            <input 
              type="file" 
              onChange={handleFileChange} 
              accept="image/jpeg,image/jpg,image/png,image/webp" 
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
          </div>
          <button onClick={handleUpload} disabled={isLoading || !selectedFile} className="process-btn">
            {isLoading ? 'Processing...' : 'Process Image'}
          </button>
          <button onClick={handleLogout} className="logout-button">Logout</button>
        </div>
        {isLoading && (
          <div className="progress-container">
            <div className="progress-info">
              <div className="progress-text">{processingStep}</div>
              <div className="progress-percent">{uploadProgress}%</div>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}
        {error && <p className="error-message">{error}</p>}
        {analysisLoaded && (
          <div className="success-message" style={{
            background: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1rem',
            textAlign: 'center',
            color: '#ffffff'
          }}>
            <strong>Analysis Restored:</strong> Your previous work has been loaded. You can continue from where you left off.
          </div>
        )}
        
        {showPreview && imagePreview && (
          <div className="preview-modal">
            <div className="preview-content">
              <div className="preview-header">
                <h3>Image Preview</h3>
                <button 
                  className="close-preview"
                  onClick={() => setShowPreview(false)}
                >
                  ✕
                </button>
              </div>
              <div className="preview-image-container">
                <img src={imagePreview} alt="Preview" className="preview-image" />
                <div className="preview-info">
                  <div className="info-item">
                    <strong>File:</strong> {selectedFile.name}
                  </div>
                  <div className="info-item">
                    <strong>Size:</strong> {formatFileSize(selectedFile.size)}
                  </div>
                  <div className="info-item">
                    <strong>Quality Check:</strong> 
                    <span className={`quality-indicator ${selectedFile.size > 5000000 ? 'quality-warning' : 'quality-good'}`}>
                      {selectedFile.size > 5000000 ? 'Large file (>5MB)' : 'Good size'}
                    </span>
                  </div>
                  <div className="info-item">
                    <strong>Dimensions:</strong> <span id="image-dimensions">{imageDimensions}</span>
                  </div>
                </div>
              </div>
              <div className="preview-actions">
                <button 
                  className="preview-process-btn"
                  onClick={() => {
                    setShowPreview(false);
                    handleUpload();
                  }}
                  disabled={isLoading}
                >
                  {isLoading ? 'Processing...' : 'Process This Image'}
                </button>
                <button 
                  className="preview-cancel-btn"
                  onClick={() => {
                    setShowPreview(false);
                    setSelectedFile(null);
                    setImagePreview(null);
                    setImageDimensions('Loading...');
                  }}
                >
                  Choose Different Image
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

       <div className="results-container">
         <div className="viewer-section">
           <h3>Input Image</h3>
           {imagePreview && (
             <div className="image-preview-container" style={{ position: 'relative', marginBottom: '1rem', background: '#eee', minHeight: '100px' }}>
               <img src={imagePreview} alt="Input" style={{ maxWidth: '100%', display: 'block', borderRadius: '4px' }} />
               {showSaliency && recognitionResult?.matches?.[0]?.saliency_url && (
                 <img
                   src={`https://api.localhost${recognitionResult.matches?.[0]?.saliency_url}`}
                   alt="Saliency Overlay"
                   style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0.6, borderRadius: '4px' }}
                 />
               )}
             </div>
           )}
           <h3>3D Reconstruction</h3>
           {isLoading && !reconstructionData && <div className="loader"></div>}
           <FaceViewer reconstructionData={reconstructionData} />
         </div>

         <div className="viewer-section">
           <h3>Recognition Results</h3>
           {isLoading && !reconstructionData && <div className="loader"></div>}
           {reconstructionData && (
             <>
               {/* Display forensic scores from reconstruction result */}
               {reconstructionData?.image_quality_score != null && reconstructionData.image_quality_score < 0.4 && (
                 <div className="error-message low-confidence-banner">
                   <strong>Warning:</strong> Input image quality is low ({reconstructionData.image_quality_score.toFixed(2)}/1.0). Results may be unreliable.
                 </div>
               )}
               <div className="quality-metrics">
                 <p><strong>Image Quality:</strong> <span className={(reconstructionData?.image_quality_score || 0) > 0.6 ? 'quality-good' : 'quality-warning'}>{reconstructionData?.image_quality_score != null ? (reconstructionData.image_quality_score * 100).toFixed(0) + '%' : 'N/A'}</span></p>
                 <p><strong>Confidence:</strong> <span className={(reconstructionData?.prediction_entropy || 1) < 0.5 ? 'quality-good' : 'quality-warning'}>{reconstructionData?.prediction_entropy != null ? ((1 - reconstructionData.prediction_entropy) * 100).toFixed(0) + '%' : 'N/A'}</span></p>
               </div>
               {reconstructionData?.matches?.length > 0 ? (
                 <div className="recognition-list">
                   <ul>
                    {reconstructionData.matches.map((match, index) => (
                      <li key={index} className={match.wanted ? 'wanted' : 'not-wanted'}>
                        <div className="match-text">
                          <strong>{match.wanted ? 'WANTED' : 'NOT WANTED'}</strong> - {match.name || `Person ${match.person_id}`}
                        </div>
                        <div className="match-details">
                          Similarity: {(match.similarity * 100).toFixed(1)}% | ID: {match.person_id}
                        </div>
                      </li>
                    ))}
                   </ul>
                   <div className="report-action">
                     <button onClick={handleGenerateReport} disabled={isReportLoading || !reconstructionData}>
                       {isReportLoading ? 'Generating...' : 'Generate AI Report'}
                     </button>
                     <button onClick={saveAnalysis} disabled={!reconstructionData}>
                       Save Analysis
                     </button>
                     <button onClick={clearSavedAnalysis}>
                       Clear Saved
                     </button>
                   </div>
                 </div>
               ) : (
                 <p>No potential matches found in the database.</p>
               )}
             </>
           )}
         </div>

         <div className="viewer-section yolo-section">
           <h3>OBJECT DETECTION (YOLO)</h3>
           {yoloDetections && (
             <div className="yolo-results">
               <div className="detection-summary">
                 <div className="detection-stats">
                   <div className="stat-item persons">
                     <span className="stat-label">PERSONS DETECTED:</span>
                     <span className="stat-value">{yoloDetections.persons?.length || 0}</span>
                   </div>
                   <div className="stat-item weapons">
                     <span className="stat-label">WEAPONS DETECTED:</span>
                     <span className="stat-value">{yoloDetections.weapons?.length || 0}</span>
                   </div>
                   <div className="stat-item location">
                     <span className="stat-label">LOCATION:</span>
                     <span className="stat-value">{yoloDetections.location_id || 'N/A'}</span>
                   </div>
                   <div className="stat-item camera">
                     <span className="stat-label">CAMERA:</span>
                     <span className="stat-value">{yoloDetections.camera_id || 'N/A'}</span>
                   </div>
                 </div>
               </div>
               
               {yoloDetections.image_quality && (
                 <div className="image-quality-info">
                   <p><strong>Image Size:</strong> {yoloDetections.image_quality.original_size?.join('×') || 'N/A'}</p>
                   {yoloDetections.image_quality.was_resized && (
                     <p><em>Image was resized for processing</em></p>
                   )}
                   {yoloDetections.image_quality.enhanced && (
                     <p><em>Image was enhanced for better detection</em></p>
                   )}
                   <p><strong>Quality Level:</strong> <span className={`quality-${yoloDetections.image_quality.quality_level}`}>
                     {yoloDetections.image_quality.quality_level?.replace('_', ' ').toUpperCase()}
                   </span></p>
                 </div>
               )}
               
               {yoloDetections.persons && yoloDetections.persons.length > 0 && (
                 <div className="detections-list persons-list">
                   <h4 style={{color: '#4ade80', fontWeight: 'bold'}}>PERSON DETECTIONS:</h4>
                   <div className="detection-text">
                     {yoloDetections.persons.map((person, index) => (
                       <div key={index} className="person-detection">
                         <div className="person-header" style={{color: '#4ade80'}}>Person {index + 1}</div>
                         <div className="person-confidence" style={{color: '#4ade80'}}>{(person.confidence * 100).toFixed(1)}% confidence</div>
                         <div className="person-location" style={{color: '#4ade80'}}>Location: ({person.bbox?.map(coord => coord.toFixed(0)).join(', ')})</div>
                         {person.recognition && (
                           <div className="person-recognition">
                             <div className="recognition-label" style={{color: '#4ade80'}}>Recognition:</div>
                             <div className="recognition-result" style={{color: '#4ade80'}}>
                               ID {person.recognition.pred_class} ({(person.recognition.pred_conf * 100).toFixed(1)}% match)
                             </div>
                           </div>
                         )}
                       </div>
                     ))}
                   </div>
                 </div>
               )}
               
               {yoloDetections.weapons && yoloDetections.weapons.length > 0 && (
                 <div className="detections-list weapons-list">
                   <h4>WEAPON DETECTIONS:</h4>
                   <div className="detection-cards">
                     {yoloDetections.weapons.map((weapon, index) => (
                       <div key={index} className="detection-card weapon-card alert">
                         <div className="detection-header">
                           <span className="detection-index">Weapon {index + 1}</span>
                           <span className="detection-confidence confidence-high">
                             {(weapon.confidence * 100).toFixed(1)}% confidence
                           </span>
                         </div>
                         <div className="detection-details">
                           <span className="weapon-type">
                             Type: {weapon.weapon_type || weapon.class || 'Unknown'}
                           </span>
                           <span className="detection-location">
                             Location: ({weapon.bbox?.map(coord => coord.toFixed(0)).join(', ')})
                           </span>
                         </div>
                       </div>
                     ))}
                   </div>
                 </div>
               )}
             </div>
           )}
           {!yoloDetections && !isLoading && (
             <p>YOLO object detection results will appear here after processing.</p>
           )}
         </div>
       </div>

      {report && (
        <div className="report-container">
          <h3>Investigation Report</h3>
          <div className="report-content">
            {report}
          </div>
        </div>
      )}
    </div>
  );
}

export default Upload;
