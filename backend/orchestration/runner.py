"""
Template-based report generation for investigative analysis.
Generates structured reports from facial recognition and 3D reconstruction results.
"""

def run_report_generation(case_id: str, case_context: str, options: dict = None):
    """
    Generates a structured investigation report using template-based formatting.
    
    Args:
        case_id: Unique identifier for the case
        case_context: Background information and case details
        options: Dictionary containing matches, vertices, faces, and other analysis data
        
    Returns:
        str: Formatted investigation report text
    """
    options = options or {}
    
    matches = options.get("matches", [])
    vertices = options.get("vertices", [])
    faces = options.get("faces", [])
    
    # Handle vertices/faces - could be list, dict, or count
    vertex_count = len(vertices) if isinstance(vertices, (list, dict)) else (vertices if isinstance(vertices, int) else 0)
    face_count = len(faces) if isinstance(faces, (list, dict)) else (faces if isinstance(faces, int) else 0)
    
    # Generate structured report
    report = f"""**INVESTIGATION REPORT**
**Case ID:** {case_id}

**Case Summary**
{case_context}

**3D Reconstruction Analysis**
- Mesh Quality: {vertex_count} vertices, {face_count} faces
- Reconstruction Status: {'High quality' if vertex_count > 1000 else 'Basic reconstruction'}
- Topology: {'Complete mesh' if face_count > 500 else 'Simplified mesh'}

**Identity Recognition Results**
"""
    
    if matches:
        for i, match in enumerate(matches, 1):
            identity = match.get('identity_id') or match.get('identity') or match.get('name', 'Unknown')
            confidence = match.get('confidence', 0)
            
            # Handle confidence - could be float (0-1) or percentage string
            if isinstance(confidence, str):
                confidence_str = confidence
            else:
                confidence_str = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence:.1f}%"
            
            report += f"{i}. Identity: {identity}\n"
            report += f"   Confidence: {confidence_str}\n"
            
            status = match.get('status', match.get('watchlist_status', 'Unknown'))
            report += f"   Status: {status}\n"
            
            if match.get('similarity_score') is not None:
                sim_score = match.get('similarity_score')
                report += f"   Similarity Score: {sim_score:.3f}\n"
            elif match.get('similarity') is not None:
                sim_score = match.get('similarity')
                report += f"   Similarity: {sim_score}\n"
            report += "\n"
    else:
        report += "No identity matches found in the database.\n\n"
        
    report += """**Recommendations**
1. Review the 3D reconstruction for facial feature accuracy
2. Cross-reference recognition results with additional evidence
3. Consider manual verification of high-confidence matches
4. Document any additional contextual information
5. Ensure proper chain of custody for all evidence

**Report Generated:** Automated facial recognition analysis system
**Timestamp:** System timestamp applied at generation
"""
    
    return report
