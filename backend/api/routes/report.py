from fastapi import APIRouter, Depends, HTTPException, status, Response
from backend.api.models import schemas
from backend.orchestration.runner import run_report_generation
from fpdf import FPDF
from datetime import datetime

router = APIRouter()

@router.post("/report", response_class=Response)
async def generate_report(
    report_request: dict
):
    """
    Generates a structured investigation report from facial recognition analysis.

    This endpoint collects reconstruction and recognition results and generates
    a formal, template-based investigative report in PDF format.

    Args:
        report_request (dict): Request body containing:
            - case_id: Unique case identifier
            - case_context: Investigation background
            - matches: List of identity match results
            - vertices: 3D mesh vertex data
            - faces: 3D mesh face data

    Returns:
        Response: PDF file containing the investigation report

    Raises:
        HTTPException: If report generation fails
    """
    try:
        print(f"=== Report Request Debug ===")
        print(f"Request type: {type(report_request)}")
        print(f"Request data: {report_request}")
        
        case_id = report_request.get('case_id', 'Unknown')
        case_context = report_request.get('case_context', 'No context provided')
        matches = report_request.get('matches', [])
        vertices = report_request.get('vertices', [])
        faces = report_request.get('faces', [])
        
        print(f"Parsed - Case ID: {case_id}")
        print(f"Parsed - Matches count: {len(matches)}")
        print(f"Parsed - Vertices: {type(vertices)}, Faces: {type(faces)}")
        
        # Generate report text using template-based system
        report_text = run_report_generation(
            case_id=case_id,
            case_context=case_context,
            options={
                'matches': matches,
                'vertices': vertices,
                'faces': faces
            }
        )
        
        print(f"Report text generated, length: {len(report_text)}")
        
        # Generate PDF with proper margins and safer rendering
        pdf = FPDF()
        pdf.add_page()
        # Use more conservative margins to ensure content fits
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=20)
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="Investigation Report", ln=True, align='C')
        pdf.ln(5)
        
        # Add timestamp
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(8)

        # Add report content with safer text handling
        pdf.set_font("Arial", size=11)
        
        # Process report line by line with error handling
        for line in report_text.split('\n'):
            # Skip empty lines
            if not line.strip():
                pdf.ln(3)
                continue
            
            # Clean the line
            clean_line = line.strip()
            
            # Limit line length to prevent overflow
            if len(clean_line) > 150:
                # Use textwrap for very long lines
                import textwrap
                wrapped = textwrap.wrap(clean_line, width=90, break_long_words=False, break_on_hyphens=False)
                for wrapped_line in wrapped:
                    if wrapped_line.strip():
                        try:
                            pdf.set_font("Arial", size=10)
                            pdf.cell(0, 5, txt=wrapped_line, ln=True)
                        except Exception as e:
                            print(f"PDF rendering error (wrapped): {e}")
                            pdf.cell(0, 5, txt="[Content too long to display]", ln=True)
                pdf.set_font("Arial", size=11)
                continue
            
            # Handle section headers (bold)
            if clean_line.startswith('**') and clean_line.endswith('**'):
                header_text = clean_line.strip('*').strip()
                if header_text:
                    try:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 7, txt=header_text, ln=True)
                        pdf.set_font("Arial", size=11)
                    except Exception as e:
                        print(f"PDF rendering error (header): {e}")
            else:
                # Regular content
                try:
                    pdf.cell(0, 6, txt=clean_line, ln=True)
                except Exception as e:
                    print(f"PDF rendering error (content): {e}")
                    # Try with multi_cell as fallback
                    try:
                        pdf.multi_cell(0, 6, clean_line)
                    except:
                        pass  # Skip problematic lines

        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, bytearray):
            pdf_output = bytes(pdf_output)
        
        return Response(
            content=pdf_output,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=investigation_report_{report_request.get('case_id', 'unknown')}.pdf"
            }
        )
        
    except Exception as e:
        print(f"Report generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")




