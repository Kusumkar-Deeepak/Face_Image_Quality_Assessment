import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        # Footer dividing line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(54, 42, 612 - 54, 42)
        # Footer text
        self.drawString(54, 28, "Face Image Quality Assessment (FIQA) — Engineering Status Report")
        self.drawRightString(612 - 54, 28, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename="FIQA_Project_Status_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=45,
        bottomMargin=55,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    meta_bold = ParagraphStyle(
        "MetaBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
    )
    badge_done = ParagraphStyle(
        "BadgeDone",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#065F46"),
    )
    badge_prog = ParagraphStyle(
        "BadgeProg",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#92400E"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("Face Image Quality Assessment (FIQA) System", header_title_style))
    story.append(Paragraph("Engineering Status, Architectural Deliverables & Technical Report", subtitle_style))
    
    # Metadata Box
    meta_data = [
        [
            Paragraph("<b>Project:</b> eDifFIQA & UniFace Pipeline", meta_style),
            Paragraph("<b>Author / Engineer:</b> Deepak", meta_style),
        ],
        [
            Paragraph("<b>Organization:</b> Graymatics India Pvt. Ltd", meta_style),
            Paragraph("<b>Date:</b> October 2026", meta_style),
        ],
        [
            Paragraph("<b>Status:</b> Core Architecture & Localhost Complete", meta_style),
            Paragraph("<b>Deliverable:</b> Video Walkthrough + Codebase", meta_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 240])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=0))
    story.append(Paragraph(
        "This project implements a complete, end-to-end <b>Face Image Quality Assessment (FIQA)</b> application "
        "designed to evaluate face biometrics, predict utility scores, and enforce rigorous quality gates for biometric enrollment "
        "and surveillance verification. The system combines deep neural networks (eDifFIQA-S, SCRFD, RetinaFace, 6DRepNet, MobileGaze) "
        "with explicit photographic diagnostics and a reactive, user-friendly React/Vite dashboard backed by FastAPI.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Current Status:</b> The complete application is <b>100% operational on the local environment</b>, with real-time multi-face "
        "detection, quality scoring, landmark alignment, pose/gaze estimation, and batch ZIP/CSV exports verified. A full demo video has "
        "been recorded showcasing end-to-end execution. Cloud deployment to public URLs has been implemented, with technical optimizations "
        "applied to address free-tier memory constraints.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Section 1: Detailed Deliverables (What Is Done)
    story.append(Paragraph("1. Modules & Capabilities Delivered (What is Done)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=0))

    modules_data = [
        [Paragraph("Component", table_cell_bold), Paragraph("Technical Specification", table_cell_bold), Paragraph("Status", table_cell_bold)],
        
        [
            Paragraph("<b>eDifFIQA-S Scoring</b>", table_cell),
            Paragraph("Pretrained ResNet backbone predicting ISO/IEC compliant quality utility scores (0.0 to 1.0).", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>Multi-Stage Detection</b>", table_cell),
            Paragraph("UniFace SCRFD as primary detector + RetinaFace fallback with IoU deduplication for crowded scenes.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>Facial Alignment</b>", table_cell),
            Paragraph("5-point facial landmark affine transformation generating normalized 112×112 face crops.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>Head-Pose Estimation</b>", table_cell),
            Paragraph("UniFace 6DRepNet estimating real 3D Euler angles (Pitch, Yaw, Roll) with configurable thresholds.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>Gaze Estimation</b>", table_cell),
            Paragraph("UniFace MobileGaze estimating pitch and yaw gaze vectors to detect direct camera focus.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>Explainable Diagnostics</b>", table_cell),
            Paragraph("Laplacian variance (blur/sharpness), exposure histograms (under/overexposure), and illumination asymmetry.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>6-State Gate Engine</b>", table_cell),
            Paragraph("PASS, REVIEW, REJECT, NO_FACE, NOT_EVALUATED, ERROR based on calibrated operational thresholds.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>FastAPI Backend</b>", table_cell),
            Paragraph("Asynchronous chunk streaming, dynamic threshold calibration API, session stats, and latency metrics.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
        [
            Paragraph("<b>React/Vite Dashboard</b>", table_cell),
            Paragraph("Single-image inspection, landmark visualization, live batch progress bar, and 1-click ZIP/CSV export.", table_cell),
            Paragraph("COMPLETED", badge_done)
        ],
    ]

    mod_table = Table(modules_data, colWidths=[130, 310, 70])
    mod_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 10))

    # Section 2: Technical Challenges & Solutions
    story.append(Paragraph("2. Technical Challenges Faced & Engineering Solutions", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=0))

    story.append(Paragraph("<b>Challenge A: Missed Faces in Dense / Crowded Surveillance Photos</b>", h2_style))
    story.append(Paragraph(
        "• <i>Issue:</i> Standard single-stage detectors missed smaller, partially occluded, or boundary-adjacent faces in crowded tourist images.<br/>"
        "• <i>Solution:</i> Implemented a two-stage hybrid detector chain. UniFace SCRFD acts as primary for speed, while RetinaFace runs as a complementary fallback. An IoU-based deduplication algorithm merges overlapping bounding boxes without suppressing adjacent distinct individuals.",
        body_style
    ))

    story.append(Paragraph("<b>Challenge B: Deep Learning Multi-Model RAM Footprint vs. 512 MB Cloud Limits</b>", h2_style))
    story.append(Paragraph(
        "• <i>Issue:</i> Running 5 neural networks simultaneously (SCRFD + RetinaFace + eDifFIQA + 6DRepNet + MobileGaze) in PyTorch/ONNX on CPU naturally requires <b>~700–830 MB of RAM</b>. On free-tier cloud containers (such as Render) with a hard 512 MB ceiling, batch inference triggered the Linux Out-Of-Memory (OOM) killer, resulting in HTTP 502 Bad Gateway errors.<br/>"
        "• <i>Solutions Applied:</i><br/>"
        "&nbsp;&nbsp;1. <b>Shared Singleton Weights:</b> Prevented redundant duplicate 200MB model re-loads in <code>QualityPipeline</code>.<br/>"
        "&nbsp;&nbsp;2. <b>Kernel Memory Trimming:</b> Injected <code>ctypes.CDLL('libc.so.6').malloc_trim(0)</code> to immediately return unmapped memory to the OS.<br/>"
        "&nbsp;&nbsp;3. <b>Micro-Chunk Streaming:</b> Configured the React batch client to stream files in micro-chunks of 2, bounding memory spikes.<br/>"
        "&nbsp;&nbsp;4. <b>Safe Pre-Scaling:</b> Automatically downscaled 4K/oversized images to 1920px max dimension, cutting raw bitmap memory by >80%.<br/>"
        "&nbsp;&nbsp;5. <b>Dual Deployment:</b> In addition to Render, deployed a public instance on <b>Hugging Face Spaces (16 GB Free RAM)</b> where large concurrent multi-face batches run without any memory constraints.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 3: Project Status Matrix
    story.append(Paragraph("3. Current Status Matrix & Scope Overview", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=0))

    status_data = [
        [Paragraph("Environment / Capability", table_cell_bold), Paragraph("Observed Performance & Behavior", table_cell_bold), Paragraph("Current State", table_cell_bold)],
        [
            Paragraph("<b>Localhost Environment</b>", table_cell),
            Paragraph("Full functionality, high speed, unlimited batch capacity, all 5 models run concurrently. Demo video captured.", table_cell),
            Paragraph("100% READY", badge_done)
        ],
        [
            Paragraph("<b>Hugging Face Spaces</b><br/>(16 GB RAM Tier)", table_cell),
            Paragraph("Runs full multi-model pipeline without memory bottlenecks. Ideal public platform for stakeholder review.", table_cell),
            Paragraph("DEPLOYED & LIVE", badge_done)
        ],
        [
            Paragraph("<b>Render Cloud</b><br/>(512 MB Free Tier)", table_cell),
            Paragraph("Operational for single images and lightweight batches. Heavy multi-model batches require high-RAM tier or light preset.", table_cell),
            Paragraph("CONSTRAINED", badge_prog)
        ],
        [
            Paragraph("<b>Operational Threshold Calibration</b>", table_cell),
            Paragraph("Engine supports real-time threshold updates via API/UI. Final calibration pending customer domain dataset.", table_cell),
            Paragraph("IN PROGRESS", badge_prog)
        ],
    ]

    status_table = Table(status_data, colWidths=[140, 300, 70])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 10))

    # Section 4: Next Steps
    story.append(Paragraph("4. Next Steps & Recommendations", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=0))
    story.append(Paragraph(
        "1. <b>Video Review:</b> Review the recorded video demonstration showcasing localhost performance.<br/>"
        "2. <b>Customer Dataset Calibration:</b> Establish domain-specific quality score acceptance/rejection thresholds based on Graymatics target camera feeds.<br/>"
        "3. <b>Production Hosting Architecture:</b> For production deployment of multi-face batches, a compute instance with ≥2 GB RAM is recommended to support all 5 models comfortably under peak load.",
        body_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {filename}")

if __name__ == "__main__":
    build_pdf()
