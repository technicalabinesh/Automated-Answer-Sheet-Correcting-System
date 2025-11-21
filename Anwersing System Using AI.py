import gradio as gr
import json
import os
from PyPDF2 import PdfReader
from docx import Document
import re
from datetime import datetime

# ReportLab imports for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠ ReportLab not available. Please install: pip install reportlab")

# IBM Watsonx imports
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai import Credentials
    IBM_WATSON_AVAILABLE = True
except ImportError:
    IBM_WATSON_AVAILABLE = False
    print("⚠ IBM Watsonx libraries not available. Please install: pip install ibm-watsonx-ai")

# Global variables
watsonx_model = None

# === Initialize IBM Watsonx ===
def initialize_watsonx(api_key, project_id, url="https://us-south.ml.cloud.ibm.com"):
    """Initialize IBM Watsonx AI model"""
    global watsonx_model
    
    if not IBM_WATSON_AVAILABLE:
        return "⚠ IBM Watsonx libraries not installed. Please install: pip install ibm-watsonx-ai"
    
    if not api_key or not project_id:
        return "⚠ Please provide both API Key and Project ID"
    
    try:
        credentials = Credentials(api_key=api_key, url=url)
        
        model_params = {
            GenParams.MAX_NEW_TOKENS: 2000,
            GenParams.MIN_NEW_TOKENS: 50,
            GenParams.TEMPERATURE: 0.3,
            GenParams.TOP_P: 0.9,
        }
        
        watsonx_model = Model(
            model_id='mistralai/mistral-small-3-1-24b-instruct-2503',
            params=model_params,
            credentials=credentials,
            project_id=project_id
        )
        
        return "✅ IBM Watsonx initialized successfully! You can now use all AI features."
    except Exception as e:
        watsonx_model = None
        return f"❌ Failed to initialize Watsonx: {str(e)}"

def generate_watsonx_text(prompt):
    """Generate text using IBM Watsonx"""
    global watsonx_model
    
    if watsonx_model is None:
        return "Error: Watsonx not initialized. Please configure API credentials first in the Watsonx Setup tab."
    
    try:
        response = watsonx_model.generate_text(prompt=prompt)
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"Error generating text: {str(e)}"

# === File Text Extraction ===
def extract_text_from_file(file_path):
    """Extract text from uploaded file (PDF, DOCX, or TXT)"""
    if not file_path:
        return ""
    
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Extract from PDF
        if file_extension == '.pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        # Extract from DOCX
        elif file_extension in ['.docx', '.doc']:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        
        # Extract from TXT
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            return f"Unsupported file format: {file_extension}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# === Enhanced Keyword Extraction ===
def extract_keywords_from_teacher_answer(teacher_answer):
    """Extract important keywords and concepts from teacher's answer"""
    # Remove common words
    common_words = {
        'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 
        'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'also', 'they', 'them', 'their'
    }
    
    # Tokenize and clean - extract words 3+ characters
    words = re.findall(r'\b[a-zA-Z]{3,}\b', teacher_answer.lower())
    
    # Count word frequency
    word_freq = {}
    for word in words:
        if word not in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords (prioritize words with frequency > 1)
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_keywords[:25]]  # Top 25 keywords
    
    return keywords

# === Advanced Keyword Checking ===
def check_keywords_offline(student_answer, keywords_list, total_marks=100):
    """Enhanced offline keyword checking with partial matching"""
    if not student_answer or not keywords_list:
        return {
            'score': 0,
            'marks': 0,
            'found_keywords': [],
            'missing_keywords': [],
            'total_keywords': 0,
            'coverage': 0
        }
    
    # Normalize
    answer_lower = student_answer.lower()
    keywords = [kw.lower().strip() for kw in keywords_list if kw]
    
    # Check which keywords are present
    found_keywords = []
    missing_keywords = []
    
    for kw in keywords:
        if kw in answer_lower:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)
    
    # Calculate score and marks
    total_keywords = len(keywords)
    found_count = len(found_keywords)
    
    score = round((found_count / total_keywords) * 100) if total_keywords > 0 else 0
    marks = round((score / 100) * total_marks, 2)
    coverage = round((found_count / total_keywords) * 100, 1) if total_keywords > 0 else 0
    
    return {
        'score': score,
        'marks': marks,
        'found_keywords': found_keywords,
        'missing_keywords': missing_keywords,
        'total_keywords': total_keywords,
        'coverage': coverage
    }

# === Enhanced AI Comparison ===
def compare_answers_with_ai(question, teacher_answer, student_answer, keyword_result, total_marks):
    """Use IBM Watsonx to deeply compare answers and provide detailed feedback"""
    try:
        prompt = f"""You are an expert educational evaluator analyzing student work. Your job is to:
1. Compare the student's answer with the teacher's model answer
2. Evaluate understanding, accuracy, and completeness
3. Provide constructive feedback
4. Assign a fair grade

QUESTION:
{question}

TEACHER'S MODEL ANSWER:
{teacher_answer}

STUDENT'S ANSWER:
{student_answer}

KEYWORD ANALYSIS (Preliminary Assessment):
- Keywords Coverage: {keyword_result['coverage']}%
- Found Keywords: {', '.join(keyword_result['found_keywords'][:10])}
- Missing Keywords: {', '.join(keyword_result['missing_keywords'][:10])}
- Preliminary Marks: {keyword_result['marks']}/{total_marks}

Please provide a comprehensive evaluation in JSON format:

{{
  "finalScore": <0-100 percentage score based on your evaluation>,
  "finalMarks": <calculated marks out of {total_marks}>,
  "accuracyRating": "<Excellent/Good/Fair/Poor>",
  "feedback": "Detailed comparison of student answer vs model answer. What did they cover? What did they miss?",
  "strengths": "Specific points the student answered well",
  "weaknesses": "Specific concepts or details the student missed or got wrong",
  "improvements": "Clear, actionable suggestions for improvement",
  "improvedAnswer": "A corrected and enhanced version of the student's answer incorporating missing elements",
  "gradingJustification": "Explain why this grade was assigned",
  "keyConceptsCovered": ["list", "of", "key", "concepts", "student", "covered"],
  "keyConceptsMissed": ["list", "of", "important", "concepts", "not", "addressed"]
}}

IMPORTANT: 
- Be fair and thorough in your evaluation
- Don't just rely on keyword matching
- Consider conceptual understanding
- Evaluate accuracy of information
- Check if the answer addresses the question
- Return ONLY valid JSON, no additional text"""

        response = generate_watsonx_text(prompt)
        
        # Clean JSON response
        if 'Error' in response and 'Watsonx not initialized' in response:
            raise Exception("Watsonx not initialized")
        
        # Extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response.split('```json')[1].split('```')[0].strip()
        elif response.startswith('```'):
            response = response.split('```')[1].split('```')[0].strip()
        
        # Parse JSON
        ai_response = json.loads(response)
        
        return {
            'final_score': ai_response.get('finalScore', keyword_result['score']),
            'final_marks': ai_response.get('finalMarks', keyword_result['marks']),
            'accuracy_rating': ai_response.get('accuracyRating', 'Fair'),
            'feedback': ai_response.get('feedback', 'No feedback available'),
            'strengths': ai_response.get('strengths', 'No strengths identified'),
            'weaknesses': ai_response.get('weaknesses', 'No weaknesses identified'),
            'improvements': ai_response.get('improvements', 'No improvements identified'),
            'improved_answer': ai_response.get('improvedAnswer', 'No improved answer available'),
            'justification': ai_response.get('gradingJustification', 'No justification available'),
            'concepts_covered': ai_response.get('keyConceptsCovered', []),
            'concepts_missed': ai_response.get('keyConceptsMissed', []),
            'ai_enabled': True
        }
        
    except Exception as e:
        # Fallback to keyword-based grading if AI fails
        return {
            'final_score': keyword_result['score'],
            'final_marks': keyword_result['marks'],
            'accuracy_rating': 'Fair' if keyword_result['score'] >= 50 else 'Poor',
            'feedback': f"AI analysis unavailable. Grading based on keyword matching only. Error: {str(e)}",
            'strengths': f"Found {len(keyword_result['found_keywords'])} key concepts.",
            'weaknesses': f"Missing {len(keyword_result['missing_keywords'])} important concepts.",
            'improvements': "Review the missing keywords and ensure your answer covers all key concepts.",
            'improved_answer': "AI-generated improved answer unavailable. Please review teacher's answer.",
            'justification': f"Score calculated based on keyword coverage: {keyword_result['coverage']}%",
            'concepts_covered': keyword_result['found_keywords'][:5],
            'concepts_missed': keyword_result['missing_keywords'][:5],
            'ai_enabled': False
        }

# === Generate PDF Report ===
def generate_pdf_report(question, teacher_answer, student_answer, keyword_result, ai_result, total_marks):
    """Generate a comprehensive PDF report using ReportLab"""
    
    if not REPORTLAB_AVAILABLE:
        return None, "⚠️ ReportLab not installed. Install with: pip install reportlab"
    
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grading_report_{timestamp}.pdf"
        filepath = os.path.join("/tmp", filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#3949ab'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph("📚 Answer Sheet Grading Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Report Info
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Determine grade
        score = ai_result['final_score']
        marks = ai_result['final_marks']
        
        if score >= 80:
            grade = "Excellent (A)"
            grade_color = colors.green
        elif score >= 60:
            grade = "Good (B)"
            grade_color = colors.blue
        elif score >= 40:
            grade = "Fair (C)"
            grade_color = colors.orange
        else:
            grade = "Needs Improvement (D)"
            grade_color = colors.red
        
        # Score Summary Table
        story.append(Paragraph("📊 FINAL GRADE SUMMARY", heading_style))
        
        score_data = [
            ['Metric', 'Result'],
            ['Overall Grade', grade],
            ['Marks Obtained', f"{marks} / {total_marks}"],
            ['Percentage Score', f"{score}%"],
            ['Accuracy Rating', ai_result['accuracy_rating']],
            ['Keyword Coverage', f"{keyword_result['coverage']}%"]
        ]
        
        score_table = Table(score_data, colWidths=[3*inch, 3*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(score_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Question
        story.append(Paragraph("📌 QUESTION / TOPIC", heading_style))
        story.append(Paragraph(question, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Keyword Analysis
        story.append(Paragraph("🔍 KEYWORD ANALYSIS", heading_style))
        
        story.append(Paragraph(f"<b>Total Keywords:</b> {keyword_result['total_keywords']}", body_style))
        story.append(Paragraph(f"<b>Keywords Found:</b> {len(keyword_result['found_keywords'])}", body_style))
        story.append(Paragraph(f"<b>Keywords Missing:</b> {len(keyword_result['missing_keywords'])}", body_style))
        
        if keyword_result['found_keywords']:
            found_text = ', '.join(keyword_result['found_keywords'][:15])
            story.append(Paragraph(f"<b>✅ Found:</b> {found_text}", body_style))
        
        if keyword_result['missing_keywords']:
            missing_text = ', '.join(keyword_result['missing_keywords'][:15])
            story.append(Paragraph(f"<b>❌ Missing:</b> {missing_text}", body_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Key Concepts
        story.append(Paragraph("🎯 KEY CONCEPTS EVALUATION", heading_style))
        
        if ai_result['concepts_covered']:
            covered_text = ', '.join(ai_result['concepts_covered'])
            story.append(Paragraph(f"<b>✅ Covered Topics:</b> {covered_text}", body_style))
        
        if ai_result['concepts_missed']:
            missed_text = ', '.join(ai_result['concepts_missed'])
            story.append(Paragraph(f"<b>❌ Missed Topics:</b> {missed_text}", body_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # AI Feedback
        story.append(Paragraph("💬 DETAILED EVALUATION", heading_style))
        story.append(Paragraph(ai_result['feedback'], body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Strengths
        story.append(Paragraph("✨ STRENGTHS", subheading_style))
        story.append(Paragraph(ai_result['strengths'], body_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Weaknesses
        story.append(Paragraph("⚠️ WEAKNESSES", subheading_style))
        story.append(Paragraph(ai_result['weaknesses'], body_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Improvements
        story.append(Paragraph("🔧 RECOMMENDATIONS FOR IMPROVEMENT", subheading_style))
        story.append(Paragraph(ai_result['improvements'], body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Page break before improved answer
        story.append(PageBreak())
        
        # Improved Answer
        story.append(Paragraph("✍️ MODEL IMPROVED ANSWER", heading_style))
        story.append(Paragraph(ai_result['improved_answer'], body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Grading Justification
        story.append(Paragraph("📚 GRADING JUSTIFICATION", heading_style))
        story.append(Paragraph(ai_result['justification'], body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Page break before reference materials
        story.append(PageBreak())
        
        # Teacher's Answer
        story.append(Paragraph("📖 TEACHER'S MODEL ANSWER", heading_style))
        teacher_text = teacher_answer[:1000] + "..." if len(teacher_answer) > 1000 else teacher_answer
        story.append(Paragraph(teacher_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Student's Answer
        story.append(Paragraph("📝 STUDENT'S SUBMITTED ANSWER", heading_style))
        student_text = student_answer[:1000] + "..." if len(student_answer) > 1000 else student_answer
        story.append(Paragraph(student_text, body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        grading_method = 'AI-Enhanced Evaluation (IBM Watsonx + Keyword Analysis)' if ai_result['ai_enabled'] else 'Keyword-Based Evaluation Only'
        story.append(Paragraph(f"<b>Grading Method:</b> {grading_method}", body_style))
        story.append(Paragraph("Report generated by AI Answer Sheet Grading System", body_style))
        
        # Build PDF
        doc.build(story)
        
        return filepath, "✅ PDF report generated successfully!"
        
    except Exception as e:
        return None, f"❌ Error generating PDF: {str(e)}"

# === Main Processing Function ===
def process_answer_sheets(question, teacher_file, student_file, total_marks, auto_extract_keywords):
    """Main function that processes both answer sheets"""
    
    # Validate inputs
    if not question:
        return "⚠️ Please enter a question!", "", "", "", ""
    
    if not teacher_file:
        return "⚠️ Please upload teacher's answer sheet!", "", "", "", ""
    
    if not student_file:
        return "⚠️ Please upload student's answer sheet!", "", "", "", ""
    
    # Extract text from files
    status_msg = "📄 **Step 1:** Extracting text from uploaded files...\n\n"
    
    teacher_answer = extract_text_from_file(teacher_file)
    student_answer = extract_text_from_file(student_file)
    
    if "Error" in teacher_answer or "Unsupported" in teacher_answer:
        return f"❌ Teacher's file error: {teacher_answer}", "", "", "", ""
    
    if "Error" in student_answer or "Unsupported" in student_answer:
        return f"❌ Student's file error: {student_answer}", "", "", "", ""
    
    status_msg += f"✅ Teacher's answer: {len(teacher_answer)} characters\n"
    status_msg += f"✅ Student's answer: {len(student_answer)} characters\n\n"
    
    # Extract keywords
    status_msg += "📄 **Step 2:** Extracting keywords from teacher's answer...\n\n"
    keywords_list = extract_keywords_from_teacher_answer(teacher_answer)
    status_msg += f"✅ Extracted {len(keywords_list)} important keywords\n\n"
    
    # Offline keyword checking
    status_msg += "📄 **Step 3:** Performing keyword analysis...\n\n"
    keyword_result = check_keywords_offline(student_answer, keywords_list, total_marks)
    status_msg += f"✅ Keyword coverage: {keyword_result['coverage']}%\n"
    status_msg += f"✅ Preliminary marks: {keyword_result['marks']}/{total_marks}\n\n"
    
    # AI comparison
    status_msg += "📄 **Step 4:** AI-powered deep analysis using IBM Watsonx...\n\n"
    ai_result = compare_answers_with_ai(question, teacher_answer, student_answer, keyword_result, total_marks)
    
    if ai_result['ai_enabled']:
        status_msg += "✅ AI analysis complete!\n\n"
    else:
        status_msg += "⚠️ AI analysis unavailable - using keyword-based grading\n\n"
    
    status_msg += "🎉 **Grading Complete!**\n\n"
    
    # Determine grade
    score = ai_result['final_score']
    marks = ai_result['final_marks']
    
    if score >= 80:
        score_emoji = "🟢"
        grade = "Excellent"
    elif score >= 60:
        score_emoji = "🟡"
        grade = "Good"
    elif score >= 40:
        score_emoji = "🟠"
        grade = "Fair"
    else:
        score_emoji = "🔴"
        grade = "Needs Improvement"
    
    # Score Report
    score_text = f"""
# {score_emoji} Final Grade Report

## 📊 **Overall Assessment: {grade}**

### Final Score
- **Marks Obtained:** {marks} / {total_marks}
- **Percentage:** {score}%
- **Accuracy Rating:** {ai_result['accuracy_rating']}

---

## 📈 Keyword Analysis

### ✅ Concepts Found ({len(keyword_result['found_keywords'])}/{keyword_result['total_keywords']})
{', '.join(keyword_result['found_keywords'][:15]) if keyword_result['found_keywords'] else 'None'}

### ❌ Concepts Missing ({len(keyword_result['missing_keywords'])})
{', '.join(keyword_result['missing_keywords'][:15]) if keyword_result['missing_keywords'] else 'None'}

---

## 🎯 Key Concepts Evaluation

### ✅ Covered Topics
{', '.join(ai_result['concepts_covered']) if ai_result['concepts_covered'] else 'Not analyzed'}

### ❌ Missed Topics
{', '.join(ai_result['concepts_missed']) if ai_result['concepts_missed'] else 'Not analyzed'}
"""

    # Detailed Feedback
    feedback_text = f"""
# 💬 Detailed Evaluation

## 📝 Comparison Analysis
{ai_result['feedback']}

---

## ✨ Strengths
{ai_result['strengths']}

---

## ⚠️ Weaknesses
{ai_result['weaknesses']}

---

## 🔧 Recommendations for Improvement
{ai_result['improvements']}
"""

    # Improved Answer
    improved_text = f"""
# ✍️ Model Improved Answer

{ai_result['improved_answer']}

---

**Note:** This is how your answer could be enhanced to earn full marks.
"""

    # Grading Explanation
    explanation_text = f"""
# 📚 Grading Justification

{ai_result['justification']}

---

## 📖 Reference Materials

### Teacher's Model Answer (First 600 chars):
```
{teacher_answer[:600]}...
```

### Student's Submitted Answer (First 600 chars):
```
{student_answer[:600]}...
```

---

### 🔍 Keywords Used for Evaluation:
{', '.join(keywords_list[:20])}

---

### ⚙️ Grading Method:
{'AI-Enhanced Evaluation (IBM Watsonx + Keyword Analysis)' if ai_result['ai_enabled'] else 'Keyword-Based Evaluation Only'}
"""

    return status_msg, score_text, feedback_text, improved_text, explanation_text


# === Wrapper Function for PDF Generation ===
def generate_report_pdf(question, teacher_file, student_file, total_marks, auto_extract_keywords):
    """Generate PDF report after processing"""
    
    if not question or not teacher_file or not student_file:
        return None, "⚠️ Please complete grading first before generating PDF report!"
    
    # Extract text from files
    teacher_answer = extract_text_from_file(teacher_file)
    student_answer = extract_text_from_file(student_file)
    
    if "Error" in teacher_answer or "Error" in student_answer:
        return None, "❌ Error reading files. Please try again."
    
    # Extract keywords and perform analysis
    keywords_list = extract_keywords_from_teacher_answer(teacher_answer)
    keyword_result = check_keywords_offline(student_answer, keywords_list, total_marks)
    ai_result = compare_answers_with_ai(question, teacher_answer, student_answer, keyword_result, total_marks)
    
    # Generate PDF
    pdf_path, status = generate_pdf_report(
        question, teacher_answer, student_answer, 
        keyword_result, ai_result, total_marks
    )
    
    if pdf_path:
        return pdf_path, status
    else:
        return None, status


# === Gradio Interface ===
with gr.Blocks(theme=gr.themes.Soft(), title="AI Answer Sheet Grading System") as demo:
    
    gr.Markdown("""
    # 📚 AI-Powered Answer Sheet Grading System
    ### Powered by IBM Watsonx AI
    
    Upload teacher's model answer and student's answer to get:
    - ✅ Automatic grading with marks
    - 📊 Detailed feedback and analysis
    - 💡 Improvement suggestions
    - ✍️ Model improved answer
    
    **Supports PDF, DOCX, and TXT files**
    """)
    
    # Watsonx Setup Tab
    with gr.Tab("🔧 IBM Watsonx Setup"):
        gr.Markdown("### Configure IBM Watsonx AI Credentials")
        gr.Markdown("""
        **Get your credentials:**
        1. Visit [IBM Cloud](https://cloud.ibm.com/)
        2. Create a Watsonx.ai project
        3. Get your API Key and Project ID
        """)
        
        with gr.Row():
            api_key_input = gr.Textbox(
                label="🔑 API Key",
                type="password",
                placeholder="Enter your IBM Watsonx API Key"
            )
            project_id_input = gr.Textbox(
                label="📋 Project ID",
                placeholder="Enter your Project ID"
            )
        
        url_input = gr.Textbox(
            label="🌐 Watsonx URL",
            value="https://us-south.ml.cloud.ibm.com"
        )
        
        init_btn = gr.Button("🚀 Initialize Watsonx", variant="primary", size="lg")
        init_status = gr.Textbox(label="Status", interactive=False, lines=3)
        
        init_btn.click(
            initialize_watsonx,
            inputs=[api_key_input, project_id_input, url_input],
            outputs=init_status
        )
    
    # Main Grading Tab
    with gr.Tab("📝 Grade Answer Sheets"):
        gr.Markdown("### Upload and Compare Answer Sheets")
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="📌 Question / Topic",
                    placeholder="Enter the question or topic...",
                    lines=4
                )
                
                teacher_file = gr.File(
                    label="📄 Teacher's Model Answer (PDF/DOCX/TXT)",
                    file_types=['.pdf', '.docx', '.doc', '.txt']
                )
                
                student_file = gr.File(
                    label="📝 Student's Answer (PDF/DOCX/TXT)",
                    file_types=['.pdf', '.docx', '.doc', '.txt']
                )
                
                with gr.Row():
                    total_marks = gr.Number(
                        label="💯 Total Marks",
                        value=100,
                        minimum=1,
                        maximum=1000
                    )
                    auto_extract = gr.Checkbox(
                        label="🔍 Auto-extract keywords",
                        value=True
                    )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🚀 Grade Answer",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.ClearButton(
                        components=[question_input, teacher_file, student_file],
                        value="🔄 Reset",
                        size="lg"
                    )
        
        with gr.Column():
            status_output = gr.Markdown(label="📊 Processing Status")
        
        with gr.Row():
            score_output = gr.Markdown(label="📈 Score Report")
        
        with gr.Row():
            feedback_output = gr.Markdown(label="💬 Detailed Feedback")
        
        with gr.Row():
            with gr.Column():
                improved_output = gr.Markdown(label="✍️ Improved Answer")
            with gr.Column():
                explanation_output = gr.Markdown(label="📚 Grading Explanation")
        
        # PDF Download Section
        gr.Markdown("---")
        gr.Markdown("## 📄 Download Complete Report")
        
        with gr.Row():
            pdf_download_btn = gr.Button(
                "📥 Generate & Download PDF Report",
                variant="secondary",
                size="lg"
            )
        
        with gr.Row():
            pdf_output = gr.File(label="📑 PDF Report", interactive=False)
            pdf_status = gr.Textbox(label="PDF Generation Status", interactive=False, lines=2)
        
        # Button events
        submit_btn.click(
            fn=process_answer_sheets,
            inputs=[question_input, teacher_file, student_file, total_marks, auto_extract],
            outputs=[status_output, score_output, feedback_output, improved_output, explanation_output]
        )
        
        pdf_download_btn.click(
            fn=generate_report_pdf,
            inputs=[question_input, teacher_file, student_file, total_marks, auto_extract],
            outputs=[pdf_output, pdf_status]
        )
    
    # Help Tab
    with gr.Tab("❓ Help & Documentation"):
        gr.Markdown("""
        ## 📖 How to Use
        
        ### Step 1: Setup Watsonx
        1. Go to "IBM Watsonx Setup" tab
        2. Enter your API credentials
        3. Click "Initialize Watsonx"
        4. Wait for success message
        
        ### Step 2: Grade Answers
        1. Enter the question/topic
        2. Upload teacher's model answer
        3. Upload student's answer
        4. Set total marks
        5. Click "Grade Answer"
        6. Review comprehensive results
        
        ---
        
        ## 🎯 Features
        
        ✅ **AI-Powered Analysis** - Deep comparison using IBM Watsonx  
        ✅ **Keyword Extraction** - Automatic identification of key concepts  
        ✅ **Multi-Format Support** - PDF, DOCX, TXT files  
        ✅ **Detailed Feedback** - Strengths, weaknesses, improvements  
        ✅ **Model Answers** - AI-generated improved versions  
        ✅ **Fair Grading** - Combines keyword matching + AI evaluation  
        ✅ **PDF Reports** - Professional downloadable reports with ReportLab  
        
        ---
        
        ## 📥 Generating PDF Reports
        
        1. Complete the grading process first
        2. Click "Generate & Download PDF Report"
        3. Wait for PDF generation
        4. Download the comprehensive report
        
        **PDF Report Includes:**
        - Complete grade summary with visual tables
        - Keyword and concept analysis
        - Detailed AI feedback
        - Strengths and weaknesses
        - Improvement recommendations
        - Model improved answer
        - Reference to teacher's and student's answers
        - Professional formatting with colors and structure  
        
        ---
        
        ## 📋 Grading Criteria
        
        The system evaluates:
        - **Content Coverage** - Are key concepts present?
        - **Accuracy** - Is the information correct?
        - **Completeness** - Does it fully address the question?
        - **Understanding** - Does the student understand the topic?
        
        ---
        
        ## 🔧 Troubleshooting
        
        **Watsonx Issues:**
        - Ensure API credentials are correct
        - Check internet connection
        - Verify Watsonx service is active
        
        **File Reading Issues:**
        - Use PDF, DOCX, or TXT format
        - Ensure file is not corrupted
        - Check text is readable (not scanned images)
        
        ---
        
        ## ⚠️ Important Notes
        
        - AI analysis provides guidance, not absolute truth
        - Always verify results with human judgment
        - Best for educational and formative assessment
        - Not a replacement for teacher evaluation
        
        ---
        
        ## 📦 Requirements
        
        ```bash
        pip install gradio ibm-watsonx-ai PyPDF2 python-docx reportlab
        ```
        
        ---
        
        ## 🔒 Privacy
        
        - Files processed in session only
        - No permanent storage
        - Data not shared externally
        """)

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 AI Answer Sheet Grading System")
    print("Powered by IBM Watsonx AI")
    print("="*60)
    print("\n✨ Features:")
    print("• AI-powered answer comparison")
    print("• Automatic keyword extraction")
    print("• Detailed feedback generation")
    print("• Multi-format file support")
    print("• Comprehensive grading reports")
    print("• Professional PDF report generation")
    print("\n📦 Requirements:")
    print("• pip install gradio ibm-watsonx-ai PyPDF2 python-docx reportlab")
    print("\n" + "="*60 + "\n")
    
    demo.launch(share=False, debug=True)
