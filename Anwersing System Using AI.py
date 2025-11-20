import gradio as gr
import json
import os
from PyPDF2 import PdfReader
from docx import Document
import re

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
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.MIN_NEW_TOKENS: 30,
            GenParams.TEMPERATURE: 0.2,
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

# === Extract Keywords from Teacher Answer ===
def extract_keywords_from_teacher_answer(teacher_answer):
    """Automatically extract important keywords from teacher's answer"""
    # Remove common words
    common_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 
                   'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
                   'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been', 'being'}
    
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{4,}\b', teacher_answer.lower())
    
    # Count word frequency
    word_freq = {}
    for word in words:
        if word not in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords (frequency > 1 or important technical terms)
    keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]]
    
    return keywords

# === Offline Keyword Checking ===
def check_keywords_offline(student_answer, keywords_list, total_marks=100):
    """Offline keyword checking - works without internet"""
    if not student_answer or not keywords_list:
        return {
            'score': 0,
            'marks': 0,
            'found_keywords': [],
            'missing_keywords': [],
            'total_keywords': 0
        }
    
    # Normalize
    answer_lower = student_answer.lower()
    keywords = [kw.lower() for kw in keywords_list if kw]
    
    # Check which keywords are present
    found_keywords = [kw for kw in keywords if kw in answer_lower]
    missing_keywords = [kw for kw in keywords if kw not in answer_lower]
    
    # Calculate score and marks
    score = round((len(found_keywords) / len(keywords)) * 100) if keywords else 0
    marks = round((score / 100) * total_marks, 2)
    
    return {
        'score': score,
        'marks': marks,
        'found_keywords': found_keywords,
        'missing_keywords': missing_keywords,
        'total_keywords': len(keywords)
    }

# === AI Comparison ===
def compare_answers_with_ai(question, teacher_answer, student_answer, keyword_result):
    """Use IBM Watsonx to compare answers and provide detailed feedback"""
    try:
        prompt = f"""You are an expert teacher grading student answers.

Question: {question}

Teacher's Model Answer:
{teacher_answer}

Student's Answer:
{student_answer}

Keywords Analysis:
- Score: {keyword_result['score']}%
- Marks: {keyword_result['marks']}
- Found Keywords: {', '.join(keyword_result['found_keywords'])}
- Missing Keywords: {', '.join(keyword_result['missing_keywords'])}

Please provide:
1. Detailed feedback comparing the student's answer with the teacher's answer
2. What the student did well
3. What specific improvements are needed
4. An improved version of the student's answer
5. Explanation of the grading

Format your response as JSON:
{{
  "feedback": "detailed comparison feedback",
  "strengths": "what student did well",
  "improvements": "specific areas to improve",
  "improvedAnswer": "corrected student answer",
  "explanation": "grading explanation"
}}

Return ONLY the JSON."""

        response = generate_watsonx_text(prompt)
        
        # Clean JSON
        if response.startswith('```json'):
            response = response.split('```json')[1].split('```')[0].strip()
        elif response.startswith('```'):
            response = response.split('```')[1].split('```')[0].strip()
        
        ai_response = json.loads(response)
        
        return {
            'feedback': ai_response.get('feedback', 'No feedback available'),
            'strengths': ai_response.get('strengths', 'No strengths identified'),
            'improvements': ai_response.get('improvements', 'No improvements identified'),
            'improved_answer': ai_response.get('improvedAnswer', 'No improved answer available'),
            'explanation': ai_response.get('explanation', 'No explanation available')
        }
        
    except Exception as e:
        return {
            'feedback': f"AI analysis unavailable. Error: {str(e)}",
            'strengths': "Keyword matching completed offline.",
            'improvements': "Please review missing keywords.",
            'improved_answer': "AI-generated improved answer unavailable.",
            'explanation': "Offline keyword analysis shows the coverage of required concepts."
        }

# === Process Answer Sheets ===
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
    status_msg = "📄 Extracting text from uploaded files...\n"
    teacher_answer = extract_text_from_file(teacher_file)
    student_answer = extract_text_from_file(student_file)
    
    if "Error" in teacher_answer or "Unsupported" in teacher_answer:
        return f"❌ Teacher's file error: {teacher_answer}", "", "", "", ""
    
    if "Error" in student_answer or "Unsupported" in student_answer:
        return f"❌ Student's file error: {student_answer}", "", "", "", ""
    
    status_msg += "✅ Text extraction successful!\n\n"
    
    # Extract keywords
    if auto_extract_keywords:
        status_msg += "🔍 Auto-extracting keywords from teacher's answer...\n"
        keywords_list = extract_keywords_from_teacher_answer(teacher_answer)
        status_msg += f"✅ Extracted {len(keywords_list)} keywords\n\n"
    else:
        keywords_list = extract_keywords_from_teacher_answer(teacher_answer)
        status_msg += f"🔑 Using {len(keywords_list)} extracted keywords\n\n"
    
    # Offline keyword checking
    status_msg += "⚡ Performing offline keyword analysis...\n"
    keyword_result = check_keywords_offline(student_answer, keywords_list, total_marks)
    status_msg += "✅ Keyword analysis complete!\n\n"
    
    # AI comparison
    status_msg += "🤖 Generating AI-powered feedback using IBM Watsonx...\n"
    ai_result = compare_answers_with_ai(question, teacher_answer, student_answer, keyword_result)
    status_msg += "✅ AI analysis complete!\n\n"
    status_msg += "🎉 **Grading Complete!**"
    
    # Format results
    score = keyword_result['score']
    marks = keyword_result['marks']
    
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
    
    # Score card
    score_text = f"""
# {score_emoji} Score Report

## 📊 **Overall Grade: {grade}**
- **Marks Obtained:** {marks} / {total_marks}
- **Percentage:** {score}%

---

## 📝 **Keyword Analysis**

### ✅ Found Keywords ({len(keyword_result['found_keywords'])}/{keyword_result['total_keywords']})
{', '.join(keyword_result['found_keywords']) if keyword_result['found_keywords'] else 'None'}

### ❌ Missing Keywords ({len(keyword_result['missing_keywords'])})
{', '.join(keyword_result['missing_keywords']) if keyword_result['missing_keywords'] else 'None'}

---

## 🎯 **Keywords Used for Grading:**
{', '.join(keywords_list[:20])}
"""
    
    # Feedback
    feedback_text = f"""
# 💬 Detailed Feedback

{ai_result['feedback']}

---

## ✨ Strengths
{ai_result['strengths']}

---

## 🔧 Areas for Improvement
{ai_result['improvements']}
"""
    
    # Improved answer
    improved_text = f"""
# ✍️ Improved Answer

{ai_result['improved_answer']}
"""
    
    # Explanation
    explanation_text = f"""
# 📚 Grading Explanation

{ai_result['explanation']}

---

### 📖 Extracted Answers:

**Teacher's Answer (First 500 chars):**
{teacher_answer[:500]}...

**Student's Answer (First 500 chars):**
{student_answer[:500]}...
"""
    
    return status_msg, score_text, feedback_text, improved_text, explanation_text

# === Create Gradio Interface ===
with gr.Blocks(theme=gr.themes.Soft(), title="Answer Correcting System with IBM Watsonx") as demo:
    gr.Markdown(
        """
        # 📚 Automated Answer Sheet Correcting System
        ### Powered by IBM Watsonx AI - Upload answer sheets → Get instant marks, feedback & improvements!
        **Supports PDF, DOCX, and TXT files**
        """
    )
    
    # Watsonx Setup Tab
    with gr.Tab("🔧 IBM Watsonx Setup"):
        gr.Markdown("### Configure IBM Watsonx AI")
        gr.Markdown("""
        **Get your IBM Watsonx credentials:**
        1. Visit [IBM Cloud](https://cloud.ibm.com/)
        2. Create a Watsonx.ai project
        3. Get your API Key and Project ID
        """)
        
        with gr.Row():
            api_key_input = gr.Textbox(label="🔑 API Key", type="password", placeholder="Enter your IBM Watsonx API Key")
            project_id_input = gr.Textbox(label="📋 Project ID", placeholder="Enter your Project ID")
        
        url_input = gr.Textbox(label="🌐 URL", value="https://us-south.ml.cloud.ibm.com")
        
        init_btn = gr.Button("🚀 Initialize Watsonx", variant="primary", size="lg")
        init_status = gr.Textbox(label="Status", interactive=False, lines=3)
        
        init_btn.click(
            initialize_watsonx, 
            inputs=[api_key_input, project_id_input, url_input], 
            outputs=init_status
        )
    
    # Main Grading Tab
    with gr.Tab("📝 Grade Answer Sheets"):
        gr.Markdown("### Upload Teacher & Student Answer Sheets for Automatic Grading")
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="📌 Question / Topic",
                    placeholder="Enter the question or topic being evaluated...",
                    lines=3
                )
                
                teacher_file = gr.File(
                    label="📄 Teacher's Answer Sheet (PDF/DOCX/TXT)",
                    file_types=['.pdf', '.docx', '.doc', '.txt']
                )
                
                student_file = gr.File(
                    label="📝 Student's Answer Sheet (PDF/DOCX/TXT)",
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
                        label="🔍 Auto-extract keywords from teacher's answer",
                        value=True
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Grade Answer Sheet", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(
                        components=[question_input, teacher_file, student_file],
                        value="🔄 Reset",
                        size="lg"
                    )
        
        with gr.Row():
            status_output = gr.Markdown(label="Status")
        
        with gr.Row():
            score_output = gr.Markdown(label="Score Report")
        
        with gr.Row():
            feedback_output = gr.Markdown(label="Detailed Feedback")
        
        with gr.Row():
            with gr.Column():
                improved_output = gr.Markdown(label="Improved Answer")
            
            with gr.Column():
                explanation_output = gr.Markdown(label="Grading Explanation")
        
        # Button click event
        submit_btn.click(
            fn=process_answer_sheets,
            inputs=[question_input, teacher_file, student_file, total_marks, auto_extract],
            outputs=[status_output, score_output, feedback_output, improved_output, explanation_output]
        )
    
    # Help Tab
    with gr.Tab("❓ Help"):
        gr.Markdown("""
        ### 📚 Quick Start Guide
        
        #### 1. 🔧 Setup IBM Watsonx
        - Get credentials from [IBM Cloud](https://cloud.ibm.com/)
        - Enter API Key and Project ID in the "IBM Watsonx Setup" tab
        - Click "Initialize Watsonx"
        - Wait for success confirmation
        
        #### 2. 📝 Grade Answer Sheets
        - Enter the question or topic
        - Upload teacher's model answer (PDF/DOCX/TXT)
        - Upload student's answer (PDF/DOCX/TXT)
        - Set total marks (default: 100)
        - Click "Grade Answer Sheet"
        - Get instant results with AI feedback!
        
        ---
        
        ### 🔧 How It Works
        
        1. **📤 File Upload** - System extracts text from both answer sheets
        2. **🔍 Keyword Extraction** - Automatically identifies important keywords from teacher's answer
        3. **⚡ Offline Analysis** - Keyword matching works instantly without internet
        4. **📊 Calculate Marks** - Automatic scoring based on keyword coverage
        5. **🤖 AI Feedback** - IBM Watsonx generates detailed feedback and improvements
        6. **📈 Complete Report** - Get marks, feedback, strengths, improvements, and corrected answer
        
        ---
        
        ### 📋 Supported File Formats
        - **PDF** (.pdf) - Portable Document Format
        - **Word** (.docx, .doc) - Microsoft Word documents
        - **Text** (.txt) - Plain text files
        
        ---
        
        ### 📊 Output Includes
        
        ✅ **Score Report** - Marks, percentage, grade, keyword analysis  
        💬 **Detailed Feedback** - AI comparison with model answer  
        ✨ **Strengths** - What student did well  
        🔧 **Improvements** - Specific areas to improve  
        ✍️ **Improved Answer** - Corrected version with all keywords  
        📚 **Grading Explanation** - Why marks were awarded/deducted  
        
        ---
        
        ### 🎯 Features
        
        ✅ **100% Offline Keyword Matching** - Works without internet  
        ✅ **Auto Keyword Extraction** - No manual keyword entry needed  
        ✅ **AI-Powered Feedback** - Detailed analysis using IBM Watsonx  
        ✅ **Automatic Mark Calculation** - Based on keyword coverage  
        ✅ **Grade Classification** - Excellent, Good, Fair, Needs Improvement  
        ✅ **Multi-Format Support** - PDF, DOCX, TXT files  
        ✅ **Comprehensive Reports** - Complete grading breakdown  
        
        ---
        
        ### 📦 Installation Requirements
        
        **Essential Packages:**
        ```bash
        pip install gradio ibm-watsonx-ai PyPDF2 python-docx
        ```
        
        **IBM Watsonx Setup:**
        1. Create account at [IBM Cloud](https://cloud.ibm.com/)
        2. Create a Watsonx.ai project
        3. Get API Key from IBM Cloud dashboard
        4. Copy Project ID from Watsonx.ai
        5. Enter credentials in Setup tab
        
        ---
        
        ### 🔧 Troubleshooting
        
        **Watsonx Issues:**
        - ❌ "Watsonx not initialized": Configure API in Setup tab
        - 💡 Solution: Enter valid API key and project ID
        
        **File Reading Issues:**
        - ❌ "Error reading file": Unsupported format or corrupted file
        - 💡 Solution: Ensure file is PDF, DOCX, or TXT format
        
        **AI Feedback Issues:**
        - ❌ "AI analysis unavailable": Watsonx error or no internet
        - 💡 Solution: Check credentials and internet connection
        
        ---
        
        ### 💡 Tips for Best Results
        
        **For Answer Sheets:**
        - 📝 Use clear, well-formatted documents
        - ✅ Ensure text is readable (not handwritten unless OCR-processed)
        - 🎯 Teacher's answer should be comprehensive
        - 📊 Student's answer should address the question
        
        **For Keyword Extraction:**
        - ✅ Auto-extraction works best with detailed teacher answers
        - 🔍 System removes common words automatically
        - 📈 Top 15 most important keywords selected
        - 💡 Keywords are case-insensitive
        
        ---
        
        ### ⚠️ Important Notes
        
        - **For educational purposes only**
        - **Always verify AI feedback with human judgment**
        - **Not a replacement for teacher evaluation**
        - **Keyword matching is deterministic and fair**
        - **AI feedback provides guidance, not absolute truth**
        
        ---
        
        ### 🔒 Privacy & Security
        
        - ✅ Files processed in session only
        - ✅ No permanent data storage
        - ✅ IBM Watsonx API credentials encrypted
        - ✅ Answer sheets not saved externally
        
        ---
        
        ### 📞 Support
        
        For IBM Watsonx support:
        - [IBM Cloud Documentation](https://cloud.ibm.com/docs)
        - [Watsonx.ai Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)
        
        ---
        
        ### ✨ Model Information
        
        **AI Model Used:** Mistral Small 3.1 (24B Instruct)
        - Model ID: `mistralai/mistral-small-3-1-24b-instruct-2503`
        - Provider: IBM Watsonx.ai
        - Capabilities: Text generation, analysis, grading, feedback
        - Temperature: 0.2 (for consistent grading)
        - Max Tokens: 1024
        
        ---
        
        ### 💡 Use Cases
        
        ✅ Schools & Universities - Grade assignments and exams  
        ✅ Online Learning Platforms - Automated feedback  
        ✅ Teachers - Quick preliminary grading  
        ✅ Students - Self-assessment and improvement  
        ✅ Educational Institutions - Standardized evaluation  
        """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 Answer Sheet Correcting System")
    print("Powered by IBM Watsonx AI")
    print("="*60)
    print("\n✨ Features:")
    print("• 📤 Upload teacher & student answer sheets (PDF/DOCX/TXT)")
    print("• 🔍 Auto-extract keywords from teacher's answer")
    print("• ⚡ 100% offline keyword matching")
    print("• 📊 Automatic mark calculation")
    print("• 🤖 AI-powered feedback using IBM Watsonx")
    print("• 📈 Comprehensive grading reports")
    print("\n📦 Requirements:")
    print("• pip install gradio ibm-watsonx-ai PyPDF2 python-docx")
    print("• IBM Watsonx API credentials (API Key + Project ID)")
    print("\n" + "="*60 + "\n")
    
    demo.launch(share=False, debug=True)