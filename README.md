# Automated-Answer-Sheet-Correcting-System
📝 AI Answer Sheet Grading System

An advanced automatic grading system that compares a teacher’s answer sheet and a student’s answer sheet, extracts keywords, evaluates concept coverage, and generates an AI-powered score, feedback, improved answer, and a full PDF report.
Supports offline keyword-based grading and online AI-enhanced grading using IBM Watsonx.

🚀 Features
✅ 1. Multi-Format File Upload

Supports:

PDF

DOCX / DOC

TXT

Automatically extracts text using:

PyPDF2 for PDFs

python-docx for DOCX

Native text reading for TXT

🔍 2. Smart Keyword Extraction

The system:

Extracts top 25 meaningful keywords from teacher’s answer

Removes stop words

Performs frequency-based ranking

Supports partial matching in student answers

🧠 3. Dual Evaluation System
A. Offline Evaluation (No Internet Required)

Keyword matching

Concept coverage (%)

Auto score calculation

Offline result reliability

B. Online AI Evaluation (IBM Watsonx)

Uses:

mistralai/mistral-small-3-1-24b-instruct-2503


AI generates:

Final score

Strengths & weaknesses

Concept coverage

A fully rewritten improved answer

A complete JSON-based structured evaluation

📄 4. Automatic PDF Report Generation

Uses ReportLab to generate a professional report containing:

Final marks & grade

Keyword analysis

Concepts covered & missed

AI feedback

Strengths & weaknesses

Teacher answer

Student answer

Improved answer

Timestamp & metadata

Generated as:

grading_report_YYYYMMDD_HHMMSS.pdf

🧩 Tech Stack
Component	Purpose
Gradio	UI frontend
IBM Watsonx	AI text evaluation
ReportLab	PDF generation
PyPDF2	PDF text extraction
python-docx	DOCX reading
Regex + NLP	Keyword extraction
⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/ai-grading-system.git
cd ai-grading-system

2. Install Dependencies
pip install gradio PyPDF2 python-docx reportlab ibm-watsonx-ai

🔑 IBM Watsonx Setup

You need:

API Key

Project ID

Deployment URL (default automatically used)

Initialize inside the app:

initialize_watsonx(api_key, project_id)

▶️ Run the Application
python app.py


Gradio will open in the browser:

http://localhost:7860

📘 How It Works
Step 1 — Upload Files

Upload teacher answer file

Upload student answer file

Enter the question

Step 2 — Keyword Extraction

Extracts topics from teacher answer

Matches them inside student answer

Step 3 — AI Evaluation (Optional)

IBM Watsonx produces:

JSON score

Feedback

Improved answer

Step 4 — PDF Report

System exports a beautifully formatted report with:

Grades

Keyword coverage

Strengths & weaknesses

Corrected answer

Teacher vs Student comparison

🧪 JSON Output Example
{
  "finalScore": 82,
  "finalMarks": 41,
  "accuracyRating": "Good",
  "feedback": "The student demonstrated clear understanding...",
  "strengths": "Covered core concepts effectively",
  "weaknesses": "Missed advanced points",
  "improvements": "Add explanation for ...",
  "improvedAnswer": "A refined, corrected version of the student's answer...",
  "gradingJustification": "Score assigned based on conceptual coverage...",
  "keyConceptsCovered": ["definition", "purpose", "workflow"],
  "keyConceptsMissed": ["limitations", "use cases"]
}

🛠️ Folder Structure
├── app.py
├── README.md
├── assets/
│   └── sample_reports/
└── requirements.txt

📌 Future Enhancements

Add OCR support (images → text)

Add multilingual grading

Add real-time teacher keyword editor

Integration with Google Classroom / Moodle

Export results to Excel

🤝 Contributing

Pull requests and suggestions are welcome!

📜 License

Open-source under MIT License.
