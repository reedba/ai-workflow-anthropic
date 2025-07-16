# 🤖 AI Workflow Demonstrations Suite

A comprehensive collection of AI-powered workflow demonstrations showcasing document processing, intelligent routing, and parallelization techniques using OpenAI's API and Gradio interfaces.

## 🚀 Available Workflows

### 📄 **1. Document Processing & Analysis**
**File**: `prompt_chaining.py` | **Port**: `7860`
- **4-Step Analysis**: Summary → Main Points → Categorization → Insights
- **Multi-Format Support**: PDF, DOCX, TXT file processing
- **Smart Categorization**: Automatic tagging and classification
- **Business Intelligence**: Actionable recommendations and next steps

### 🎯 **2. AI-Powered Request Routing**
**File**: `routing.py` | **Port**: `7861`
- **Intelligent Classification**: 8 specialized routing categories
- **Priority Assessment**: Automatic urgency detection with SLA management
- **Veterans Affairs Support**: Specialized VA claims and appeals routing
- **Action Planning**: Detailed routing summaries with next steps

### ⚡ **3. AI Parallelization Demo**
**File**: `parallelization.py` | **Port**: `7862`
- **Performance Comparison**: Sequential vs Parallel processing
- **6 Concurrent AI Tasks**: Summary, sentiment, keywords, questions, translation, categorization
- **Real-time Metrics**: Processing times, speedup ratios, efficiency gains
- **Business Applications**: Batch processing and workflow accelerationent Processing & Analysis

A comprehensive AI-powered document processing system using OpenAI's API and Gradio for intelligent document analysis through advanced prompt chaining.

## 🚀 Features

- **� 4-Step Document Analysis**: Summary → Main Points → Categorization → Insights
- **📁 Multi-Format Support**: PDF, DOCX, and TXT file processing
- **🌐 Interactive Web Interface**: Built with Gradio
- **🏷️ Smart Categorization**: Automatic tagging and classification
- **💡 Actionable Insights**: AI-generated recommendations and next steps
- **🔧 Modern OpenAI API**: Uses the latest OpenAI Python client (v1.0+)

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## 🛠️ Quick Start

### **Launch All Workflows**
```bash
# Terminal 1: Document Processing
python prompt_chaining.py
# Access at: http://localhost:7860

# Terminal 2: Request Routing  
python routing.py
# Access at: http://localhost:7861

# Terminal 3: Parallelization Demo
python parallelization.py
# Access at: http://localhost:7862
```

### **Single Workflow Setup**
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key**
   ```env
   OPENAI_API_KEY=your-actual-api-key-here
   ```

3. **Run desired workflow**
   ```bash
   python [workflow_file].py
   ```

## 🎯 Usage

### **Web Interface (Recommended)**
Each workflow provides a user-friendly Gradio interface:

- **📄 Document Processing**: Upload files or paste text for comprehensive analysis
- **🎯 Request Routing**: Enter customer requests for intelligent routing decisions  
- **⚡ Parallelization**: Compare sequential vs parallel AI processing performance

### **Features Available:**
- **Sample Data**: Pre-loaded examples for testing
- **Real-time Processing**: Live AI analysis and results
- **Performance Metrics**: Processing times and efficiency measurements
- **Export Options**: Copy results for further use

## 🔄 Workflow Descriptions

### 📄 **Document Processing Workflow**
Advanced 4-step document analysis pipeline:
1. **📄 Document Summary**: Extracts text from uploaded files and creates comprehensive summary
2. **🎯 Main Points**: Identifies 5-7 key topics, themes, and important information  
3. **🏷️ Smart Categorization**: Assigns relevant categories, tags, and industry classification
4. **💡 Actionable Insights**: Generates strategic recommendations, key takeaways, and next steps

### 🎯 **Request Routing Workflow**
Intelligent 3-step routing system:
1. **🎯 Classification**: Analyzes request content and assigns appropriate category
2. **⚡ Priority Assessment**: Determines urgency level and special handling needs
3. **📋 Action Planning**: Provides routing summary and recommended next steps

**Routing Categories**: Technical Support, Customer Service, Billing, Sales, Legal, HR, Veterans Affairs, Emergency

### ⚡ **Parallelization Workflow**
Performance optimization demonstration:
1. **📊 Task Creation**: Generates 6 different AI analysis tasks
2. **🔄 Processing Modes**: Sequential vs Parallel execution comparison
3. **📈 Performance Analysis**: Real-time metrics showing speedup and efficiency gains

**AI Tasks**: Summarization, Sentiment Analysis, Keyword Extraction, Question Generation, Translation, Categorization

## 📁 Project Structure

```
ai-workflow-anthropic/
├── prompt_chaining.py          # Document processing workflow
├── routing.py                  # Intelligent request routing  
├── parallelization.py          # AI parallelization demo
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore file
├── README.md                 # Main project documentation
├── ROUTING_README.md         # Routing workflow details
├── PARALLELIZATION_README.md # Parallelization documentation
└── sample_business_report.txt # Sample document for testing
```

## 🤝 Business Applications

This AI-powered document processing can be applied to:

**Business Intelligence Applications:**
- **📊 Business Reports**: Automated analysis and strategic insight extraction
- **📋 Policy Documents**: Compliance review and action item identification
- **📄 Contract Analysis**: Risk assessment and key terms extraction
- **📈 Market Research**: Data analysis and strategic recommendations

**Operational Applications:**
- **📚 Research Papers**: Key findings summarization and categorization
- **📝 Technical Documentation**: Main points extraction and user guidance
- **💼 Proposal Analysis**: Opportunity assessment and decision support
- **🔍 Due Diligence**: Document review and risk identification

**Public Service Applications:**
- **🎖️ Veterans Affairs Appeals**: Medical records analysis, evidence summarization, and appeals strategy development
- **⚖️ Legal Document Review**: Case analysis and precedent identification
- **🏥 Healthcare Documentation**: Patient record analysis and care coordination

## 🛡️ Security Notes

- Keep your `.env` file private (it's already in `.gitignore`)
- Never commit API keys to version control
- Consider using environment variables in production

## 📚 Dependencies

- `openai>=1.0.0` - OpenAI API client
- `gradio>=4.0.0` - Web interface framework  
- `python-dotenv>=1.0.0` - Environment variable loading
- `PyPDF2>=3.0.0` - PDF document processing
- `python-docx>=0.8.11` - Microsoft Word document processing
