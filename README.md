# � AI Document Processing & Analysis

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

## 🛠️ Setup

1. **Clone and navigate to the repository**
   ```bash
   cd ai-workflow-anthropic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your-actual-api-key-here
   ```
   
   Or set as environment variable:
   ```bash
   export OPENAI_API_KEY="your-actual-api-key-here"
   ```

## 🎯 Usage

### Web Interface (Recommended)
```bash
python prompt_chaining.py
```
This will:
1. Launch an interactive web interface for document processing
2. Allow you to upload documents (PDF, DOCX, TXT) or paste text directly
3. Process documents through the 4-step AI analysis pipeline
4. Open automatically in your browser at `http://localhost:7860`

### Command Line Only
The script will automatically demonstrate the 3-step process with sample text about AI.

## 🔄 How Document Processing Works

The system uses advanced prompt chaining to analyze documents through a sophisticated 4-step workflow:

### 📄 Document Processing Chain (4 Steps)
1. **📄 Document Summary**: Extracts text from uploaded files and creates a comprehensive summary
2. **🎯 Main Points**: Identifies 5-7 key topics, themes, and important information
3. **🏷️ Smart Categorization**: Assigns relevant categories, tags, and industry classification
4. **💡 Actionable Insights**: Generates strategic recommendations, key takeaways, and next steps

Each step uses the output from the previous step as input, creating a sophisticated analysis pipeline that transforms raw documents into actionable business intelligence.

## 📁 Project Structure

```
ai-workflow-anthropic/
├── prompt_chaining.py      # Main application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore file
└── README.md             # This file
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
