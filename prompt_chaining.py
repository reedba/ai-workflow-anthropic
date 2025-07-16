import openai
import os
import gradio as gr
from dotenv import load_dotenv
import PyPDF2
import docx
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key (replace with your actual key or set as environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_file(file_path):
    """Extract text from various file formats"""
    if file_path is None:
        return ""
    
    file_extension = Path(file_path).suffix.lower()
    text = ""
    
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        
        elif file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        elif file_extension in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        else:
            return f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
    return text.strip()

def document_processing_chain(file_path, input_text=""):
    """
    Enhanced 4-step prompt chaining workflow for document processing:
    1. Extract and summarize the document/text
    2. Identify main points 
    3. Categorize with appropriate tags
    4. Generate actionable insights
    """
    results = {}
    
    # Extract text from file if provided, otherwise use input text
    if file_path:
        content = extract_text_from_file(file_path)
        if content.startswith("Error") or content.startswith("Unsupported"):
            return {"error": content}
    else:
        content = input_text
    
    if not content.strip():
        return {"error": "No content provided"}
    
    results["original_content"] = content[:500] + "..." if len(content) > 500 else content
    
    # Step 1: Summarize the document
    summarize_prompt = f"""Analyze and summarize the following document in 3-4 clear, concise sentences. Focus on the key message and purpose:

Document Content:
{content}"""
    
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert document analyst who creates clear, professional summaries."},
            {"role": "user", "content": summarize_prompt}
        ],
        max_tokens=200
    )
    summary = summary_response.choices[0].message.content.strip()
    results["summary"] = summary
    
    # Step 2: Identify main points
    main_points_prompt = f"""Based on this document summary, identify and list the 5-7 most important main points or key topics. Present them as a numbered list with brief explanations.

Summary: {summary}

Original content for reference:
{content[:1000]}..."""
    
    main_points_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at identifying key points and themes in documents. Be specific and actionable."},
            {"role": "user", "content": main_points_prompt}
        ],
        max_tokens=300
    )
    main_points = main_points_response.choices[0].message.content.strip()
    results["main_points"] = main_points
    
    # Step 3: Categorize with tags
    categorization_prompt = f"""Based on the summary and main points below, provide:

1. **Primary Category**: Choose the most appropriate category (e.g., Business Report, Technical Documentation, Research Paper, Policy Document, Educational Material, Marketing Content, Legal Document, etc.)

2. **Tags**: Generate 5-8 relevant tags that describe the content, topics, and themes

3. **Industry/Domain**: Identify the primary industry or domain this document belongs to

Summary: {summary}

Main Points: {main_points}"""
    
    categorization_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert document classifier and tagging specialist. Provide accurate, specific categories and tags."},
            {"role": "user", "content": categorization_prompt}
        ],
        max_tokens=250
    )
    categorization = categorization_response.choices[0].message.content.strip()
    results["categorization"] = categorization
    
    # Step 4: Generate actionable insights
    insights_prompt = f"""Based on the document analysis, provide actionable insights and recommendations:

Summary: {summary}
Main Points: {main_points}
Categories/Tags: {categorization}

Provide:
1. Key takeaways for stakeholders
2. Potential next steps or actions
3. Areas that might need further attention or investigation"""
    
    insights_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a strategic analyst who provides actionable business insights and recommendations."},
            {"role": "user", "content": insights_prompt}
        ],
        max_tokens=300
    )
    insights = insights_response.choices[0].message.content.strip()
    results["insights"] = insights
    
    return results



def format_document_output(results):
    """Format the document processing results for display"""
    if "error" in results:
        return f"❌ **Error**: {results['error']}"
    
    output = f"""
## 📄 Document Summary
{results['summary']}

## 🎯 Main Points
{results['main_points']}

## 🏷️ Categorization & Tags
{results['categorization']}

## 💡 Actionable Insights
{results['insights']}

---
*Document processed through 4-step AI analysis: Summary → Main Points → Categorization → Insights*
"""
    return output

def gradio_document_interface(file, input_text):
    """Gradio interface function for document processing"""
    if not file and not input_text.strip():
        return "Please upload a document or enter text to analyze."
    
    try:
        results = document_processing_chain(file.name if file else None, input_text)
        return format_document_output(results)
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease make sure your OpenAI API key is set correctly in your .env file."

# Create Gradio interface
with gr.Blocks(title="AI Document Processing Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # � AI Document Processing & Analysis
    
    Upload documents (PDF, DOCX, TXT) or paste text directly for comprehensive AI analysis using advanced prompt chaining.
    """)
    
    gr.Markdown("""
    ## 🔄 4-Step AI Analysis Process:
    1. **📄 Document Summary**: Extracts and summarizes content
    2. **🎯 Main Points**: Identifies key topics and themes  
    3. **🏷️ Categorization**: Assigns categories and relevant tags
    4. **💡 Insights**: Generates actionable recommendations
    
    **Supported formats:** PDF, DOCX, TXT
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )
            gr.Markdown("**OR**")
            doc_text_input = gr.Textbox(
                label="📝 Enter Text Directly",
                placeholder="Paste document content here if not uploading a file...",
                lines=8
            )
            doc_submit_btn = gr.Button("� Analyze Document", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            doc_output = gr.Markdown(label="📊 Analysis Results")
    
    doc_submit_btn.click(
        fn=gradio_document_interface,
        inputs=[file_input, doc_text_input],
        outputs=doc_output
    )
    
    gr.Markdown("""
    ### 🎯 Business Applications:
    - **📊 Business Reports**: Automated analysis and insight extraction
    - **📋 Policy Documents**: Compliance review and action items
    - **📚 Research Papers**: Key findings and categorization  
    - **📄 Contracts**: Risk assessment and important terms
    - **📈 Market Research**: Data analysis and strategic recommendations
    - **🎖️ Veterans Affairs Appeals**: Medical records analysis, evidence summarization, and appeals strategy development
    
    This AI-powered document processing demonstrates how prompt chaining creates sophisticated analysis workflows!
    """)

if __name__ == "__main__":
    # Command line demo
    print("=== AI Document Processing Demo ===\n")
    
    print("🚀 Starting Document Processing Interface...")
    print("Upload documents or paste text for comprehensive AI analysis\n")
    
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)
