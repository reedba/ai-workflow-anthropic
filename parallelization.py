import openai
import os
import gradio as gr
from dotenv import load_dotenv
import asyncio
import concurrent.futures
import time
from threading import Thread
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class ParallelAIProcessor:
    """Handles parallel processing of AI tasks"""
    
    def __init__(self):
        self.results = {}
        self.processing_times = {}
    
    def single_ai_task(self, task_id, prompt, system_message, max_tokens=200):
        """Single AI task that can be run in parallel"""
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            return {
                "task_id": task_id,
                "result": result,
                "processing_time": round(processing_time, 2),
                "status": "success"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "task_id": task_id,
                "result": f"Error: {str(e)}",
                "processing_time": round(processing_time, 2),
                "status": "error"
            }
    
    def process_sequential(self, tasks):
        """Process tasks one by one (sequential)"""
        start_time = time.time()
        results = []
        
        for task in tasks:
            result = self.single_ai_task(
                task["id"], 
                task["prompt"], 
                task["system_message"], 
                task.get("max_tokens", 200)
            )
            results.append(result)
        
        total_time = time.time() - start_time
        return results, round(total_time, 2)
    
    def process_parallel_threads(self, tasks):
        """Process tasks in parallel using ThreadPoolExecutor"""
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self.single_ai_task, 
                    task["id"], 
                    task["prompt"], 
                    task["system_message"], 
                    task.get("max_tokens", 200)
                ): task for task in tasks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
        
        # Sort results by task_id to maintain order
        results.sort(key=lambda x: x["task_id"])
        total_time = time.time() - start_time
        return results, round(total_time, 2)
    
    def process_parallel_async(self, tasks):
        """Process tasks in parallel using asyncio (simulated)"""
        # Note: OpenAI client is not async, so we simulate with threads
        return self.process_parallel_threads(tasks)

def create_sample_tasks(input_text):
    """Create sample AI tasks for parallel processing demonstration"""
    tasks = [
        {
            "id": "summary",
            "prompt": f"Summarize this text in 2-3 sentences:\n{input_text}",
            "system_message": "You are a professional summarization expert.",
            "max_tokens": 150
        },
        {
            "id": "sentiment",
            "prompt": f"Analyze the sentiment of this text and provide a brief explanation:\n{input_text}",
            "system_message": "You are a sentiment analysis expert.",
            "max_tokens": 100
        },
        {
            "id": "keywords",
            "prompt": f"Extract 5-7 key topics or keywords from this text:\n{input_text}",
            "system_message": "You are an expert at identifying key topics and themes.",
            "max_tokens": 100
        },
        {
            "id": "questions",
            "prompt": f"Generate 3 insightful questions based on this text:\n{input_text}",
            "system_message": "You are an expert at creating thought-provoking questions.",
            "max_tokens": 150
        },
        {
            "id": "translation",
            "prompt": f"Translate this text to Spanish:\n{input_text}",
            "system_message": "You are a professional translator.",
            "max_tokens": 200
        },
        {
            "id": "category",
            "prompt": f"Categorize this text into one of these categories: Business, Technology, Health, Education, Entertainment, News, Other. Provide category and brief reasoning:\n{input_text}",
            "system_message": "You are a content categorization expert.",
            "max_tokens": 100
        }
    ]
    return tasks

def format_comparison_results(sequential_results, sequential_time, parallel_results, parallel_time):
    """Format the comparison results for display"""
    
    # Calculate time savings
    time_saved = sequential_time - parallel_time
    speedup = round(sequential_time / parallel_time, 2) if parallel_time > 0 else 0
    efficiency = round((time_saved / sequential_time) * 100, 2) if sequential_time > 0 else 0
    
    output = f"""
# ⚡ Parallelization Performance Comparison

## 📊 Processing Time Analysis
- **🐌 Sequential Processing**: {sequential_time}s
- **⚡ Parallel Processing**: {parallel_time}s
- **⏱️ Time Saved**: {time_saved}s
- **🚀 Speedup**: {speedup}x faster
- **📈 Efficiency Gain**: {efficiency}%

---

## 📋 Task Results Comparison

### 🐌 Sequential Results:
"""
    
    # Add sequential results
    for result in sequential_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        output += f"""
**{status_icon} {result['task_id'].title()}** ({result['processing_time']}s)
{result['result'][:150]}{'...' if len(result['result']) > 150 else ''}

"""
    
    output += """
### ⚡ Parallel Results:
"""
    
    # Add parallel results
    for result in parallel_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        output += f"""
**{status_icon} {result['task_id'].title()}** ({result['processing_time']}s)
{result['result'][:150]}{'...' if len(result['result']) > 150 else ''}

"""
    
    output += f"""
---

## 💡 Parallelization Benefits Demonstrated:

1. **⚡ Speed Improvement**: {speedup}x faster processing
2. **📈 Resource Efficiency**: Better utilization of API rate limits
3. **🔄 Concurrent Execution**: Multiple AI tasks run simultaneously
4. **⏰ Reduced Latency**: Faster overall response times
5. **📊 Scalability**: Can handle more tasks in same timeframe

**💰 Cost Efficiency**: Same API usage but {efficiency}% faster results!

*Note: Actual speedup depends on API rate limits and network conditions*
"""
    
    return output

def run_parallelization_demo(input_text, processing_mode):
    """Main function to demonstrate parallelization"""
    if not input_text.strip():
        return "Please enter some text to analyze."
    
    try:
        processor = ParallelAIProcessor()
        tasks = create_sample_tasks(input_text)
        
        if processing_mode == "Sequential Only":
            # Run only sequential
            results, total_time = processor.process_sequential(tasks)
            
            output = f"""
# 🐌 Sequential Processing Results

**Total Processing Time**: {total_time} seconds

## 📋 Task Results:
"""
            for result in results:
                status_icon = "✅" if result["status"] == "success" else "❌"
                output += f"""
**{status_icon} {result['task_id'].title()}** ({result['processing_time']}s)
{result['result'][:200]}{'...' if len(result['result']) > 200 else ''}

"""
            return output
            
        elif processing_mode == "Parallel Only":
            # Run only parallel
            results, total_time = processor.process_parallel_threads(tasks)
            
            output = f"""
# ⚡ Parallel Processing Results

**Total Processing Time**: {total_time} seconds

## 📋 Task Results:
"""
            for result in results:
                status_icon = "✅" if result["status"] == "success" else "❌"
                output += f"""
**{status_icon} {result['task_id'].title()}** ({result['processing_time']}s)
{result['result'][:200]}{'...' if len(result['result']) > 200 else ''}

"""
            return output
            
        else:  # "Compare Both"
            # Run both and compare
            print("Running sequential processing...")
            sequential_results, sequential_time = processor.process_sequential(tasks)
            
            print("Running parallel processing...")
            parallel_results, parallel_time = processor.process_parallel_threads(tasks)
            
            return format_comparison_results(
                sequential_results, sequential_time,
                parallel_results, parallel_time
            )
    
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease make sure your OpenAI API key is set correctly."

# Sample texts for demonstration
SAMPLE_TEXTS = {
    "AI Technology Article": """Artificial Intelligence is transforming industries at an unprecedented pace. Machine learning algorithms are now capable of processing vast amounts of data to identify patterns and make predictions with remarkable accuracy. From healthcare diagnostics to financial trading, AI systems are becoming integral to modern business operations. However, this rapid advancement also raises important questions about ethics, privacy, and the future of human employment in an increasingly automated world.""",
    
    "Business Report": """Our Q4 financial results show strong growth across all major product lines. Revenue increased by 23% compared to the previous quarter, driven primarily by our new cloud services offering. Customer acquisition costs have decreased by 15% while customer lifetime value has increased by 18%. The expansion into European markets has exceeded expectations, contributing 12% to total revenue. Moving forward, we plan to invest heavily in R&D and continue our international expansion strategy.""",
    
    "Healthcare Study": """A recent clinical study involving 1,200 patients demonstrated the effectiveness of a new treatment protocol for Type 2 diabetes. Patients following the new regimen showed a 34% improvement in blood glucose control compared to the control group. Side effects were minimal and reported in less than 8% of participants. The study suggests that this approach could significantly improve quality of life for millions of diabetes patients worldwide while reducing long-term healthcare costs.""",
    
    "Technology News": """The latest smartphone release features groundbreaking battery technology that promises 48-hour usage on a single charge. The device incorporates advanced AI chips for enhanced photography and voice recognition capabilities. Early reviews praise the improved display quality and 5G connectivity speeds. However, the premium pricing may limit market adoption initially. Industry analysts predict this technology will become standard across all manufacturers within two years."""
}

# Create Gradio interface
with gr.Blocks(title="AI Parallelization Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ⚡ AI Parallelization Workflow Demonstration
    
    Experience the power of parallel processing with AI tasks. This demo shows how running multiple AI operations concurrently can dramatically improve performance and efficiency.
    """)
    
    gr.Markdown("""
    ## 🔄 What This Demo Does:
    
    **6 Simultaneous AI Tasks:**
    1. **📝 Text Summarization** - Creates concise summary
    2. **😊 Sentiment Analysis** - Analyzes emotional tone  
    3. **🔑 Keyword Extraction** - Identifies key topics
    4. **❓ Question Generation** - Creates insightful questions
    5. **🌍 Translation** - Translates to Spanish
    6. **📂 Categorization** - Classifies content type
    
    **Processing Modes:**
    - **🐌 Sequential**: Tasks run one after another
    - **⚡ Parallel**: Tasks run simultaneously  
    - **📊 Compare**: Shows performance difference
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="📝 Input Text",
                placeholder="Enter text to analyze with parallel AI processing...",
                lines=8,
                value=SAMPLE_TEXTS["AI Technology Article"]
            )
            
            # Sample text selection
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_TEXTS.keys()),
                label="📚 Try Sample Texts",
                value="AI Technology Article"
            )
            
            processing_mode = gr.Radio(
                choices=["Sequential Only", "Parallel Only", "Compare Both"],
                label="⚙️ Processing Mode",
                value="Compare Both"
            )
            
            process_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            output = gr.Markdown(label="📊 Processing Results")
    
    # Handle sample text selection
    def load_sample_text(sample_key):
        if sample_key and sample_key in SAMPLE_TEXTS:
            return SAMPLE_TEXTS[sample_key]
        return ""
    
    sample_dropdown.change(
        fn=load_sample_text,
        inputs=sample_dropdown,
        outputs=input_text
    )
    
    process_btn.click(
        fn=run_parallelization_demo,
        inputs=[input_text, processing_mode],
        outputs=output
    )
    
    gr.Markdown("""
    ### 🎯 Parallelization Benefits:
    
    **⚡ Performance Gains:**
    - **Speed**: 3-6x faster processing times
    - **Efficiency**: Better resource utilization
    - **Scalability**: Handle more concurrent requests
    - **Throughput**: Process larger workloads
    
    **💼 Business Applications:**
    - **📊 Batch Document Processing**: Analyze multiple documents simultaneously
    - **🎖️ VA Claims Processing**: Parallel analysis of medical records and evidence
    - **📞 Customer Service**: Concurrent sentiment analysis and routing
    - **📈 Market Research**: Simultaneous analysis of multiple data sources
    - **🔍 Content Moderation**: Parallel safety and quality checks
    
    **🔧 Technical Implementation:**
    - **ThreadPoolExecutor**: Concurrent API calls
    - **Async Processing**: Non-blocking operations
    - **Resource Management**: Optimal API rate limit usage
    - **Error Handling**: Graceful failure management
    
    This demonstrates how parallelization can transform AI workflow performance in real-world applications!
    """)

if __name__ == "__main__":
    print("=== AI Parallelization Demo ===\n")
    print("⚡ Demonstrating the power of parallel AI processing...")
    print("Compare sequential vs parallel execution times\n")
    
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7862)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)