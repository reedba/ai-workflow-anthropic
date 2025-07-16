import openai
import os
import gradio as gr
from dotenv import load_dotenv
import asyncio
import concurrent.futures
import time
import json
import queue
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class WorkerType(Enum):
    DOCUMENT_ANALYZER = "document_analyzer"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    TRANSLATOR = "translator"
    SUMMARIZER = "summarizer"
    QUESTION_GENERATOR = "question_generator"
    CATEGORY_CLASSIFIER = "category_classifier"

@dataclass
class Task:
    id: str
    worker_type: WorkerType
    input_data: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AIWorker:
    """Individual AI worker that processes specific types of tasks"""
    
    def __init__(self, worker_id: str, worker_type: WorkerType):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.is_busy = False
        self.tasks_completed = 0
        self.total_processing_time = 0.0
        
    def can_handle(self, task: Task) -> bool:
        """Check if this worker can handle the given task"""
        return task.worker_type == self.worker_type
    
    def process_task(self, task: Task) -> Task:
        """Process a single task"""
        if self.is_busy:
            raise Exception(f"Worker {self.worker_id} is already busy")
        
        self.is_busy = True
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Route to appropriate AI processing method
            if self.worker_type == WorkerType.DOCUMENT_ANALYZER:
                result = self._analyze_document(task.input_data)
            elif self.worker_type == WorkerType.SENTIMENT_ANALYZER:
                result = self._analyze_sentiment(task.input_data)
            elif self.worker_type == WorkerType.TRANSLATOR:
                result = self._translate_text(task.input_data)
            elif self.worker_type == WorkerType.SUMMARIZER:
                result = self._summarize_text(task.input_data)
            elif self.worker_type == WorkerType.QUESTION_GENERATOR:
                result = self._generate_questions(task.input_data)
            elif self.worker_type == WorkerType.CATEGORY_CLASSIFIER:
                result = self._classify_category(task.input_data)
            else:
                raise Exception(f"Unknown worker type: {self.worker_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        
        finally:
            task.completed_at = datetime.now()
            task.processing_time = (task.completed_at - task.started_at).total_seconds()
            self.is_busy = False
            self.tasks_completed += 1
            self.total_processing_time += task.processing_time
        
        return task
    
    def _analyze_document(self, text: str) -> str:
        """Document analysis worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Analyze the document and extract key insights."},
                {"role": "user", "content": f"Analyze this document and provide key insights:\n{text}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def _analyze_sentiment(self, text: str) -> str:
        """Sentiment analysis worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Determine the emotional tone and sentiment."},
                {"role": "user", "content": f"Analyze the sentiment of this text:\n{text}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    def _translate_text(self, text: str) -> str:
        """Translation worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate to Spanish."},
                {"role": "user", "content": f"Translate this text to Spanish:\n{text}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def _summarize_text(self, text: str) -> str:
        """Summarization worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a summarization expert. Create concise, informative summaries."},
                {"role": "user", "content": f"Summarize this text in 2-3 sentences:\n{text}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    
    def _generate_questions(self, text: str) -> str:
        """Question generation worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating insightful questions based on content."},
                {"role": "user", "content": f"Generate 3 thoughtful questions based on this text:\n{text}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    
    def _classify_category(self, text: str) -> str:
        """Category classification worker"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content classification expert. Categorize content accurately."},
                {"role": "user", "content": f"Classify this text into appropriate categories:\n{text}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

class AIOrchestrator:
    """Central orchestrator that manages workers and coordinates task execution"""
    
    def __init__(self):
        self.workers: List[AIWorker] = []
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.is_running = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize worker pool with different AI capabilities"""
        worker_configs = [
            (WorkerType.DOCUMENT_ANALYZER, 2),   # 2 document analyzers
            (WorkerType.SENTIMENT_ANALYZER, 1),  # 1 sentiment analyzer
            (WorkerType.TRANSLATOR, 1),          # 1 translator
            (WorkerType.SUMMARIZER, 1),          # 1 summarizer
            (WorkerType.QUESTION_GENERATOR, 1),  # 1 question generator
            (WorkerType.CATEGORY_CLASSIFIER, 1), # 1 classifier
        ]
        
        for worker_type, count in worker_configs:
            for i in range(count):
                worker_id = f"{worker_type.value}_{i+1}"
                worker = AIWorker(worker_id, worker_type)
                self.workers.append(worker)
    
    def submit_task(self, worker_type: WorkerType, input_data: str, priority: int = 1) -> str:
        """Submit a new task to the orchestrator"""
        task_id = f"task_{int(time.time()*1000)}_{len(self.completed_tasks)}"
        task = Task(
            id=task_id,
            worker_type=worker_type,
            input_data=input_data,
            priority=priority
        )
        
        # Higher priority number = higher priority (negative for min-heap)
        self.task_queue.put((-priority, time.time(), task))
        return task_id
    
    def submit_workflow(self, input_data: str, workflow_tasks: List[WorkerType]) -> List[str]:
        """Submit a complete workflow with multiple tasks"""
        task_ids = []
        for i, worker_type in enumerate(workflow_tasks):
            # Higher priority for earlier tasks in workflow
            priority = len(workflow_tasks) - i
            task_id = self.submit_task(worker_type, input_data, priority)
            task_ids.append(task_id)
        return task_ids
    
    def get_available_worker(self, worker_type: WorkerType) -> Optional[AIWorker]:
        """Find an available worker for the given task type"""
        available_workers = [
            worker for worker in self.workers 
            if worker.worker_type == worker_type and not worker.is_busy
        ]
        return available_workers[0] if available_workers else None
    
    def process_tasks(self) -> Dict[str, Any]:
        """Process all pending tasks using available workers"""
        self.is_running = True
        start_time = time.time()
        
        # Collect tasks to process
        tasks_to_process = []
        while not self.task_queue.empty():
            _, _, task = self.task_queue.get()
            tasks_to_process.append(task)
        
        if not tasks_to_process:
            return {"message": "No tasks to process", "results": []}
        
        # Submit tasks to thread pool
        future_to_task = {}
        for task in tasks_to_process:
            worker = self.get_available_worker(task.worker_type)
            if worker:
                future = self.executor.submit(worker.process_task, task)
                future_to_task[future] = task
            else:
                # If no worker available, put task back in queue
                self.task_queue.put((-task.priority, time.time(), task))
        
        # Collect results
        completed_tasks = []
        for future in concurrent.futures.as_completed(future_to_task):
            task = future.result()
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks.append(task)
                completed_tasks.append(task)
            else:
                self.failed_tasks.append(task)
        
        total_time = time.time() - start_time
        self.is_running = False
        
        return {
            "total_time": round(total_time, 2),
            "tasks_processed": len(completed_tasks),
            "tasks_failed": len([t for t in completed_tasks if t.status == TaskStatus.FAILED]),
            "results": completed_tasks
        }
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about worker performance"""
        stats = {}
        for worker in self.workers:
            stats[worker.worker_id] = {
                "type": worker.worker_type.value,
                "tasks_completed": worker.tasks_completed,
                "total_processing_time": round(worker.total_processing_time, 2),
                "average_time": round(
                    worker.total_processing_time / worker.tasks_completed 
                    if worker.tasks_completed > 0 else 0, 2
                ),
                "is_busy": worker.is_busy
            }
        return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_workers = len(self.workers)
        busy_workers = sum(1 for worker in self.workers if worker.is_busy)
        pending_tasks = self.task_queue.qsize()
        
        return {
            "total_workers": total_workers,
            "busy_workers": busy_workers,
            "available_workers": total_workers - busy_workers,
            "pending_tasks": pending_tasks,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "is_running": self.is_running
        }

# Global orchestrator instance
orchestrator = AIOrchestrator()

def format_orchestrator_results(results: Dict[str, Any]) -> str:
    """Format orchestrator results for display"""
    if "message" in results and results["message"]:
        return f"ℹ️ {results['message']}"
    
    output = f"""
# 🎭 Orchestrator-Worker Processing Results

## ⚡ Performance Summary
- **⏱️ Total Processing Time**: {results['total_time']} seconds
- **📋 Tasks Processed**: {results['tasks_processed']}
- **❌ Tasks Failed**: {results['tasks_failed']}
- **✅ Success Rate**: {round((results['tasks_processed'] - results['tasks_failed']) / results['tasks_processed'] * 100, 1) if results['tasks_processed'] > 0 else 0}%

---

## 📊 Task Results

"""
    
    for task in results['results']:
        status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌"
        worker_icon = {
            WorkerType.DOCUMENT_ANALYZER: "📄",
            WorkerType.SENTIMENT_ANALYZER: "😊", 
            WorkerType.TRANSLATOR: "🌍",
            WorkerType.SUMMARIZER: "📝",
            WorkerType.QUESTION_GENERATOR: "❓",
            WorkerType.CATEGORY_CLASSIFIER: "📂"
        }.get(task.worker_type, "🔧")
        
        output += f"""
### {status_icon} {worker_icon} {task.worker_type.value.replace('_', ' ').title()}
**Task ID**: `{task.id}`  
**Processing Time**: {task.processing_time:.2f}s  
**Status**: {task.status.value}

**Result**: {task.result[:200] if task.result else task.error}{'...' if task.result and len(task.result) > 200 else ''}

---
"""
    
    return output

def format_worker_stats(stats: Dict[str, Any]) -> str:
    """Format worker statistics for display"""
    output = """
# 👥 Worker Performance Statistics

| Worker ID | Type | Tasks Completed | Total Time (s) | Avg Time (s) | Status |
|-----------|------|-----------------|----------------|--------------|--------|
"""
    
    for worker_id, stat in stats.items():
        status = "🔴 Busy" if stat['is_busy'] else "🟢 Available"
        worker_type = stat['type'].replace('_', ' ').title()
        
        output += f"| `{worker_id}` | {worker_type} | {stat['tasks_completed']} | {stat['total_processing_time']} | {stat['average_time']} | {status} |\n"
    
    return output

def format_system_status(status: Dict[str, Any]) -> str:
    """Format system status for display"""
    utilization = round((status['busy_workers'] / status['total_workers']) * 100, 1) if status['total_workers'] > 0 else 0
    
    return f"""
# 🖥️ System Status Dashboard

## 👥 Worker Pool Status
- **Total Workers**: {status['total_workers']}
- **🔴 Busy Workers**: {status['busy_workers']}
- **🟢 Available Workers**: {status['available_workers']}
- **📊 Utilization**: {utilization}%

## 📋 Task Queue Status
- **⏳ Pending Tasks**: {status['pending_tasks']}
- **✅ Completed Tasks**: {status['completed_tasks']}
- **❌ Failed Tasks**: {status['failed_tasks']}

## 🔄 System State
- **Status**: {'🟢 Running' if status['is_running'] else '🔴 Idle'}
"""

def run_single_task(input_text: str, worker_type_str: str):
    """Run a single task through the orchestrator"""
    if not input_text.strip():
        return "Please enter text to process."
    
    try:
        worker_type = WorkerType(worker_type_str.lower().replace(' ', '_'))
        task_id = orchestrator.submit_task(worker_type, input_text, priority=1)
        results = orchestrator.process_tasks()
        return format_orchestrator_results(results)
    except Exception as e:
        return f"Error: {str(e)}"

def run_workflow(input_text: str):
    """Run a complete workflow with multiple AI tasks"""
    if not input_text.strip():
        return "Please enter text to process."
    
    try:
        # Define workflow: all AI tasks
        workflow_tasks = [
            WorkerType.SUMMARIZER,
            WorkerType.SENTIMENT_ANALYZER,
            WorkerType.CATEGORY_CLASSIFIER,
            WorkerType.QUESTION_GENERATOR,
            WorkerType.TRANSLATOR,
            WorkerType.DOCUMENT_ANALYZER
        ]
        
        task_ids = orchestrator.submit_workflow(input_text, workflow_tasks)
        results = orchestrator.process_tasks()
        return format_orchestrator_results(results)
    except Exception as e:
        return f"Error: {str(e)}"

def get_stats():
    """Get worker and system statistics"""
    worker_stats = orchestrator.get_worker_stats()
    system_status = orchestrator.get_system_status()
    
    return (
        format_worker_stats(worker_stats),
        format_system_status(system_status)
    )

# Sample texts for demonstration
SAMPLE_TEXTS = {
    "Technology Report": """The latest developments in artificial intelligence are reshaping business operations across industries. Machine learning algorithms now process data with unprecedented accuracy, enabling companies to make data-driven decisions faster than ever before. Cloud computing infrastructure supports scalable AI deployments, while edge computing brings intelligence closer to data sources. However, organizations must address challenges around data privacy, algorithm bias, and workforce adaptation to fully realize AI's potential.""",
    
    "Customer Feedback": """I'm extremely disappointed with the service I received today. The staff was unhelpful and seemed uninterested in resolving my issue. I waited over an hour just to speak with someone, and when I finally did, they couldn't provide a satisfactory solution. This experience has really changed my opinion of your company. I've been a loyal customer for five years, but I'm seriously considering taking my business elsewhere if things don't improve.""",
    
    "Medical Research": """A recent study of 2,400 participants examined the effectiveness of a new treatment approach for chronic pain management. Results showed a 42% reduction in pain scores over a 12-week period, with minimal side effects reported in less than 6% of patients. The treatment combines physical therapy with mindfulness-based interventions and shows promise for improving quality of life. Further research is needed to understand long-term effects and optimal treatment duration."""
}

# Create Gradio interface
with gr.Blocks(title="AI Orchestrator-Worker Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 AI Orchestrator-Worker Pattern Demonstration
    
    Experience distributed AI processing through a sophisticated orchestrator-worker architecture. This system coordinates multiple specialized AI workers to handle complex workflows efficiently.
    """)
    
    gr.Markdown("""
    ## 🏗️ Architecture Overview:
    
    **🎭 Orchestrator**: Central coordinator that manages task distribution and worker allocation
    **👥 Worker Pool**: 7 specialized AI workers handling different types of analysis
    **📋 Task Queue**: Priority-based task scheduling and load balancing
    **⚡ Parallel Execution**: Concurrent processing for maximum efficiency
    
    **Available Workers**: Document Analyzer (2x), Sentiment Analyzer, Translator, Summarizer, Question Generator, Category Classifier
    """)
    
    with gr.Tabs():
        # Tab 1: Single Task Processing
        with gr.TabItem("🔧 Single Task Processing"):
            gr.Markdown("**Process individual tasks through specific workers**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    single_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text to process...",
                        lines=6,
                        value=SAMPLE_TEXTS["Technology Report"]
                    )
                    
                    worker_type_single = gr.Dropdown(
                        choices=[
                            "Document Analyzer",
                            "Sentiment Analyzer", 
                            "Translator",
                            "Summarizer",
                            "Question Generator",
                            "Category Classifier"
                        ],
                        label="🔧 Select Worker Type",
                        value="Document Analyzer"
                    )
                    
                    single_sample = gr.Dropdown(
                        choices=list(SAMPLE_TEXTS.keys()),
                        label="📚 Sample Texts",
                        value="Technology Report"
                    )
                    
                    single_btn = gr.Button("🚀 Process Task", variant="primary")
                    
                with gr.Column(scale=2):
                    single_output = gr.Markdown(label="📊 Task Results")
            
            # Load sample text
            def load_single_sample(sample_key):
                return SAMPLE_TEXTS.get(sample_key, "")
            
            single_sample.change(
                fn=load_single_sample,
                inputs=single_sample,
                outputs=single_input
            )
            
            single_btn.click(
                fn=run_single_task,
                inputs=[single_input, worker_type_single],
                outputs=single_output
            )
        
        # Tab 2: Complete Workflow
        with gr.TabItem("🔄 Complete Workflow"):
            gr.Markdown("**Process text through all AI workers in a coordinated workflow**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    workflow_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text for complete AI workflow processing...",
                        lines=8,
                        value=SAMPLE_TEXTS["Customer Feedback"]
                    )
                    
                    workflow_sample = gr.Dropdown(
                        choices=list(SAMPLE_TEXTS.keys()),
                        label="📚 Sample Texts",
                        value="Customer Feedback"
                    )
                    
                    workflow_btn = gr.Button("🎭 Run Complete Workflow", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    workflow_output = gr.Markdown(label="🎯 Workflow Results")
            
            # Load sample text for workflow
            def load_workflow_sample(sample_key):
                return SAMPLE_TEXTS.get(sample_key, "")
            
            workflow_sample.change(
                fn=load_workflow_sample,
                inputs=workflow_sample,
                outputs=workflow_input
            )
            
            workflow_btn.click(
                fn=run_workflow,
                inputs=workflow_input,
                outputs=workflow_output
            )
        
        # Tab 3: System Monitoring
        with gr.TabItem("📊 System Monitoring"):
            gr.Markdown("**Monitor worker performance and system status in real-time**")
            
            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
                    
                with gr.Column():
                    gr.Markdown("*Click refresh to update worker statistics and system status*")
            
            with gr.Row():
                with gr.Column():
                    worker_stats_output = gr.Markdown(label="👥 Worker Statistics")
                    
                with gr.Column():
                    system_status_output = gr.Markdown(label="🖥️ System Status")
            
            def refresh_stats():
                return get_stats()
            
            stats_btn.click(
                fn=refresh_stats,
                outputs=[worker_stats_output, system_status_output]
            )
            
            # Initial load of stats
            demo.load(
                fn=refresh_stats,
                outputs=[worker_stats_output, system_status_output]
            )
    
    gr.Markdown("""
    ### 🎯 Orchestrator-Worker Benefits:
    
    **🏗️ Architectural Advantages:**
    - **Scalability**: Add/remove workers dynamically based on demand
    - **Specialization**: Workers optimized for specific AI tasks
    - **Fault Tolerance**: Failed workers don't bring down the entire system
    - **Load Balancing**: Intelligent task distribution across available workers
    
    **💼 Business Applications:**
    - **🎖️ Veterans Affairs**: Distributed processing of claims, medical records, and appeals
    - **📊 Enterprise Document Processing**: Large-scale document analysis workflows  
    - **📞 Customer Service Centers**: Parallel analysis of support tickets and feedback
    - **🏥 Healthcare Systems**: Coordinated analysis of patient records and research data
    - **📈 Financial Services**: Distributed risk analysis and compliance checking
    
    **⚡ Performance Benefits:**
    - **Parallel Processing**: Multiple tasks execute simultaneously
    - **Resource Optimization**: Efficient utilization of AI API limits
    - **Priority Handling**: Critical tasks processed first
    - **Monitoring & Analytics**: Real-time performance tracking
    
    This orchestrator-worker pattern demonstrates how complex AI workflows can be distributed, coordinated, and scaled efficiently!
    """)

if __name__ == "__main__":
    print("=== AI Orchestrator-Worker Demo ===\n")
    print("🎭 Starting distributed AI processing system...")
    print("Orchestrator managing multiple specialized AI workers\n")
    
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7863)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)