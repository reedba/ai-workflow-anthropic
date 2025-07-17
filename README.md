# 🤖 AI Workflow Demonstrations Suite

A comprehensive collection of AI-powered workflow demonstrations showcasing docum### 🧪 **Evaluator-Optimizer Workflow**
Advanced AI evaluation and optimization system:
1. **🧪 Test Case Creation**: Define input-output pairs for systematic AI testing
2. **📊 Multi-Metric Assessment**: Evaluate accuracy, relevance, completeness, and clarity
3. **🎯 Systematic Optimization**: Test multiple prompt variations and parameter configurations
4. **📈 Performance Analytics**: Track improvements and generate optimization recommendations

**Optimization Focus Areas**: Speed & Cost, Quality & Accuracy, Balanced Performance

### 🎭 **Orchestrator-Worker Workflow**
Distributed AI processing system with specialized workers:
1. **🎭 Central Orchestrator**: Manages task distribution, priority queuing, and worker allocation
2. **👥 Specialized Workers**: 7 AI workers handling different analysis types (Document Analysis, Sentiment, Translation, etc.)
3. **📋 Task Coordination**: Priority-based scheduling with concurrent processing capabilities
4. **📊 Performance Monitoring**: Real-time worker statistics and system health tracking

**Worker Types**: Document Analyzer (2x), Sentiment Analyzer, Translator, Summarizer, Question Generator, Category Classifier

**Key Features**: Fault tolerance, load balancing, scalable architecture, distributed processingessing, intelligent routing, parallelization techniques, and orchestrator-worker patterns using OpenAI's API and Gradio interfaces.

## 🚀 Available Workflows

### 📄 **1. True Prompt Chaining & Document Processing**
**File**: `prompt_chaining.py` | **Port**: `7860`
- **Modular Architecture**: Each step is an independent, reusable method
- **4-Step Chain**: Summarize → Extract Points → Categorize → Generate Insights
- **Composable Workflows**: Execute individual steps or full chains
- **Dependency Tracking**: Monitor step relationships and token usage
- **Production-Ready**: Proper error handling and observability

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
- **Business Applications**: Batch processing and workflow acceleration

### � **4. Orchestrator-Worker System**
**File**: `orchestrator_worker.py` | **Port**: `7863`
- **Distributed Architecture**: Central orchestrator managing specialized AI workers
- **7 Specialized Workers**: Document Analyzer (2x), Sentiment, Translation, Summarization, Questions, Classification
- **Priority Task Queue**: Intelligent task scheduling and load balancing
- **Real-time Monitoring**: Worker performance tracking and system health analytics
- **Enterprise Features**: Fault tolerance, horizontal scaling, cost optimization

### 🧪 **5. AI Evaluator-Optimizer System**
**File**: `evaluator_optimizer.py` | **Port**: `7864`
- **Multi-Metric Evaluation**: Accuracy, relevance, completeness, clarity assessment
- **Systematic Optimization**: Prompt engineering and parameter tuning
- **Performance Analytics**: Real-time metrics and optimization tracking
- **A/B Testing**: Scientific comparison of AI configurations

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

# Terminal 4: Orchestrator-Worker System
python orchestrator_worker.py
# Access at: http://localhost:7863

# Terminal 5: Evaluator-Optimizer System
python evaluator_optimizer.py
# Access at: http://localhost:7864
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

- **📄 Prompt Chaining**: Experience modular prompt chaining with step-by-step execution
- **🎯 Request Routing**: Enter customer requests for intelligent routing decisions  
- **⚡ Parallelization**: Compare sequential vs parallel AI processing performance
- **🎭 Orchestrator-Worker**: Distributed AI processing with specialized worker coordination
- **🧪 Evaluator-Optimizer**: Systematic AI testing and optimization dashboard

### **Features Available:**
- **Sample Data**: Pre-loaded examples for testing
- **Real-time Processing**: Live AI analysis and results
- **Performance Metrics**: Processing times and efficiency measurements
- **Export Options**: Copy results for further use

## 🔄 Workflow Descriptions

### 📄 **Prompt Chaining Workflow**
Modular 4-step prompt chaining architecture:
1. **📄 Document Summary**: Analyzes input text using specialized summarization prompts
2. **🎯 Main Points**: Extracts key insights through targeted analysis chains  
3. **🏷️ Smart Categorization**: Applies classification prompts with dependency tracking
4. **💡 Actionable Insights**: Generates recommendations through multi-step reasoning chains

**Key Features**: Step-by-step execution, partial chain capabilities, result tracking, composable workflows

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

### � **Evaluator-Optimizer Workflow**
Advanced AI evaluation and optimization system:
1. **🧪 Test Case Creation**: Define input-output pairs for systematic AI testing
2. **� Multi-Metric Assessment**: Evaluate accuracy, relevance, completeness, and clarity
3. **🎯 Systematic Optimization**: Test multiple prompt variations and parameter configurations
4. **� Performance Analytics**: Track improvements and generate optimization recommendations

**Optimization Focus Areas**: Speed & Cost, Quality & Accuracy, Balanced Performance

## 📁 Project Structure

```
ai-workflow-anthropic/
├── prompt_chaining.py          # Modular prompt chaining workflow
├── routing.py                  # Intelligent request routing  
├── parallelization.py          # Concurrent AI processing demonstration
├── orchestrator_worker.py      # Distributed orchestrator-worker system
├── evaluator_optimizer.py      # AI evaluation and optimization system
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore file
├── README.md                 # Main project documentation
├── ROUTING_README.md         # Routing workflow details
├── PARALLELIZATION_README.md # Parallelization documentation
├── ORCHESTRATOR_README.md    # Orchestrator-worker pattern details
├── EVALUATOR_README.md       # Evaluation-optimization documentation
└── sample_business_report.txt # Sample document for testing
```

## 🎭 Orchestrator-Worker Architecture Deep Dive

The orchestrator-worker pattern represents a sophisticated distributed AI processing system designed for enterprise-scale applications.

### 🏗️ **System Architecture**

**🎭 Central Orchestrator:**
- Task queue management with priority-based scheduling
- Dynamic worker allocation and load balancing
- Fault tolerance and error handling
- Real-time performance monitoring and analytics
- Thread pool management for concurrent execution

**👥 Specialized Worker Pool:**
```
📄 Document Analyzer (2x)    - Comprehensive document analysis and insights
😊 Sentiment Analyzer (1x)   - Emotional tone and sentiment detection  
🌍 Translator (1x)           - Multi-language translation services
📝 Summarizer (1x)           - Intelligent text summarization
❓ Question Generator (1x)    - Contextual question creation
📂 Category Classifier (1x)  - Content categorization and tagging
```

### ⚡ **Key Advantages**

**Scalability & Performance:**
- **Horizontal Scaling**: Add/remove workers dynamically based on demand
- **Parallel Processing**: Multiple tasks execute simultaneously across workers
- **Resource Optimization**: Efficient utilization of AI API rate limits
- **Load Distribution**: Intelligent task routing to available workers

**Reliability & Monitoring:**
- **Fault Isolation**: Worker failures don't affect the entire system
- **Health Monitoring**: Real-time worker status and performance tracking
- **Task Recovery**: Failed tasks can be redistributed to healthy workers
- **Performance Analytics**: Detailed metrics on processing times and success rates

**Enterprise Features:**
- **Priority Queuing**: Critical tasks processed first
- **Worker Specialization**: Optimized prompts and configurations per task type
- **Workflow Orchestration**: Coordinate complex multi-step AI processes
- **Cost Management**: Track and optimize AI API usage across the system

## 🤝 Business Applications

This AI workflow system can be applied to:

**Prompt Chaining Applications:**
- **📊 Business Reports**: Automated analysis and strategic insight extraction
- **📋 Policy Documents**: Compliance review and action item identification
- **📄 Contract Analysis**: Risk assessment and key terms extraction
- **📈 Market Research**: Data analysis and strategic recommendations

**Intelligent Routing Applications:**
- **� Customer Support**: Automated ticket classification and priority assignment
- **� Email Management**: Smart sorting and department routing
- **🎯 Lead Qualification**: Sales inquiry categorization and follow-up routing
- **🏥 Healthcare Triage**: Patient inquiry routing and urgency assessment

**System Optimization Applications:**
- **🧪 A/B Testing**: Prompt variation testing and performance optimization
- **📈 Performance Monitoring**: Real-time AI system evaluation and improvement
- **🎯 Quality Assurance**: Automated testing of AI outputs and accuracy tracking
- **⚡ Resource Optimization**: Processing efficiency and cost optimization

**Distributed Processing Applications:**
- **🎭 Enterprise Document Processing**: Large-scale document analysis with specialized workers
- **🏥 Healthcare Systems**: Coordinated analysis of patient records, medical imaging, and research data
- **🎖️ Veterans Affairs**: Distributed processing of claims, appeals, medical records, and benefits analysis
- **📊 Financial Services**: Parallel risk analysis, compliance checking, and fraud detection
- **📞 Contact Centers**: Real-time analysis of support tickets, chat logs, and customer feedback
- **🔍 Legal Discovery**: Coordinated review of legal documents with specialized analysis workers
- **⚡ Resource Optimization**: Processing efficiency and cost optimization

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
