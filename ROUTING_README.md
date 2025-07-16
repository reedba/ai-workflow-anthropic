# 🎯 AI-Powered Request Routing System

An intelligent routing workflow that automatically analyzes, classifies, and routes incoming requests to appropriate departments using advanced AI and Gradio interface.

## 🚀 Features

- **🎯 Smart Classification**: Automatically categorizes requests into 8 specialized departments
- **⚡ Priority Assessment**: Determines urgency levels and special handling requirements
- **📋 Action Planning**: Provides detailed routing summaries and next steps
- **🌐 Interactive Interface**: User-friendly Gradio web interface
- **📚 Sample Requests**: Pre-loaded examples for testing different scenarios
- **⏰ SLA Management**: Built-in service level agreements for each category

## 🔄 How AI Routing Works

The system uses a sophisticated 3-step AI workflow:

### 1. 🎯 **Classification**
- Analyzes request content and context
- Assigns to one of 8 specialized categories
- Uses natural language understanding to identify intent

### 2. ⚡ **Priority Assessment** 
- Evaluates urgency and time-sensitivity
- Identifies safety, security, or legal implications
- Assesses customer sentiment and frustration levels
- Determines special handling requirements

### 3. 📋 **Action Planning**
- Creates comprehensive routing summary
- Recommends immediate actions
- Identifies information needed from customer
- Suggests resolution approach and timeline

## 📂 Routing Categories

| Category | Handler | Priority | SLA | Description |
|----------|---------|----------|-----|-------------|
| 🛠️ **Technical Support** | Technical Team | High | 4 hours | System issues, bugs, troubleshooting |
| 📞 **Customer Support** | Customer Service | Medium | 24 hours | General inquiries, complaints |
| 💰 **Billing & Financial** | Billing Dept | High | 8 hours | Payment issues, refunds, billing |
| 📈 **Sales & Pre-sales** | Sales Team | Medium | 12 hours | Product inquiries, demos, quotes |
| ⚖️ **Legal & Compliance** | Legal Dept | Critical | 2 hours | Legal matters, compliance, privacy |
| 👥 **HR & Employment** | Human Resources | Medium | 48 hours | Job applications, employee relations |
| 🎖️ **Veterans Affairs** | VA Specialist | Critical | 1 hour | Claims, appeals, veteran services |
| 🚨 **Emergency/Urgent** | Emergency Team | Critical | 30 minutes | Critical issues requiring immediate attention |

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## 🛠️ Setup

1. **Install dependencies**
   ```bash
   pip install openai gradio python-dotenv
   ```

2. **Set up your OpenAI API key**
   
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your-actual-api-key-here
   ```

## 🎯 Usage

### Launch the Routing Interface
```bash
python routing.py
```

This will:
1. Start the AI routing analysis system
2. Launch interactive web interface at `http://localhost:7861`
3. Provide sample requests for testing

### How to Use:
1. **Enter Request**: Paste or type customer request
2. **Add Context** (optional): Provide additional information
3. **Try Samples**: Use dropdown to test with example requests
4. **Analyze**: Click "Analyze & Route" for intelligent routing

## 🏢 Business Applications

### **Customer Service Centers**
- Automated ticket routing and prioritization
- Reduced response times through intelligent classification
- Improved customer satisfaction with faster resolution

### **Healthcare Systems** 
- Patient inquiry routing and medical triage
- Emergency vs. routine classification
- Specialist referral optimization

### **Veterans Affairs**
- VA claim and appeal routing to appropriate specialists
- Medical record analysis and case classification
- Priority handling for urgent veteran needs

### **Enterprise Support**
- Internal request routing and SLA management
- IT help desk automation
- Cross-department workflow optimization

### **Communication Systems**
- Email classification and auto-routing
- Chat system intelligent agent assignment
- Multi-channel support coordination

## 💡 Sample Use Cases

**Technical Emergency**: 
*"Production server down, customers can't access service"*
→ Routes to Emergency Team (30-minute SLA)

**VA Claim Appeal**:
*"Need help with denied PTSD claim, have new medical evidence"*
→ Routes to VA Specialist (1-hour SLA)

**Billing Issue**:
*"Charged twice for subscription, need refund"*
→ Routes to Billing Department (8-hour SLA)

## 🔧 Technical Architecture

- **Frontend**: Gradio web interface
- **Backend**: OpenAI GPT-3.5-turbo for classification and analysis
- **Routing Logic**: Rule-based priority system with AI enhancement
- **Output**: Structured routing decisions with actionable insights

## 🛡️ Security & Compliance

- API keys stored securely in environment variables
- No customer data stored or logged
- GDPR and privacy-compliant processing
- Audit trail for routing decisions

## 🚀 Advanced Features

- **Multi-language Support**: Can be extended for international routing
- **Custom Categories**: Easily add new routing categories
- **Integration Ready**: API endpoints for CRM/ticketing system integration
- **Analytics Dashboard**: Track routing patterns and performance metrics

This AI-powered routing system demonstrates how intelligent automation can transform customer service operations, reduce response times, and ensure requests reach the right expertise quickly and efficiently.
