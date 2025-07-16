import openai
import os
import gradio as gr
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define routing categories and handlers
ROUTING_CATEGORIES = {
    "customer_support": {
        "name": "Customer Support",
        "description": "General customer inquiries, complaints, and support requests",
        "priority": "Medium",
        "handler": "Customer Service Team",
        "sla": "24 hours"
    },
    "technical_support": {
        "name": "Technical Support", 
        "description": "Technical issues, bugs, system problems, and troubleshooting",
        "priority": "High",
        "handler": "Technical Support Team",
        "sla": "4 hours"
    },
    "billing_financial": {
        "name": "Billing & Financial",
        "description": "Billing inquiries, payment issues, refunds, and financial questions",
        "priority": "High",
        "handler": "Billing Department",
        "sla": "8 hours"
    },
    "sales_presales": {
        "name": "Sales & Pre-sales",
        "description": "Product inquiries, quotes, demos, and new customer onboarding",
        "priority": "Medium",
        "handler": "Sales Team",
        "sla": "12 hours"
    },
    "legal_compliance": {
        "name": "Legal & Compliance",
        "description": "Legal matters, compliance issues, privacy concerns, and policy questions",
        "priority": "Critical",
        "handler": "Legal Department",
        "sla": "2 hours"
    },
    "hr_employment": {
        "name": "HR & Employment",
        "description": "Job applications, employee relations, benefits, and HR policies",
        "priority": "Medium",
        "handler": "Human Resources",
        "sla": "48 hours"
    },
    "veterans_affairs": {
        "name": "Veterans Affairs",
        "description": "VA claims, appeals, medical records, and veteran services",
        "priority": "Critical",
        "handler": "Veterans Affairs Specialist",
        "sla": "1 hour"
    },
    "emergency_urgent": {
        "name": "Emergency/Urgent",
        "description": "Critical issues requiring immediate attention",
        "priority": "Critical",
        "handler": "Emergency Response Team",
        "sla": "30 minutes"
    }
}

def analyze_and_route(input_text, context_info=""):
    """
    AI-powered routing system that analyzes input and determines optimal routing
    """
    try:
        # Step 1: Initial Analysis and Classification
        classification_prompt = f"""
        Analyze the following request and classify it into the most appropriate category.
        
        Request: {input_text}
        Additional Context: {context_info}
        
        Available Categories:
        - customer_support: General customer inquiries, complaints, and support requests
        - technical_support: Technical issues, bugs, system problems, and troubleshooting  
        - billing_financial: Billing inquiries, payment issues, refunds, and financial questions
        - sales_presales: Product inquiries, quotes, demos, and new customer onboarding
        - legal_compliance: Legal matters, compliance issues, privacy concerns, and policy questions
        - hr_employment: Job applications, employee relations, benefits, and HR policies
        - veterans_affairs: VA claims, appeals, medical records, and veteran services
        - emergency_urgent: Critical issues requiring immediate attention
        
        Respond with ONLY the category name (e.g., "technical_support").
        """
        
        classification_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert routing classifier. Respond with only the category name."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=50
        )
        
        category = classification_response.choices[0].message.content.strip().lower()
        
        # Validate category exists
        if category not in ROUTING_CATEGORIES:
            category = "customer_support"  # Default fallback
        
        # Step 2: Determine Priority and Urgency
        priority_prompt = f"""
        Based on this request, determine the urgency level and any special handling needed:
        
        Request: {input_text}
        Category: {ROUTING_CATEGORIES[category]['name']}
        
        Consider:
        - Is this time-sensitive?
        - Does it involve safety, security, or legal issues?
        - Is the customer expressing frustration or anger?
        - Are there financial implications?
        
        Respond with:
        1. Urgency Level: Low/Medium/High/Critical
        2. Special Notes: Any special handling instructions
        """
        
        priority_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at assessing request urgency and priority."},
                {"role": "user", "content": priority_prompt}
            ],
            max_tokens=150
        )
        
        priority_analysis = priority_response.choices[0].message.content.strip()
        
        # Step 3: Generate Routing Summary and Next Steps
        routing_prompt = f"""
        Create a routing summary and recommended actions for this request:
        
        Request: {input_text}
        Category: {ROUTING_CATEGORIES[category]['name']}
        Handler: {ROUTING_CATEGORIES[category]['handler']}
        Priority Analysis: {priority_analysis}
        
        Provide:
        1. Brief summary of the request
        2. Key issues identified
        3. Recommended immediate actions
        4. Information needed from customer
        5. Estimated resolution approach
        """
        
        routing_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert customer service routing specialist who provides clear, actionable guidance."},
                {"role": "user", "content": routing_prompt}
            ],
            max_tokens=300
        )
        
        routing_summary = routing_response.choices[0].message.content.strip()
        
        # Prepare results
        results = {
            "category": category,
            "category_info": ROUTING_CATEGORIES[category],
            "priority_analysis": priority_analysis,
            "routing_summary": routing_summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_request": input_text[:200] + "..." if len(input_text) > 200 else input_text
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Routing analysis failed: {str(e)}"}

def format_routing_output(results):
    """Format the routing results for display"""
    if "error" in results:
        return f"❌ **Error**: {results['error']}"
    
    category_info = results['category_info']
    
    # Priority color coding
    priority_colors = {
        "Critical": "🔴",
        "High": "🟠", 
        "Medium": "🟡",
        "Low": "🟢"
    }
    
    priority_icon = priority_colors.get(category_info['priority'], "⚪")
    
    output = f"""
## 🎯 Routing Decision

**📂 Category**: {category_info['name']}  
**{priority_icon} Priority**: {category_info['priority']}  
**👥 Handler**: {category_info['handler']}  
**⏰ SLA**: {category_info['sla']}  
**🕐 Analyzed**: {results['timestamp']}

---

## 📋 Priority Analysis
{results['priority_analysis']}

---

## 📊 Routing Summary & Next Steps
{results['routing_summary']}

---

## 📄 Original Request (Preview)
*{results['original_request']}*

---
*Request processed through AI-powered routing: Classification → Priority Analysis → Action Planning*
"""
    return output

def gradio_routing_interface(request_text, context_info):
    """Gradio interface function for routing"""
    if not request_text.strip():
        return "Please enter a request to analyze and route."
    
    try:
        results = analyze_and_route(request_text, context_info)
        return format_routing_output(results)
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease make sure your OpenAI API key is set correctly in your .env file."

# Sample requests for demonstration
SAMPLE_REQUESTS = {
    "Technical Issue": "My application keeps crashing when I try to upload files larger than 5MB. I've tried restarting but the problem persists. This is affecting my daily work.",
    "Billing Question": "I was charged twice for my monthly subscription. I need a refund for the duplicate charge of $99.99 from last week.",
    "Sales Inquiry": "I'm interested in your enterprise plan for my company of 150 employees. Can you provide pricing and schedule a demo?",
    "VA Claim": "I need help with my VA disability claim appeal. The VA denied my claim for PTSD and I have new medical evidence to submit.",
    "Emergency": "Our production server is down and customers cannot access our service. We need immediate assistance.",
    "HR Question": "I have questions about my health insurance benefits and want to add my new spouse to my coverage."
}

# Create Gradio interface
with gr.Blocks(title="AI-Powered Request Routing", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 AI-Powered Request Routing System
    
    Intelligent routing system that analyzes incoming requests and automatically routes them to the appropriate department with priority assessment and action recommendations.
    """)
    
    gr.Markdown("""
    ## 🔄 3-Step AI Routing Process:
    1. **🎯 Classification**: Analyzes request content and assigns appropriate category
    2. **⚡ Priority Assessment**: Determines urgency level and special handling needs
    3. **📋 Action Planning**: Provides routing summary and recommended next steps
    
    **Routing Categories**: Technical Support, Customer Service, Billing, Sales, Legal, HR, Veterans Affairs, Emergency
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            request_input = gr.Textbox(
                label="📝 Customer Request",
                placeholder="Enter the customer request, inquiry, or issue...",
                lines=6
            )
            context_input = gr.Textbox(
                label="📋 Additional Context (Optional)",
                placeholder="Customer type, account info, previous interactions, etc.",
                lines=2
            )
            
            # Sample requests dropdown
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_REQUESTS.keys()),
                label="📚 Try Sample Requests",
                value=None
            )
            
            route_btn = gr.Button("🚀 Analyze & Route", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            routing_output = gr.Markdown(label="📊 Routing Analysis")
    
    # Handle sample selection
    def load_sample(sample_key):
        if sample_key:
            return SAMPLE_REQUESTS[sample_key]
        return ""
    
    sample_dropdown.change(
        fn=load_sample,
        inputs=sample_dropdown,
        outputs=request_input
    )
    
    route_btn.click(
        fn=gradio_routing_interface,
        inputs=[request_input, context_input],
        outputs=routing_output
    )
    
    gr.Markdown("""
    ### 🎯 Business Applications:
    - **📞 Customer Service Centers**: Automated ticket routing and prioritization
    - **🏥 Healthcare Systems**: Patient inquiry routing and triage
    - **🎖️ Veterans Affairs**: Claim and appeal routing to appropriate specialists  
    - **🏢 Enterprise Support**: Internal request routing and SLA management
    - **📧 Email Management**: Intelligent email classification and routing
    - **💬 Chat Systems**: Real-time conversation routing to specialized agents
    
    This AI routing system demonstrates how intelligent classification can streamline operations and improve response times!
    """)

if __name__ == "__main__":
    print("=== AI-Powered Request Routing Demo ===\n")
    print("🎯 Starting Intelligent Routing Interface...")
    print("Analyze and route customer requests automatically\n")
    
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7861)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)