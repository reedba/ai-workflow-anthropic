
import openai
import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)



class PromptChain:
    """
    True prompt chaining implementation with separate, composable methods.
    Each step is independent and can be reused in different workflows.
    """
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.chain_history = []
    
    def step_1_summarize(self, content: str) -> dict:
        """Step 1: Extract and summarize document content"""
        prompt = f"""Analyze and summarize the following document in 3-4 clear, concise sentences. Focus on the key message and purpose:

Document Content:
{content}"""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert document analyst who creates clear, professional summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        
        result = {
            "step": "summarize",
            "input": content[:200] + "..." if len(content) > 200 else content,
            "output": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens
        }
        
        self.chain_history.append(result)
        return result
    
    def step_2_extract_points(self, summary_result: dict, original_content: str = "") -> dict:
        """Step 2: Identify main points from summary (with optional original content context)"""
        summary = summary_result["output"]
        
        prompt = f"""Based on this document summary, identify and list the 5-7 most important main points or key topics. Present them as a numbered list with brief explanations.

Summary: {summary}"""
        
        # Add original content context if available
        if original_content:
            prompt += f"\n\nOriginal content for reference:\n{original_content[:1000]}..."
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at identifying key points and themes in documents. Be specific and actionable."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        result = {
            "step": "extract_points",
            "input": summary,
            "output": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens,
            "depends_on": [summary_result["step"]]
        }
        
        self.chain_history.append(result)
        return result
    
    def step_3_categorize(self, summary_result: dict, points_result: dict) -> dict:
        """Step 3: Categorize and tag based on summary and main points"""
        summary = summary_result["output"]
        main_points = points_result["output"]
        
        prompt = f"""Based on the summary and main points below, provide:

1. **Primary Category**: Choose the most appropriate category (e.g., Business Report, Technical Documentation, Research Paper, Policy Document, Educational Material, Marketing Content, Legal Document, etc.)

2. **Tags**: Generate 5-8 relevant tags that describe the content, topics, and themes

3. **Industry/Domain**: Identify the primary industry or domain this document belongs to

Summary: {summary}

Main Points: {main_points}"""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert document classifier and tagging specialist. Provide accurate, specific categories and tags."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        
        result = {
            "step": "categorize",
            "input": {"summary": summary, "main_points": main_points},
            "output": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens,
            "depends_on": [summary_result["step"], points_result["step"]]
        }
        
        self.chain_history.append(result)
        return result
    
    def step_4_generate_insights(self, summary_result: dict, points_result: dict, categorization_result: dict) -> dict:
        """Step 4: Generate actionable insights from all previous steps"""
        summary = summary_result["output"]
        main_points = points_result["output"]
        categorization = categorization_result["output"]
        
        prompt = f"""Based on the document analysis, provide actionable insights and recommendations:

Summary: {summary}
Main Points: {main_points}
Categories/Tags: {categorization}

Provide:
1. Key takeaways for stakeholders
2. Potential next steps or actions
3. Areas that might need further attention or investigation"""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic analyst who provides actionable business insights and recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        result = {
            "step": "generate_insights",
            "input": {"summary": summary, "main_points": main_points, "categorization": categorization},
            "output": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens,
            "depends_on": [summary_result["step"], points_result["step"], categorization_result["step"]]
        }
        
        self.chain_history.append(result)
        return result
    
    def execute_full_chain(self, content: str) -> dict:
        """Execute the complete 4-step chain workflow"""
        if not content.strip():
            return {"error": "No content provided"}
        
        # Clear previous chain history
        self.chain_history = []
        
        try:
            # Execute each step, passing results to dependent steps
            step1 = self.step_1_summarize(content)
            step2 = self.step_2_extract_points(step1, content)
            step3 = self.step_3_categorize(step1, step2)
            step4 = self.step_4_generate_insights(step1, step2, step3)
            
            # Calculate total tokens used
            total_tokens = sum(step.get("tokens_used", 0) for step in self.chain_history)
            
            return {
                "success": True,
                "steps": {
                    "summary": step1["output"],
                    "main_points": step2["output"],
                    "categorization": step3["output"],
                    "insights": step4["output"]
                },
                "chain_metadata": {
                    "total_steps": len(self.chain_history),
                    "total_tokens": total_tokens,
                    "execution_order": [step["step"] for step in self.chain_history],
                    "dependencies": {step["step"]: step.get("depends_on", []) for step in self.chain_history}
                },
                "full_chain_history": self.chain_history
            }
            
        except Exception as e:
            return {"error": f"Chain execution failed: {str(e)}"}
    
    def execute_partial_chain(self, content: str, steps: list) -> dict:
        """Execute only specific steps of the chain (for testing/customization)"""
        if not content.strip():
            return {"error": "No content provided"}
        
        self.chain_history = []
        results = {}
        
        try:
            # Always need step 1 for other steps
            if any(step in steps for step in ["extract_points", "categorize", "insights"]):
                if "summarize" not in steps:
                    steps = ["summarize"] + steps
            
            step1 = None
            step2 = None
            step3 = None
            
            if "summarize" in steps:
                step1 = self.step_1_summarize(content)
                results["summary"] = step1["output"]
            
            if "extract_points" in steps and step1:
                step2 = self.step_2_extract_points(step1, content)
                results["main_points"] = step2["output"]
            
            if "categorize" in steps and step1 and step2:
                step3 = self.step_3_categorize(step1, step2)
                results["categorization"] = step3["output"]
            
            if "insights" in steps and step1 and step2 and step3:
                step4 = self.step_4_generate_insights(step1, step2, step3)
                results["insights"] = step4["output"]
            
            total_tokens = sum(step.get("tokens_used", 0) for step in self.chain_history)
            
            return {
                "success": True,
                "steps": results,
                "chain_metadata": {
                    "executed_steps": len(self.chain_history),
                    "total_tokens": total_tokens,
                    "execution_order": [step["step"] for step in self.chain_history]
                }
            }
            
        except Exception as e:
            return {"error": f"Partial chain execution failed: {str(e)}"}

def format_document_output(results):
    """Format the document processing results for display"""
    if "error" in results:
        return f"❌ **Error**: {results['error']}"
    metadata = results.get("chain_metadata", {})
    output = f"""
## 📊 Chain Execution Summary
- **Total Steps**: {metadata.get('total_steps', metadata.get('executed_steps', 0))}
- **Total Tokens**: {metadata.get('total_tokens', 0)}
- **Execution Order**: {' → '.join(metadata.get('execution_order', []))}

## 📄 Document Summary
{results['summary']}

## 🎯 Main Points
{results['main_points']}

## 🏷️ Categorization & Tags
{results['categorization']}

## 💡 Actionable Insights
{results['insights']}

---
*Processed using **true prompt chaining** - each step is a separate, reusable method*
"""
    return output

def gradio_document_interface(input_text):
    """Gradio interface function for document processing (text only)"""
    if not input_text.strip():
        return "Please enter text to analyze."
    try:
        chain = PromptChain()
        result = chain.execute_full_chain(input_text)
        if "error" in result:
            return f"❌ **Error**: {result['error']}"
        return format_document_output({
            "summary": result["steps"]["summary"],
            "main_points": result["steps"]["main_points"],
            "categorization": result["steps"]["categorization"],
            "insights": result["steps"]["insights"],
            "chain_metadata": result["chain_metadata"]
        })
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease make sure your OpenAI API key is set correctly in your .env file."

def gradio_chain_demo(input_text, selected_steps):
    """Demonstrate partial chain execution with selected steps"""
    if not input_text.strip():
        return "Please provide text content to analyze."
    
    try:
        chain = PromptChain()
        
        # Map display names to internal step names
        step_mapping = {
            "📄 Summarize": "summarize",
            "🎯 Extract Points": "extract_points", 
            "🏷️ Categorize": "categorize",
            "💡 Generate Insights": "insights"
        }
        
        # Convert selected steps to internal names
        internal_steps = [step_mapping[step] for step in selected_steps if step in step_mapping]
        
        if not internal_steps:
            return "Please select at least one step to execute."
        
        # Execute partial chain
        if len(internal_steps) == 4:
            result = chain.execute_full_chain(input_text)
        else:
            result = chain.execute_partial_chain(input_text, internal_steps)
        
        if "error" in result:
            return f"❌ **Error**: {result['error']}"
        
        # Format output
        output = f"""
## 🔗 Prompt Chain Execution Demo

### 📊 Chain Metadata
- **Steps Executed**: {result['chain_metadata'].get('executed_steps', result['chain_metadata'].get('total_steps', 0))}
- **Total Tokens**: {result['chain_metadata'].get('total_tokens', 0)}
- **Execution Order**: {' → '.join(result['chain_metadata'].get('execution_order', []))}

### 📋 Results
"""
        
        steps = result["steps"]
        if "summary" in steps:
            output += f"\n**📄 Summary:**\n{steps['summary']}\n"
        
        if "main_points" in steps:
            output += f"\n**🎯 Main Points:**\n{steps['main_points']}\n"
        
        if "categorization" in steps:
            output += f"\n**🏷️ Categorization:**\n{steps['categorization']}\n"
        
        if "insights" in steps:
            output += f"\n**💡 Insights:**\n{steps['insights']}\n"
        
        output += "\n---\n*This demonstrates **modular prompt chaining** - each step is independent and reusable*"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


# Create improved Gradio interface demonstrating true prompt chaining
with gr.Blocks(title="AI Prompt Chaining Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔗 True Prompt Chaining Architecture
    
    Demonstrates the **correct way** to implement prompt chaining with modular, reusable methods.
    """)
    
    gr.Markdown("""
    ## 🔄 Architecture Comparison:
    
    **❌ Sequential Prompting (What NOT to do):**
    ```python
    def bad_approach(content):
        # All steps in one big function
        result1 = ai_call(step1_prompt + content)
        result2 = ai_call(step2_prompt + result1)
        result3 = ai_call(step3_prompt + result2)
        return result3
    ```
    
    **✅ True Prompt Chaining (Correct approach):**
    ```python
    class PromptChain:
        def step_1_summarize(self, content): ...    # Modular
        def step_2_extract_points(self, summary): ...  # Reusable  
        def step_3_categorize(self, summary, points): ...  # Composable
        def execute_full_chain(self, content): ...  # Orchestration
    ```
    """)
    
    with gr.Tabs():
        # Tab 1: Full Chain Execution
        with gr.TabItem("🔗 Full Chain"):
            gr.Markdown("**Execute complete 4-step modular chain**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    full_chain_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text for full chain processing...",
                        lines=8,
                        value="Artificial intelligence is transforming business operations through automated decision-making, predictive analytics, and intelligent process automation. Companies are implementing AI solutions for customer service, supply chain optimization, and risk assessment. However, successful AI adoption requires careful planning, employee training, and robust data governance frameworks to ensure ethical and effective implementation."
                    )
                    full_chain_btn = gr.Button("🔗 Execute Full Chain", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    full_chain_output = gr.Markdown(label="📊 Full Chain Results")
            
            def execute_full_chain_demo(input_text):
                if not input_text.strip():
                    return "Please provide input text."
                try:
                    chain = PromptChain()
                    result = chain.execute_full_chain(input_text)
                    return format_document_output({
                        "summary": result["steps"]["summary"],
                        "main_points": result["steps"]["main_points"], 
                        "categorization": result["steps"]["categorization"],
                        "insights": result["steps"]["insights"],
                        "chain_metadata": result["chain_metadata"]
                    })
                except Exception as e:
                    return f"Error: {str(e)}"
            
            full_chain_btn.click(
                fn=execute_full_chain_demo,
                inputs=full_chain_input,
                outputs=full_chain_output
            )
        
        # Tab 2: Modular Steps
        with gr.TabItem("⚡ Modular Steps"):
            gr.Markdown("**Execute individual steps to demonstrate modularity**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    modular_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text for modular step execution...",
                        lines=6,
                        value="Cloud computing provides scalable infrastructure, cost efficiency, and improved collaboration for modern businesses. Key benefits include automatic updates, disaster recovery, and global accessibility."
                    )
                    
                    step_choices = gr.CheckboxGroup(
                        choices=["📄 Summarize", "🎯 Extract Points", "🏷️ Categorize", "💡 Generate Insights"],
                        label="🎯 Select Steps to Execute",
                        value=["📄 Summarize", "🎯 Extract Points"]
                    )
                    
                    modular_btn = gr.Button("⚡ Execute Selected Steps", variant="secondary", size="lg")
                    
                with gr.Column(scale=2):
                    modular_output = gr.Markdown(label="🔗 Modular Results")
            
            modular_btn.click(
                fn=gradio_chain_demo,
                inputs=[modular_input, step_choices],
                outputs=modular_output
            )
    
    gr.Markdown("""
    ### 🎯 True Prompt Chaining Benefits:
    
    **🔧 Modularity & Testability:**
    - Each step is an independent, testable method
    - Easy to debug individual components
    - Modify one step without affecting others
    - Unit test each step separately
    
    **🔄 Reusability & Composability:**
    - Reuse steps across different workflows
    - Mix and match steps for custom pipelines
    - Create branching logic based on conditions
    - Build complex workflows from simple components
    
    **📊 Observability & Control:**
    - Track dependencies between steps
    - Monitor token usage and costs per step
    - Cache intermediate results for efficiency
    - Retry individual steps on failure
    
    **⚡ Performance & Scalability:**
    - Parallel execution of independent steps
    - Early termination if prerequisite steps fail
    - Dynamic workflow adjustment based on results
    - Resource optimization per step
    
    **💼 Real-World Applications:**
    
    **🎖️ Veterans Affairs Claims Processing:**
    ```
    extract_claim_data() → classify_claim_type() → validate_documentation() → 
    assess_eligibility() → calculate_benefits() → generate_decision_letter()
    ```
    
    **📞 Customer Service Escalation:**
    ```
    analyze_inquiry() → determine_sentiment() → classify_urgency() → 
    route_to_specialist() → generate_response() → log_interaction()
    ```
    
    **🏥 Medical Diagnosis Workflow:**
    ```
    process_symptoms() → suggest_tests() → analyze_results() → 
    generate_diagnosis() → recommend_treatment() → create_care_plan()
    ```
    
    This is the **correct implementation** of prompt chaining - modular, reusable, and production-ready!
    """)

if __name__ == "__main__":
    print("=== AI Prompt Chaining Demo ===\n")
    print("🔗 Demonstrating true prompt chaining architecture...")
    print("Each step is modular, reusable, and composable\n")
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)
