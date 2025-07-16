import openai
import os
import gradio as gr
from dotenv import load_dotenv
import json
import time
import statistics
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class EvaluationMetric(Enum):
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RESPONSE_TIME = "response_time"
    COST_EFFICIENCY = "cost_efficiency"

class OptimizationType(Enum):
    PROMPT_ENGINEERING = "prompt_engineering"
    PARAMETER_TUNING = "parameter_tuning"
    MODEL_SELECTION = "model_selection"
    TEMPERATURE_OPTIMIZATION = "temperature_optimization"

@dataclass
class TestCase:
    id: str
    input_text: str
    expected_output: str
    category: str
    priority: int = 1

@dataclass
class EvaluationResult:
    test_case_id: str
    prompt_version: str
    model: str
    temperature: float
    max_tokens: int
    actual_output: str
    response_time: float
    token_count: int
    cost: float
    scores: Dict[str, float]
    timestamp: datetime

@dataclass
class OptimizationConfig:
    models: List[str]
    temperatures: List[float]
    max_tokens_options: List[int]
    prompt_variations: List[str]

class AIEvaluator:
    """Evaluates AI model performance across different metrics"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.evaluation_results: List[EvaluationResult] = []
        
    def add_test_case(self, input_text: str, expected_output: str, category: str, priority: int = 1):
        """Add a test case for evaluation"""
        test_id = f"test_{len(self.test_cases) + 1}_{int(time.time())}"
        test_case = TestCase(test_id, input_text, expected_output, category, priority)
        self.test_cases.append(test_case)
        return test_id
    
    def evaluate_response(self, actual_output: str, expected_output: str, 
                         input_text: str) -> Dict[str, float]:
        """Evaluate AI response against multiple metrics"""
        scores = {}
        
        # Accuracy (semantic similarity)
        scores[EvaluationMetric.ACCURACY.value] = self._evaluate_accuracy(actual_output, expected_output)
        
        # Relevance (how well it addresses the input)
        scores[EvaluationMetric.RELEVANCE.value] = self._evaluate_relevance(actual_output, input_text)
        
        # Completeness (coverage of expected elements)
        scores[EvaluationMetric.COMPLETENESS.value] = self._evaluate_completeness(actual_output, expected_output)
        
        # Clarity (readability and coherence)
        scores[EvaluationMetric.CLARITY.value] = self._evaluate_clarity(actual_output)
        
        return scores
    
    def _evaluate_accuracy(self, actual: str, expected: str) -> float:
        """Evaluate semantic accuracy using AI"""
        try:
            prompt = f"""
            Compare these two texts and rate their semantic similarity on a scale of 0-100:
            
            Expected: {expected}
            Actual: {actual}
            
            Consider:
            - Same key information
            - Similar meaning
            - Equivalent facts
            
            Respond with only a number between 0-100.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'\d+', score_text)[0]) / 100.0
            return min(max(score, 0.0), 1.0)
            
        except:
            return 0.5  # Default fallback
    
    def _evaluate_relevance(self, actual: str, input_text: str) -> float:
        """Evaluate how relevant the response is to the input"""
        try:
            prompt = f"""
            Rate how well this response addresses the input question/request on a scale of 0-100:
            
            Input: {input_text}
            Response: {actual}
            
            Consider:
            - Directly addresses the question
            - Stays on topic
            - Provides relevant information
            
            Respond with only a number between 0-100.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'\d+', score_text)[0]) / 100.0
            return min(max(score, 0.0), 1.0)
            
        except:
            return 0.5
    
    def _evaluate_completeness(self, actual: str, expected: str) -> float:
        """Evaluate completeness of response"""
        try:
            # Simple keyword overlap analysis
            expected_words = set(expected.lower().split())
            actual_words = set(actual.lower().split())
            
            if len(expected_words) == 0:
                return 1.0
            
            overlap = len(expected_words.intersection(actual_words))
            return overlap / len(expected_words)
            
        except:
            return 0.5
    
    def _evaluate_clarity(self, actual: str) -> float:
        """Evaluate clarity and readability"""
        try:
            # Basic clarity metrics
            sentences = actual.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Penalize very long or very short sentences
            if avg_sentence_length < 5 or avg_sentence_length > 30:
                length_score = 0.5
            else:
                length_score = 1.0
            
            # Check for clear structure
            structure_score = 0.8 if len(sentences) > 1 else 0.6
            
            return (length_score + structure_score) / 2
            
        except:
            return 0.7

class AIOptimizer:
    """Optimizes AI prompts and parameters based on evaluation results"""
    
    def __init__(self, evaluator: AIEvaluator):
        self.evaluator = evaluator
        self.optimization_history: List[Dict] = []
    
    def optimize_prompt(self, base_prompt: str, test_cases: List[TestCase], 
                       optimization_config: OptimizationConfig) -> Dict[str, Any]:
        """Optimize prompt through systematic testing"""
        
        optimization_results = {
            "base_prompt": base_prompt,
            "best_configuration": None,
            "best_score": 0.0,
            "all_results": [],
            "optimization_summary": {}
        }
        
        best_config = None
        best_score = 0.0
        all_results = []
        
        # Test different configurations
        for model in optimization_config.models:
            for temperature in optimization_config.temperatures:
                for max_tokens in optimization_config.max_tokens_options:
                    for i, prompt_variation in enumerate(optimization_config.prompt_variations):
                        
                        config_results = self._test_configuration(
                            prompt_variation, model, temperature, max_tokens, test_cases
                        )
                        
                        avg_score = np.mean([
                            np.mean(list(result.scores.values())) 
                            for result in config_results
                        ])
                        
                        config_summary = {
                            "prompt_version": f"v{i+1}",
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "average_score": avg_score,
                            "results": config_results
                        }
                        
                        all_results.append(config_summary)
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_config = config_summary
        
        optimization_results["best_configuration"] = best_config
        optimization_results["best_score"] = best_score
        optimization_results["all_results"] = all_results
        optimization_results["optimization_summary"] = self._generate_optimization_summary(all_results)
        
        self.optimization_history.append(optimization_results)
        return optimization_results
    
    def _test_configuration(self, prompt: str, model: str, temperature: float, 
                          max_tokens: int, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Test a specific configuration against test cases"""
        results = []
        
        for test_case in test_cases:
            start_time = time.time()
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": test_case.input_text}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                response_time = time.time() - start_time
                actual_output = response.choices[0].message.content.strip()
                token_count = response.usage.total_tokens
                
                # Calculate approximate cost (simplified)
                cost = self._calculate_cost(model, token_count)
                
                # Evaluate response
                scores = self.evaluator.evaluate_response(
                    actual_output, test_case.expected_output, test_case.input_text
                )
                
                # Add performance metrics
                scores[EvaluationMetric.RESPONSE_TIME.value] = min(1.0, max(0.0, 1.0 - (response_time / 10.0)))
                scores[EvaluationMetric.COST_EFFICIENCY.value] = min(1.0, max(0.0, 1.0 - (cost / 0.01)))
                
                result = EvaluationResult(
                    test_case_id=test_case.id,
                    prompt_version=prompt[:50] + "...",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    actual_output=actual_output,
                    response_time=response_time,
                    token_count=token_count,
                    cost=cost,
                    scores=scores,
                    timestamp=datetime.now()
                )
                
                results.append(result)
                
            except Exception as e:
                # Handle errors gracefully
                error_result = EvaluationResult(
                    test_case_id=test_case.id,
                    prompt_version=prompt[:50] + "...",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    actual_output=f"Error: {str(e)}",
                    response_time=0.0,
                    token_count=0,
                    cost=0.0,
                    scores={metric.value: 0.0 for metric in EvaluationMetric},
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    def _calculate_cost(self, model: str, token_count: int) -> float:
        """Calculate approximate API cost"""
        # Simplified cost calculation (actual costs vary)
        cost_per_1k_tokens = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01
        }
        
        rate = cost_per_1k_tokens.get(model, 0.002)
        return (token_count / 1000) * rate
    
    def _generate_optimization_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate optimization insights and recommendations"""
        if not all_results:
            return {}
        
        # Best performing configurations
        sorted_results = sorted(all_results, key=lambda x: x["average_score"], reverse=True)
        
        # Analyze patterns
        model_performance = {}
        temperature_performance = {}
        
        for result in all_results:
            model = result["model"]
            temp = result["temperature"]
            score = result["average_score"]
            
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(score)
            
            if temp not in temperature_performance:
                temperature_performance[temp] = []
            temperature_performance[temp].append(score)
        
        # Calculate averages
        model_avg = {model: np.mean(scores) for model, scores in model_performance.items()}
        temp_avg = {temp: np.mean(scores) for temp, scores in temperature_performance.items()}
        
        summary = {
            "best_configuration": sorted_results[0] if sorted_results else None,
            "worst_configuration": sorted_results[-1] if sorted_results else None,
            "model_rankings": sorted(model_avg.items(), key=lambda x: x[1], reverse=True),
            "optimal_temperature": max(temp_avg.items(), key=lambda x: x[1]) if temp_avg else None,
            "total_configurations_tested": len(all_results),
            "score_improvement": sorted_results[0]["average_score"] - sorted_results[-1]["average_score"] if len(sorted_results) > 1 else 0,
            "recommendations": self._generate_recommendations(sorted_results, model_avg, temp_avg)
        }
        
        return summary
    
    def _generate_recommendations(self, sorted_results: List[Dict], 
                                model_avg: Dict, temp_avg: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if sorted_results:
            best = sorted_results[0]
            recommendations.append(f"Use {best['model']} for best overall performance")
            recommendations.append(f"Optimal temperature setting: {best['temperature']}")
            recommendations.append(f"Recommended max_tokens: {best['max_tokens']}")
        
        if model_avg:
            best_model = max(model_avg.items(), key=lambda x: x[1])
            recommendations.append(f"Best performing model: {best_model[0]} (avg score: {best_model[1]:.3f})")
        
        if temp_avg:
            best_temp = max(temp_avg.items(), key=lambda x: x[1])
            recommendations.append(f"Optimal temperature range: {best_temp[0]} (avg score: {best_temp[1]:.3f})")
        
        return recommendations

# Global instances
evaluator = AIEvaluator()
optimizer = AIOptimizer(evaluator)

def format_optimization_results(results: Dict[str, Any]) -> str:
    """Format optimization results for display"""
    if not results:
        return "No optimization results available."
    
    best_config = results.get("best_configuration", {})
    summary = results.get("optimization_summary", {})
    
    output = f"""
# 🎯 AI Optimization Results

## 🏆 Best Configuration Found
- **Model**: {best_config.get('model', 'N/A')}
- **Temperature**: {best_config.get('temperature', 'N/A')}
- **Max Tokens**: {best_config.get('max_tokens', 'N/A')}
- **Average Score**: {best_config.get('average_score', 0):.3f}

---

## 📊 Optimization Summary
- **Configurations Tested**: {summary.get('total_configurations_tested', 0)}
- **Score Improvement**: {summary.get('score_improvement', 0):.3f}
- **Optimal Temperature**: {summary.get('optimal_temperature', ['N/A', 0])[0]}

---

## 🎖️ Model Performance Rankings
"""
    
    model_rankings = summary.get('model_rankings', [])
    for i, (model, score) in enumerate(model_rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        output += f"{medal} **{model}**: {score:.3f}\n"
    
    output += "\n---\n\n## 💡 Recommendations\n"
    recommendations = summary.get('recommendations', [])
    for rec in recommendations:
        output += f"• {rec}\n"
    
    # Detailed results table
    output += "\n---\n\n## 📋 Detailed Test Results\n\n"
    output += "| Configuration | Model | Temp | Tokens | Avg Score | Performance |\n"
    output += "|---------------|-------|------|--------|-----------|-------------|\n"
    
    all_results = results.get('all_results', [])[:10]  # Show top 10
    for i, result in enumerate(all_results, 1):
        performance = "🟢 Excellent" if result['average_score'] > 0.8 else "🟡 Good" if result['average_score'] > 0.6 else "🔴 Needs Work"
        output += f"| Config {i} | {result['model']} | {result['temperature']} | {result['max_tokens']} | {result['average_score']:.3f} | {performance} |\n"
    
    return output

def format_evaluation_metrics(results: List[EvaluationResult]) -> str:
    """Format detailed evaluation metrics"""
    if not results:
        return "No evaluation results available."
    
    output = """
# 📈 Evaluation Metrics Dashboard

## 📊 Performance Overview
"""
    
    # Calculate averages
    all_scores = {}
    for result in results:
        for metric, score in result.scores.items():
            if metric not in all_scores:
                all_scores[metric] = []
            all_scores[metric].append(score)
    
    avg_scores = {metric: np.mean(scores) for metric, scores in all_scores.items()}
    
    # Performance indicators
    for metric, avg_score in avg_scores.items():
        status = "🟢 Excellent" if avg_score > 0.8 else "🟡 Good" if avg_score > 0.6 else "🔴 Needs Improvement"
        bar_length = int(avg_score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        output += f"**{metric.replace('_', ' ').title()}**: {avg_score:.3f} {status}\n"
        output += f"`{bar}` {avg_score:.1%}\n\n"
    
    # Recent test results
    output += "## 🔍 Recent Test Results\n\n"
    output += "| Test | Model | Score | Response Time | Cost |\n"
    output += "|------|-------|-------|---------------|------|\n"
    
    for result in results[-5:]:  # Show last 5 results
        avg_score = np.mean(list(result.scores.values()))
        output += f"| {result.test_case_id[:10]}... | {result.model} | {avg_score:.3f} | {result.response_time:.2f}s | ${result.cost:.4f} |\n"
    
    return output

def run_evaluation_test(input_text: str, expected_output: str, category: str):
    """Run a single evaluation test"""
    if not input_text.strip() or not expected_output.strip():
        return "Please provide both input text and expected output."
    
    try:
        # Add test case
        test_id = evaluator.add_test_case(input_text, expected_output, category)
        
        # Run evaluation with default settings
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Provide accurate and relevant responses."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        response_time = time.time() - start_time
        actual_output = response.choices[0].message.content.strip()
        
        # Evaluate response
        scores = evaluator.evaluate_response(actual_output, expected_output, input_text)
        
        # Create result
        result = EvaluationResult(
            test_case_id=test_id,
            prompt_version="default",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200,
            actual_output=actual_output,
            response_time=response_time,
            token_count=response.usage.total_tokens,
            cost=evaluator._calculate_cost("gpt-3.5-turbo", response.usage.total_tokens),
            scores=scores,
            timestamp=datetime.now()
        )
        
        evaluator.evaluation_results.append(result)
        
        # Format result
        avg_score = np.mean(list(scores.values()))
        output = f"""
# 🧪 Evaluation Test Result

## 📋 Test Details
- **Test ID**: {test_id}
- **Category**: {category}
- **Average Score**: {avg_score:.3f}

## 📊 Detailed Scores
"""
        for metric, score in scores.items():
            status = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
            output += f"- **{metric.replace('_', ' ').title()}**: {score:.3f} {status}\n"
        
        output += f"""
## 💬 Response Comparison
**Expected**: {expected_output}

**Actual**: {actual_output}

## ⚡ Performance
- **Response Time**: {response_time:.2f} seconds
- **Token Usage**: {response.usage.total_tokens} tokens
- **Estimated Cost**: ${result.cost:.4f}
"""
        
        return output
        
    except Exception as e:
        return f"Error during evaluation: {str(e)}"

def run_optimization(base_prompt: str, optimization_focus: str):
    """Run optimization process"""
    if not base_prompt.strip():
        return "Please provide a base prompt to optimize."
    
    if not evaluator.test_cases:
        return "Please run some evaluation tests first to create test cases for optimization."
    
    try:
        # Define optimization configurations based on focus
        if optimization_focus == "Speed & Cost":
            config = OptimizationConfig(
                models=["gpt-3.5-turbo"],
                temperatures=[0.3, 0.7],
                max_tokens_options=[100, 150],
                prompt_variations=[
                    base_prompt,
                    base_prompt + " Be concise.",
                    "You are an efficient AI assistant. " + base_prompt
                ]
            )
        elif optimization_focus == "Quality & Accuracy":
            config = OptimizationConfig(
                models=["gpt-3.5-turbo", "gpt-4"],
                temperatures=[0.1, 0.3, 0.5],
                max_tokens_options=[200, 300],
                prompt_variations=[
                    base_prompt,
                    "You are an expert AI assistant. " + base_prompt + " Provide detailed and accurate responses.",
                    base_prompt + " Think step by step and provide comprehensive analysis."
                ]
            )
        else:  # Balanced
            config = OptimizationConfig(
                models=["gpt-3.5-turbo"],
                temperatures=[0.3, 0.5, 0.7],
                max_tokens_options=[150, 200, 250],
                prompt_variations=[
                    base_prompt,
                    "You are a helpful AI assistant. " + base_prompt,
                    base_prompt + " Provide clear and relevant responses."
                ]
            )
        
        # Run optimization
        results = optimizer.optimize_prompt(base_prompt, evaluator.test_cases[:3], config)  # Use first 3 test cases
        
        return format_optimization_results(results)
        
    except Exception as e:
        return f"Error during optimization: {str(e)}"

# Sample test cases for demonstration
SAMPLE_TEST_CASES = {
    "Document Summary": {
        "input": "Artificial intelligence is transforming healthcare through improved diagnostic accuracy, personalized treatment plans, and automated administrative tasks. Machine learning algorithms can analyze medical images, predict patient outcomes, and optimize hospital operations. However, challenges include data privacy, algorithm bias, and the need for regulatory compliance.",
        "expected": "AI is revolutionizing healthcare by enhancing diagnostics, personalizing treatments, and automating administration. While ML improves medical imaging and predictions, challenges remain around privacy, bias, and regulation."
    },
    "Sentiment Analysis": {
        "input": "I'm absolutely thrilled with the new software update! The interface is much more intuitive and the performance improvements are incredible. This has definitely exceeded my expectations.",
        "expected": "Positive sentiment. The user expresses enthusiasm and satisfaction with software improvements, particularly praising the interface and performance enhancements."
    },
    "Question Answering": {
        "input": "What are the key benefits of cloud computing for small businesses?",
        "expected": "Key benefits include cost savings (no hardware investment), scalability (adjust resources as needed), accessibility (work from anywhere), automatic updates, improved security, and disaster recovery capabilities."
    }
}

# Create Gradio interface
with gr.Blocks(title="AI Evaluator-Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 AI Evaluator-Optimizer System
    
    Systematically evaluate AI model performance and optimize prompts and parameters for better results through comprehensive testing and analysis.
    """)
    
    gr.Markdown("""
    ## 🔄 Evaluation-Optimization Workflow:
    
    **🧪 Evaluation Phase:**
    1. **Test Case Creation**: Define input-output pairs for testing
    2. **Multi-Metric Assessment**: Accuracy, relevance, completeness, clarity
    3. **Performance Tracking**: Response time, cost, token usage
    
    **🎯 Optimization Phase:**
    1. **Systematic Testing**: Multiple prompt variations and parameters
    2. **Configuration Comparison**: Models, temperatures, token limits
    3. **Best Practice Identification**: Data-driven recommendations
    """)
    
    with gr.Tabs():
        # Tab 1: Evaluation Testing
        with gr.TabItem("🧪 Evaluation Testing"):
            gr.Markdown("**Test AI responses against expected outputs with multiple evaluation metrics**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    eval_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter the input prompt/question...",
                        lines=4,
                        value=SAMPLE_TEST_CASES["Document Summary"]["input"]
                    )
                    
                    eval_expected = gr.Textbox(
                        label="🎯 Expected Output",
                        placeholder="Enter the expected response...",
                        lines=4,
                        value=SAMPLE_TEST_CASES["Document Summary"]["expected"]
                    )
                    
                    eval_category = gr.Dropdown(
                        choices=["Document Summary", "Sentiment Analysis", "Question Answering", "Translation", "Classification", "Other"],
                        label="📂 Test Category",
                        value="Document Summary"
                    )
                    
                    sample_test_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_TEST_CASES.keys()),
                        label="📚 Sample Test Cases",
                        value="Document Summary"
                    )
                    
                    eval_btn = gr.Button("🧪 Run Evaluation Test", variant="primary")
                    
                with gr.Column(scale=2):
                    eval_output = gr.Markdown(label="📊 Evaluation Results")
            
            # Load sample test case
            def load_sample_test(sample_key):
                if sample_key in SAMPLE_TEST_CASES:
                    return (
                        SAMPLE_TEST_CASES[sample_key]["input"],
                        SAMPLE_TEST_CASES[sample_key]["expected"],
                        sample_key
                    )
                return "", "", sample_key
            
            sample_test_dropdown.change(
                fn=load_sample_test,
                inputs=sample_test_dropdown,
                outputs=[eval_input, eval_expected, eval_category]
            )
            
            eval_btn.click(
                fn=run_evaluation_test,
                inputs=[eval_input, eval_expected, eval_category],
                outputs=eval_output
            )
        
        # Tab 2: Optimization
        with gr.TabItem("🎯 Optimization"):
            gr.Markdown("**Optimize prompts and parameters based on evaluation results**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    opt_prompt = gr.Textbox(
                        label="📝 Base Prompt",
                        placeholder="Enter the prompt to optimize...",
                        lines=6,
                        value="You are a helpful AI assistant. Provide accurate and relevant responses."
                    )
                    
                    opt_focus = gr.Radio(
                        choices=["Speed & Cost", "Quality & Accuracy", "Balanced"],
                        label="🎯 Optimization Focus",
                        value="Balanced"
                    )
                    
                    opt_btn = gr.Button("🚀 Start Optimization", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    opt_output = gr.Markdown(label="🎯 Optimization Results")
            
            opt_btn.click(
                fn=run_optimization,
                inputs=[opt_prompt, opt_focus],
                outputs=opt_output
            )
        
        # Tab 3: Analytics Dashboard
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.Markdown("**Monitor evaluation metrics and optimization history**")
            
            with gr.Row():
                with gr.Column():
                    analytics_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")
                    
                with gr.Column():
                    gr.Markdown("*Click refresh to update analytics dashboard*")
            
            with gr.Row():
                analytics_output = gr.Markdown(label="📈 Performance Analytics")
            
            def refresh_analytics():
                if evaluator.evaluation_results:
                    return format_evaluation_metrics(evaluator.evaluation_results)
                else:
                    return "No evaluation data available. Run some tests first!"
            
            analytics_btn.click(
                fn=refresh_analytics,
                outputs=analytics_output
            )
            
            # Initial load
            demo.load(
                fn=refresh_analytics,
                outputs=analytics_output
            )
    
    gr.Markdown("""
    ### 🎯 Evaluator-Optimizer Benefits:
    
    **📊 Systematic Evaluation:**
    - **Multi-Metric Assessment**: Accuracy, relevance, completeness, clarity
    - **Performance Tracking**: Response time, cost efficiency, token usage
    - **Comparative Analysis**: Test different configurations objectively
    - **Quality Assurance**: Consistent evaluation standards
    
    **🎯 Intelligent Optimization:**
    - **Data-Driven Decisions**: Optimize based on actual performance data
    - **Parameter Tuning**: Find optimal temperature, token limits, models
    - **Prompt Engineering**: Systematic prompt variation testing
    - **Cost Optimization**: Balance quality with efficiency
    
    **💼 Business Applications:**
    - **🎖️ Veterans Affairs**: Optimize claim processing accuracy and efficiency
    - **📞 Customer Service**: Improve response quality and satisfaction scores
    - **📚 Content Generation**: Enhance content quality and relevance
    - **🏥 Healthcare**: Optimize medical AI for accuracy and safety
    - **📊 Business Intelligence**: Improve analytical AI performance
    
    **🔬 Research & Development:**
    - **A/B Testing**: Compare different AI configurations scientifically
    - **Performance Baselines**: Establish quality benchmarks
    - **Continuous Improvement**: Iterative optimization cycles
    - **ROI Analysis**: Measure improvement vs. investment
    
    This evaluation-optimization system ensures your AI implementations deliver optimal performance for your specific use cases!
    """)

if __name__ == "__main__":
    print("=== AI Evaluator-Optimizer Demo ===\n")
    print("🎯 Systematic AI evaluation and optimization system...")
    print("Evaluate performance and optimize for better results\n")
    
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7864)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("\nTrying to start interface anyway...")
        demo.launch(share=True)