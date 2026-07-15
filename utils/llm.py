import re
import requests
import json
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from google import genai

class LLMManager:
    """
    Manages connections to LLM providers (Ollama, Gemini, Groq, OpenAI, Hugging Face)
    and handles NL-to-SQL conversion, query correction, and insight generation.
    """
    def __init__(self, provider: str = "Ollama", api_key: str = "", model_name: str = "", host: str = "http://localhost:11434"):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.host = host # Used for Ollama (e.g. http://localhost:11434)
        
        # Set default models if not provided
        if not self.model_name:
            if self.provider == "Ollama":
                self.model_name = "llama3.2:latest"
            elif self.provider == "Gemini":
                self.model_name = "gemini-2.5-flash"
            elif self.provider == "Groq":
                self.model_name = "llama-3.3-70b-versatile"
            elif self.provider == "OpenAI":
                self.model_name = "gpt-4o-mini"
            elif self.provider == "Hugging Face":
                self.model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    def _call_llm(self, prompt: str, system_instruction: str = "") -> str:
        """
        Generic private method to call the configured LLM provider.
        """
        try:
            if self.provider == "Ollama":
                url = f"{self.host.rstrip('/')}/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_instruction,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json().get("response", "").strip()
                
            elif self.provider == "Gemini":
                # Check for newer API client
                client = genai.Client(api_key=self.api_key)
                # Prepare combined text for instruction and content
                contents = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return response.text.strip()
                
            elif self.provider == "Groq":
                client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "OpenAI":
                client = OpenAI(api_key=self.api_key)
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "Hugging Face":
                url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                combined_prompt = f"{system_instruction}\n\nUser: {prompt}\nAssistant:" if system_instruction else prompt
                payload = {
                    "inputs": combined_prompt,
                    "parameters": {"max_new_tokens": 512, "temperature": 0.1, "return_full_text": False}
                }
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                res_data = response.json()
                if isinstance(res_data, list) and len(res_data) > 0:
                    return res_data[0].get("generated_text", "").strip()
                elif isinstance(res_data, dict):
                    return res_data.get("generated_text", "").strip()
                return str(res_data)
                
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            return f"ERROR_CALLING_LLM: {str(e)}"

    def generate_sql(self, schema_text: str, user_question: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Translates a natural language question into a SQLite query based on table schema.
        Returns a tuple of (sql_query, explanation, chart_spec).
        """
        system_instruction = (
            "You are an expert SQL Translator for a Business Intelligence tool.\n"
            "Your task is to convert a user's natural language question into a standard SELECT SQLite query.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a valid SELECT query. DO NOT modify, delete, drop, or create tables.\n"
            "2. Use SQLite syntax. For example, for date extracts use strftime('%m', col) or strftime('%Y', col).\n"
            "3. Reference only columns that exist in the provided schema. Do not invent columns.\n"
            "4. Ensure correct capitalization of column names exactly matching the schema.\n"
            "5. To match text case-insensitively, use LIKE '%value%'.\n"
            "6. Always limit the results to a maximum of 500 rows unless the user explicitly requests more.\n"
            "7. Return the query inside a single ```sql block.\n"
            "8. Also recommend a chart type (bar, line, pie, scatter, or none) and explain what the query does."
        )
        
        prompt = (
            f"=== DATABASE SCHEMA ===\n"
            f"{schema_text}\n\n"
            f"=== USER QUESTION ===\n"
            f"{user_question}\n\n"
            f"=== RESPONSE FORMAT ===\n"
            f"Provide your answer in this exact format:\n"
            f"```sql\n"
            f"[SQL QUERY HERE]\n"
            f"```\n"
            f"CHART: type=[bar|line|pie|scatter|none] x=[column_name] y=[column_name]\n"
            f"EXPLANATION: [one sentence description of what the query calculates]"
        )
        
        raw_response = self._call_llm(prompt, system_instruction)
        if "ERROR_CALLING_LLM" in raw_response:
            return "", raw_response, {"type": "none"}
            
        return self._parse_sql_response(raw_response)

    def correct_sql(self, schema_text: str, faulty_query: str, error_msg: str) -> Tuple[str, str]:
        """
        Self-corrects a SQL query that failed execution.
        """
        system_instruction = (
            "You are an expert SQL debugger. A SQL query you generated failed execution against SQLite.\n"
            "Analyze the error message and the schema carefully, and output a corrected SQLite query."
        )
        
        prompt = (
            f"=== DATABASE SCHEMA ===\n"
            f"{schema_text}\n\n"
            f"=== FAULTY SQL QUERY ===\n"
            f"```sql\n"
            f"{faulty_query}\n"
            f"```\n\n"
            f"=== SQLITE ERROR ===\n"
            f"{error_msg}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Identify the syntax mistake, missing column, or bad function call, and output ONLY the corrected SQL query inside a single ```sql code block.\n"
            f"Do not write explanations, just the code block."
        )
        
        raw_response = self._call_llm(prompt, system_instruction)
        if "ERROR_CALLING_LLM" in raw_response:
            return "", raw_response
            
        sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip(), "Corrected query generated after self-check."
        return raw_response.strip(), "Raw response from LLM (failed to parse markdown block)."

    def generate_insights(self, user_question: str, sql_query: str, results_df_summary: str) -> str:
        """
        Generates business insights and explanations based on the user's question,
        the SQL query executed, and the summarized query results.
        """
        system_instruction = (
            "You are a Senior Business Intelligence Analyst. Your goal is to explain SQL results\n"
            "to business users in simple, conversational English. Point out interesting trends,\n"
            "anomalies, or key drivers, and provide actionable business recommendations."
        )
        
        prompt = (
            f"=== BUSINESS PROBLEM ===\n"
            f"User asked: '{user_question}'\n\n"
            f"=== SQL QUERY RUN ===\n"
            f"{sql_query}\n\n"
            f"=== QUERY RESULTS ===\n"
            f"{results_df_summary}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Summarize the findings and answer the user's question directly. Structure your response with:\n"
            f"1. **Summary**: A direct, simple answer to the user's question.\n"
            f"2. **Key Insights**: 2-3 bullet points outlining important trends, exceptions, or high/low points.\n"
            f"3. **Business Action**: A practical suggestion or strategy the business should take based on this data.\n"
            f"Keep your tone professional, concise, and executive-ready."
        )
        
        return self._call_llm(prompt, system_instruction)

    def explain_trend_or_forecast(self, metric_name: str, forecast_summary: str) -> str:
        """
        Generates an explanation of a forecasting trend.
        """
        system_instruction = "You are a forecasting expert. Explain time-series forecasts to business users."
        prompt = (
            f"We ran a time-series forecast on the metric: '{metric_name}'.\n"
            f"Here is a summary of the forecast outputs:\n"
            f"{forecast_summary}\n\n"
            f"Explain the future trend (is it growing, declining, seasonal, or flat?).\n"
            f"What should the business prepare for given this projection? Keep it under 150 words."
        )
        return self._call_llm(prompt, system_instruction)

    def explain_anomalies(self, anomalies_summary: str) -> str:
        """
        Explains why the flagged anomalies are unusual and what could be the potential cause.
        """
        system_instruction = "You are a risk management and audit analyst. Interpret anomalies in business data."
        prompt = (
            f"We ran anomaly detection on our dataset and flagged the following anomalous records:\n"
            f"{anomalies_summary}\n\n"
            f"Analyze these anomalies. Why are they flagged? (e.g., extremely high value, abnormal combination).\n"
            f"What should the operations or data team check to verify if these are data errors or genuine exceptions?"
        )
        return self._call_llm(prompt, system_instruction)

    def generate_dataset_summary(self, schema_text: str) -> str:
        """
        Generates an initial high-level analytical summary of the newly uploaded dataset.
        """
        system_instruction = "You are a Chief Data Officer. Welcome the user and describe their dataset."
        prompt = (
            f"Based on the following schema and sample rows, describe this dataset in 2-3 sentences:\n"
            f"{schema_text}\n\n"
            f"Identify 3 interesting business questions the user could ask this assistant about the data."
        )
        return self._call_llm(prompt, system_instruction)

    def _parse_sql_response(self, text: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parses the structured response from the LLM.
        """
        sql_query = ""
        explanation = ""
        chart_spec = {"type": "none", "x": "", "y": ""}
        
        # Extract SQL block
        sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Fallback if markdown block is missing
            lines = text.split("\n")
            sql_lines = []
            for line in lines:
                if line.strip().upper().startswith(("SELECT", "WITH")):
                    sql_lines.append(line)
                elif sql_lines and not line.strip():
                    break
                elif sql_lines:
                    sql_lines.append(line)
            sql_query = "\n".join(sql_lines).strip()
            
        # Clean up query trailing semicolons
        sql_query = sql_query.rstrip(";")
        
        # Extract explanation
        exp_match = re.search(r"EXPLANATION:\s*(.*)", text, re.IGNORECASE)
        if exp_match:
            explanation = exp_match.group(1).strip()
            
        # Extract chart spec
        chart_match = re.search(r"CHART:\s*type=(\w+)\s*x=(\w+)\s*y=(\w+)", text, re.IGNORECASE)
        if chart_match:
            chart_spec = {
                "type": chart_match.group(1).lower().strip(),
                "x": chart_match.group(2).strip(),
                "y": chart_match.group(3).strip()
            }
        else:
            # Try alternate key-value matching
            type_match = re.search(r"type=(\w+)", text, re.IGNORECASE)
            x_match = re.search(r"x=(\w+)", text, re.IGNORECASE)
            y_match = re.search(r"y=(\w+)", text, re.IGNORECASE)
            if type_match:
                chart_spec["type"] = type_match.group(1).lower().strip()
            if x_match:
                chart_spec["x"] = x_match.group(1).strip()
            if y_match:
                chart_spec["y"] = y_match.group(1).strip()
                
        return sql_query, explanation, chart_spec
