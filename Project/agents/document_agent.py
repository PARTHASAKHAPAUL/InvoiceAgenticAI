
"""Document Agent for Invoice Processing"""

# TODO: Implement agent

import os
import json
import re
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, InvoiceData, ItemDetail,
    ProcessingStatus, ValidationStatus
)
from utils.logger import StructuredLogger


load_dotenv()
logger = StructuredLogger("DocumentAgent")

def safe_json_parse(result_text: str):
    # Remove Markdown formatting if present
    cleaned = re.sub(r"^```[a-zA-Z]*\n|```$", "", result_text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback if the AI wrapped JSON in text
        start, end = cleaned.find("{"), cleaned.rfind("}") + 1
        if start >= 0 and end > 0:
            return json.loads(cleaned[start:end])
        raise

def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', '').replace('$', '').strip())
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def parse_date_safe(date_str):
    if not date_str:
        return None
    for fmt in ("%b %d %Y", "%b %d, %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


from collections import defaultdict
class APIKeyBalancer:
    SAVE_FILE = "key_stats.json"
    def __init__(self, keys):
        self.keys = keys
        self.usage = defaultdict(int)
        self.errors = defaultdict(int)
        self.load()

    def load(self):
        if os.path.exists(self.SAVE_FILE):
            data = json.load(open(self.SAVE_FILE))
            self.usage.update(data.get("usage", {}))
            self.errors.update(data.get("errors", {}))

    def save(self):
        json.dump({
            "usage": self.usage,
            "errors": self.errors
        }, open(self.SAVE_FILE, "w"))

    def get_best_key(self):
        # choose least used or least errored key
        best_key = min(self.keys, key=lambda k: (self.errors[k], self.usage[k]))
        self.usage[best_key] += 1
        self.save()
        return best_key

    def report_error(self, key):
        self.errors[key] += 1
        self.save()

        
balancer = APIKeyBalancer([
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    # os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    # os.getenv("GEMINI_API_KEY_7"),
])


class DocumentAgent(BaseAgent):
    """Agent responsible for document processing and invoice data extraction"""

    def __init__(self, config: Dict[str, Any] = None):
        # pass
        super().__init__("document_agent", config)
        self.logger = StructuredLogger("DocumentAgent")
        self.api_key = balancer.get_best_key()
        print("self.api_key..........", self.api_key)

        genai.configure(api_key=self.api_key)
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY_7"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt):
        try:
            print("generate called")
            response = self.model.generate_content(prompt)
            print("response....", response)
            return response
        except Exception as e:
            print("errrororrrooroor")
            balancer.report_error(self.api_key)
            print(balancer.keys)
            print(balancer.usage)
            print(balancer.errors)
            raise

    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        # pass
        if not state.file_name or not os.path.exists(state.file_name):
            self.logger.logger.error(f"[Document Agent] Missing or invalid file: {state.file_name}")
            return False
        return True

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        return bool(state.invoice_data and state.invoice_data.total > 0)

    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        # pass
        # file_name = state.file_name
        self.logger.logger.info(f"Executing Document Agent for file: {state.file_name}")

        if not self._validate_preconditions(state, workflow_type):
            state.overall_status = ProcessingStatus.FAILED
            self._log_decision(state, "Extraction Failed", "Preconditions not met", confidence=0.0)
            
        try:
            raw_text = await self._extract_text_from_pdf(state.file_name)
            invoice_data = await self._parse_invoice_with_ai(raw_text)
            invoice_data = await self._enhance_invoice_data(invoice_data, raw_text)
            invoice_data.file_name = state.file_name
            state.invoice_data = invoice_data
            state.overall_status = ProcessingStatus.IN_PROGRESS
            state.current_agent = self.agent_name
            state.updated_at = datetime.utcnow()

            confidence = self._calculate_extraction_confidence(invoice_data, raw_text)
            state.invoice_data.extraction_confidence = confidence
            self._log_decision(
                state,
                "Extraction Successful",
                "PDF text successfully extracted and parsed by AI",
                confidence,
                state.process_id
            )
            return state
        except Exception as e:
            self.logger.logger.exception(f"[Document Agent] Extraction failed: {e}")
            state.overall_status = ProcessingStatus.FAILED
            self._should_escalate(state, reason=str(e))
            return state


    async def _extract_text_from_pdf(self, file_name: str) -> str:
        # pass
        text = ""
        try:
            self.logger.logger.info("[DocumentAgent] Extracting text using PyMuPDF...")
            with fitz.open(file_name) as doc:
                for page in doc:
                    text += page.get_text()
            if len(text.strip()) < 5:
                raise ValueError("PyMuPDF extraction too short, switching to PDFPlumber")
        except Exception as e:
            self.logger.logger.info("[DocumentAgent] Fallback to PDFPlumber...")
            try:
                with pdfplumber.open(file_name) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception as e2:
                self.logger.logger.error("[DocumentAgent] PDFPlumber failed :{e2}")
                text = ""
        return text

    async def _parse_invoice_with_ai(self, text: str) -> InvoiceData:
        # pass
        self.logger.logger.info("[DocumentAgent] Parsing invoice data using Gemini AI...")
        print("text-----------", text)
        prompt = f"""
        Extract structured invoice information as JSON with fields:
        invoice_number, order_id, customer_name, due_date, ship_to, ship_mode,
        subtotal, discount, shipping_cost, total, and item_details (item_name, quantity, rate, amount).

        Important Note: If an item description continues on multiple lines, combine them into one item_name. Check intelligently
        that if at all there will be more than one item then it should have more numbers.
        So extract by verifying that is there only one item or more than one.

        Input Text:
        {text[:8000]}
        """
        response = self.generate(prompt)
        result_text = response.text.strip()
        data = safe_json_parse(result_text)
        print("----------------------------------text-----------------------------------",text)
        print("result text::::::::::::::::::::::::::::",data)
        # try:
        #     data = json.loads(result_text)
        # except Exception as e:
        #     self.logger.logger.warning("AI output not valid JSON, retrying with fallback parse.")
        #     data = json.loads(result_text[result_text.find('{'): result_text.rfind('}')+1])
        items = []
        for item in data.get("item_details", []):
            items.append(ItemDetail(
                item_name=item.get("item_name"),
                quantity=float(item.get("quantity", 1)),
                rate=to_float(item.get("rate", 0.0)),
                amount=to_float(item.get("amount", 0.0)),
                # category=self._categorize_item(item.get("item_name", "Unknown")),
            ))

        invoice_data = InvoiceData(
            invoice_number=data.get("invoice_number"),
            order_id=data.get("order_id"),
            customer_name=data.get("customer_name"),
            due_date=parse_date_safe(data.get("due_date")),
            ship_to=data.get("ship_to"),
            ship_mode=data.get("ship_mode"),
            subtotal=to_float(data.get("subtotal", 0.0)),
            discount=to_float(data.get("discount", 0.0)),
            shipping_cost=to_float(data.get("shipping_cost", 0.0)),
            total=to_float(data.get("total", 0.0)),
            item_details=items,
            raw_text=text,
        )
        confidence = self._calculate_extraction_confidence(invoice_data, text)
        invoice_data.extraction_confidence = confidence
        self.logger.logger.info("AI output successfully parsed into JSON format")
        return invoice_data


    async def _enhance_invoice_data(self, invoice_data: InvoiceData, raw_text: str) -> InvoiceData:
        # pass
        if not invoice_data.customer_name:
            if "Invoice To" in raw_text:
                lines = raw_text.split("\n")
                for i, line in enumerate(lines):
                    if "Invoice To" in line:
                        invoice_data.customer_name = lines[i+1].strip()
                        break
        return invoice_data

    def _categorize_item(self, item_name: str) -> str:
        # pass
        name = item_name.lower()
        prompt = f"""
        Extract the category of the Item from the item details very intelligently
        so that we can get the category in which the item belongs to very efficiently:
        Example: "Electronics", "Furniture", "Software", etc.....
        Input Text- The item is given below (provide the category in JSON format like -- category: 'extracted category') ---->
        {name}
        """
        response = self.generate(prompt)
        result_text = response.text.strip()
        category = safe_json_parse(result_text)
        print(category['category'])
        return category['category']

    def _calculate_extraction_confidence(self, invoice_data: InvoiceData, raw_text: str) -> float:
        """
        Intelligent confidence scoring for extracted invoice data.
        Combines presence, consistency, and numeric sanity checks.
        """
        score = 0.0
        weight = {
            "invoice_number": 0.1,
            "order_id": 0.05,
            "customer_name": 0.1,
            "due_date": 0.05,
            "ship_to": 0.05,
            "item_details": 0.25,
            "total_consistency": 0.25,
            "currency_detected": 0.05,
            "text_match_bonus": 0.1
        }
    
        text_lower = raw_text.lower()
    
        # Presence-based confidence
        if invoice_data.invoice_number:
            score += weight["invoice_number"]
        if invoice_data.order_id:
            score += weight["order_id"]
        if invoice_data.customer_name:
            score += weight["customer_name"]
        if invoice_data.due_date and "due_date" in text_lower:
            score += weight["due_date"]
        if not invoice_data.due_date and "due_date" not in text_lower:
            score += weight["due_date"]
        if invoice_data.item_details:
            score += weight["item_details"]
    
        # Currency detection
        if any(c in raw_text for c in ["$", "₹", "€", "usd", "inr", "eur"]):
            score += weight["currency_detected"]
    
        # Numeric Consistency: subtotal + shipping ≈ total 
        def _extract_amounts(pattern):
            import re
            matches = re.findall(pattern, raw_text)
            return [float(m.replace(",", "").replace("$", "").strip()) for m in matches if m]
    
        import re
        numbers = _extract_amounts(r"\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
        if len(numbers) >= 3 and invoice_data.total:
            approx_total = max(numbers)
            diff = abs(approx_total - invoice_data.total)
            if diff < 5:  # minor difference allowed
                score += weight["total_consistency"]
            elif diff < 50:
                score += weight["total_consistency"] * 0.5
    
        # Textual verification 
        hits = 0
        for field in [invoice_data.customer_name, invoice_data.order_id, invoice_data.invoice_number]:
            if field and str(field).lower() in text_lower:
                hits += 1
        if hits >= 2:
            score += weight["text_match_bonus"]
    
        # Penalty for empty critical fields 
        missing_critical = not invoice_data.total or not invoice_data.customer_name or not invoice_data.invoice_number
        if missing_critical:
            score *= 0.8
    
        # Clamp and finalize 
        final_conf = round(min(score, 0.99), 2)
        invoice_data.extraction_confidence = final_conf
        return final_conf * 100.0


    async def health_check(self) -> Dict[str, Any]:
        """
        Perform intelligent health diagnostics for the Document Agent.
        Collects operational, performance, and API connectivity metrics.
        """
        from datetime import datetime

        metrics_data = {}
        executions = 0
        success_rate = 0.0
        avg_duration = 0.0
        failures = 0
        last_run = None
        # latency_trend = None

        # 1. Try to get live metrics from state
        print("(self.state)-------",self.metrics)
        # print("self.state.agent_metrics-------", self.state.agent_metrics)
        if self.metrics:
            executions = self.metrics["processed"]
            avg_duration = self.metrics["avg_latency_ms"]
            failures = self.metrics["errors"]
            last_run = self.metrics["last_run_at"]
            success_rate = (executions - failures) / (executions+1e-8)

            # print(executions, avg_duration, failures, last_run, success_rate)
            # latency_trend = getattr(m, "total_duration_ms", None)

        # 2. API connectivity check
        gemini_ok = bool(self.api_key)
        # print("self.api---", self.api_key)
        # print("geminiokkkkkk", gemini_ok)
        api_status = "🟢 Active" if gemini_ok else "🔴 Missing Key"

        # 3. Health logic
        overall_status = "🟢 Healthy"
        if not gemini_ok or failures > 3:
            overall_status = "🟠 Degraded"
        if executions > 0 and success_rate < 0.5:
            overall_status = "🔴 Unhealthy"

        # 4. Extended agent diagnostics
        metrics_data = {
            "Agent": "Document Agent 🧾",
            "Executions": executions,
            "Success Rate (%)": round(success_rate * 100, 2),
            "Avg Duration (ms)": round(avg_duration, 2),
            "Total Failures": failures,
            "API Status": api_status,
            "Last Run": str(last_run) if last_run else "Not applicable",
            "Overall Health": overall_status,
            # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        self.logger.logger.info(f"[HealthCheck] Document Agent metrics: {metrics_data}")
        return metrics_data
