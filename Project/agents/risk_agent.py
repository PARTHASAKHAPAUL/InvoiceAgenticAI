
"""Risk Assessment Agent for Invoice Processing"""

# TODO: Implement agent

import os
import json
import re
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
from statistics import mean
import time
from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, RiskAssessment, RiskLevel,
    ValidationStatus, ProcessingStatus
)
from utils.logger import StructuredLogger

load_dotenv()

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

class RiskAgent(BaseAgent):
    """Agent responsible for risk assessment, fraud detection, and compliance checking"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("risk_agent",config)
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY_7"))
        self.logger = StructuredLogger("risk_agent")
        self.api_key = balancer.get_best_key()
        print("self.api_key..........", self.api_key)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        # --- Metrics tracking ---
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 50  # keep last 50 runs

    def generate(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response
        except Exception as e:
            balancer.report_error(self.api_key)
            raise
            
    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        return bool(state.invoice_data and state.validation_result)

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        return bool(state.risk_assessment and state.risk_assessment.risk_score is not None)

    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        start_time = time.time()
        success = False
        try:
            if not self._validate_preconditions(state, workflow_type):
                state.overall_status = ProcessingStatus.FAILED
                self._log_decision(state, "Risk Assessment Analysis Failed", "Preconditions not met", confidence=0.0)
            
            invoice_data = state.invoice_data
            validation_result = state.validation_result
    
            base_score = await self._calculate_base_risk_score(invoice_data, validation_result)
            print("base_score:",base_score)
            fraud_indicators = await self._detect_fraud_indicators(invoice_data, validation_result)
            print("fraud_indicators:",fraud_indicators)
            compliance_issues = await self._check_compliance(invoice_data, state)
            print("compliance_issues:",compliance_issues)
            ai_assessment = await self._ai_risk_assessment(invoice_data, validation_result, fraud_indicators)
            print("ai_assessment:",ai_assessment)
    
            combined_score = self._combine_risk_factors(base_score, fraud_indicators, compliance_issues, ai_assessment)
            print("combined_score:",combined_score)
            
            risk_level = self._determine_risk_level(combined_score)
            print("risk_level:",risk_level)
            
            recommendation = self._generate_recommendation(risk_level, fraud_indicators, compliance_issues, validation_result)
            print("recommendation:", recommendation)
            state.risk_assessment = RiskAssessment(
                risk_level = risk_level,
                risk_score = combined_score,
                fraud_indicators = fraud_indicators,
                compliance_issues = compliance_issues,
                recommendation = recommendation["action"],
                reason = recommendation["reason"],
                requires_human_review = recommendation["requires_human_review"]
            )
    
            state.current_agent = "risk_agent"
            state.overall_status = ProcessingStatus.IN_PROGRESS
            success = True
            self._log_decision(
                state,
                "Risk Assessment Successful",
                "PDF text successfully verified by Risk Agent and checked by AI",
                combined_score,
                state.process_id
            )
            return state
        finally:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._record_execution(success, duration_ms)

    async def _calculate_base_risk_score(self, invoice_data, validation_result) -> float:
        """
        Calculates an intelligent risk score (0.0–1.0) based on validation results,
        invoice metadata, and contextual financial factors.
        """
        score = 0.0
    
        # --- 1. Validation & PO related risks ---
        if validation_result:
            if validation_result.validation_status == ValidationStatus.INVALID:
                score += 0.4
            elif validation_result.validation_status == ValidationStatus.PARTIAL_MATCH:
                score += 0.25
            elif validation_result.validation_status == ValidationStatus.MISSING_PO:
                score += 0.3
    
            # Core mismatch signals
            if not validation_result.amount_match:
                score += 0.2
            if not validation_result.rate_match:
                score += 0.15
            if not validation_result.quantity_match:
                score += 0.1
    
            # Low confidence from validation adds risk
            if validation_result.confidence_score is not None:
                score += (0.5 - validation_result.confidence_score) * 0.3 if validation_result.confidence_score < 0.5 else 0
    
        # --- 2. Invoice amount-based risk ---
        if invoice_data and invoice_data.total is not None:
            total = invoice_data.total
            if total > 1_000_000:
                score += 0.4   # Extremely high-value invoices
            elif total > 100_000:
                score += 0.25
            elif total > 10_000:
                score += 0.1
            elif total < 10:
                score += 0.15  # Suspiciously small invoice
    
        # --- 3. Temporal risks (based on due date) ---
        if invoice_data and getattr(invoice_data, "due_date", None):
            try:
                score += self._calculate_due_date_risk(invoice_data.due_date)
            except Exception:
                pass  # Graceful degradation if due_date is invalid
    
        # --- 4. Vendor / Customer risks ---
        if invoice_data and getattr(invoice_data, "customer_name", None):
            name = invoice_data.customer_name.lower()
            if "new_vendor" in name or "test" in name or "demo" in name:
                score += 0.2
            elif any(flag in name for flag in ["fraud", "fake", "invalid"]):
                score += 0.3
    
        # --- 5. Data reliability / extraction confidence ---
        if invoice_data and getattr(invoice_data, "extraction_confidence", None) is not None:
            conf = invoice_data.extraction_confidence
            if conf < 0.5:
                score += 0.2
            elif conf < 0.7:
                score += 0.1
    
        # --- 6. Currency and metadata anomalies ---
        currency = getattr(invoice_data, "currency", "USD") or "USD"
        if currency.upper() not in {"USD", "EUR", "GBP", "INR"}:
            score += 0.15  # uncommon currencies add risk
    
        # Normalize score within [0, 1.0]
        return round(min(score, 1.0), 3)

    def _calculate_due_date_risk(self, due_date_str: str) -> float:
        try:
            due_date = self._parse_date(due_date_str)
            days_until_due = (due_date - datetime.utcnow().date()).days
            if days_until_due < 0:
                return 0.2
            elif days_until_due < 5:
                return 0.1
            return 0.0
        except Exception:
            return 0.05

    def _parse_date(self, date_str: str) -> datetime.date:
        return datetime.strptime(date_str,"%Y-%m-%d").date()

    async def _detect_fraud_indicators(self, invoice_data, validation_result) -> List[str]:
        """
        Performs intelligent fraud detection on the given invoice and validation results.
        Returns a list of detected fraud indicators.
        """
        indicators = []
    
        # 1. PO / Validation mismatches
        if validation_result:
            if not validation_result.po_found:
                indicators.append("No matching Purchase Order found")
            if not validation_result.amount_match:
                indicators.append("Amount discrepancy detected")
            if not validation_result.rate_match:
                indicators.append("Rate inconsistency with Purchase Order")
            if not validation_result.quantity_match:
                indicators.append("Quantity mismatch detected")
            if validation_result.confidence_score is not None and validation_result.confidence_score < 0.6:
                indicators.append(f"Low validation confidence ({validation_result.confidence_score:.2f})")
    
        # 2. Vendor / Customer anomalies
        customer_name = getattr(invoice_data, "customer_name", "") or ""
        if "test" in customer_name.lower() or "demo" in customer_name.lower():
            indicators.append("Suspicious vendor name (Test/Demo account)")
        if "new_vendor" in customer_name.lower():
            indicators.append("First-time or unverified vendor")
        if any(keyword in customer_name.lower() for keyword in ["fraud", "fake", "invalid"]):
            indicators.append("Vendor flagged with risky keywords")
    
        # 3. Amount-level risk signals
        total = getattr(invoice_data, "total", 0.0) or 0.0
        if total > 1_000_000:
            indicators.append(f"Unusually high invoice total (${total:,.2f})")
        elif total < 10:
            indicators.append(f"Suspiciously low invoice total (${total:,.2f})")
    
        # 4. Date anomalies
        due_date = getattr(invoice_data, "due_date", None)
        invoice_date = getattr(invoice_data, "invoice_date", None)
        if invoice_date and due_date and (due_date - invoice_date).days < 0:
            indicators.append("Due date earlier than invoice date (possible manipulation)")
        elif invoice_date and due_date and (due_date - invoice_date).days < 3:
            indicators.append("Unusually short payment window")
    
        # 5. Duplicate or pattern-based red flags
        if invoice_data.invoice_number and invoice_data.invoice_number.lower().startswith("dup-"):
            indicators.append("Possible duplicate invoice ID pattern")
        if invoice_data.file_name and "copy" in invoice_data.file_name.lower():
            indicators.append("Invoice filename suggests duplication")
    
        # 6. Confidence anomalies (AI extraction)
        if invoice_data.extraction_confidence is not None and invoice_data.extraction_confidence < 0.5:
            indicators.append(f"Low extraction confidence ({invoice_data.extraction_confidence:.2f}) — possible OCR tampering")
    
        # 7. Currency or unusual metadata patterns
        if getattr(invoice_data, "currency", "").upper() not in {"USD", "EUR", "GBP", "INR"}:
            indicators.append(f"Uncommon currency code: {invoice_data.currency}")
        
        return indicators


    async def _check_compliance(self, invoice_data, state: InvoiceProcessingState) -> List[str]:
        """
        Performs a multi-layer compliance check on invoice and state integrity.
        Returns a list of detected compliance issues.
        """
        issues = []
    
        # 1. Invoice integrity checks
        if not invoice_data.invoice_number:
            issues.append("Missing invoice number")
        if not invoice_data.customer_name:
            issues.append("Missing customer name")
        if not invoice_data.total or invoice_data.total <= 0:
            issues.append("Invalid or missing total amount")
        if not invoice_data.due_date:
            issues.append("Missing due date")
    
        # 2. Item-level verification
        if not invoice_data.item_details or len(invoice_data.item_details) == 0:
            issues.append("No item details present")
        else:
            for item in invoice_data.item_details:
                if not getattr(item, "item_name", None):
                    issues.append("Item missing name")
                if getattr(item, "quantity", 1) <= 0:
                    issues.append(f"Invalid quantity for item '{item.item_name or 'Unknown'}'")
    
        # 3. Confidence & quality checks
        if invoice_data.extraction_confidence and invoice_data.extraction_confidence < 0.7:
            issues.append(f"Low extraction confidence ({invoice_data.extraction_confidence:.2f})")
    
        # 4. Workflow state checks
        if not getattr(state, "approval_chain", True):
            issues.append("Approval chain incomplete")
        if getattr(state, "escalation_required", False):
            issues.append("Escalation required before payment")
        if getattr(state, "human_review_required", False):
            issues.append("Pending human review")
    
        # 5. Audit consistency
        if len(state.audit_trail) == 0:
            issues.append("No audit trail entries found")
    
        # # 6. Optional receipt confirmation
        # if not getattr(invoice_data, "receipt_confirmed", False):
        #     issues.append("Missing receipt confirmation")
    
        # 7. Risk-based compliance (if risk assessment exists)
        if state.risk_assessment and state.risk_assessment.risk_score >= 0.7:
            issues.append(f"High risk score detected ({state.risk_assessment.risk_score:.2f})")
    
        return issues


    async def _ai_risk_assessment(
        self,
        invoice_data,
        validation_result,
        fraud_indicators: List[str]
    ) -> Dict[str, Any]:
        """
        Uses a Generative AI model (Gemini) to assess risk level based on
        structured invoice data, validation results, and detected fraud indicators.
    
        Returns:
            dict: {
                "risk_score": float between 0–1,
                "reason": str (explanation for the score)
            }
        """
        self.logger.logger.info("[RiskAgent] Running AI-based risk assessment...")
        # model_name = "gemini-2.5-flash"
        result = {"risk_score": 0.0, "reason": "Default – AI assessment not available"}
    
        try:
            # Initialize model
            # model = genai.GenerativeModel(model_name)
    
            # --- Construct dynamic and context-rich prompt ---
            prompt = f"""
            You are a financial risk analysis model for invoice fraud detection.
            Carefully analyze the following details:
    
            INVOICE DATA:
            {invoice_data}
    
            VALIDATION RESULT:
            {validation_result}
    
            DETECTED FRAUD INDICATORS:
            {fraud_indicators}
    
            TASK:
            1. Assess overall risk of this invoice being fraudulent or non-compliant.
            2. Provide reasoning.
            3. Respond **only in JSON** with keys:
               - "risk_score": a float between 0 and 1 (higher = higher risk)
               - "reason": short explanation of what contributed to this score.
    
            EXAMPLES:
            {{
                "risk_score": 0.85,
                "reason": "High amount mismatch, new vendor, and unusual currency"
            }}
            {{
                "risk_score": 0.25,
                "reason": "Valid PO and consistent totals, low fraud signals"
            }}
            """
            import asyncio
            # --- Model call ---
            response = self.generate(prompt)
            # response = await asyncio.to_thread(model.generate_content, prompt)
    
            # --- Clean and parse ---
            raw_text = getattr(response, "text", "") or ""
            cleaned_json = self._clean_json_response(raw_text)
            ai_output = json.loads(cleaned_json)
    
            # --- Validate AI output ---
            score = float(ai_output.get("risk_score", 0.0))
            reason = str(ai_output.get("reason", "No reason provided"))
    
            # Clamp score between 0–1 for safety
            result = {
                "risk_score": max(0.0, min(score, 1.0)),
                "reason": reason.strip()[:400]  # limit for logs
            }
    
            self.logger.logger.info(
                f"[RiskAgent] AI Risk Assessment completed: score={result['risk_score']}, reason={result['reason']}"
            )
    
        except json.JSONDecodeError as e:
            self.logger.logger.warning(f"[RiskAgent] JSON parsing failed: {e}")
            result["reason"] = "AI response could not be parsed"
    
        except Exception as e:
            self.logger.logger.error(f"[RiskAgent] AI assessment error: {e}", exc_info=True)
            result["reason"] = "Fallback to base risk model"
    
        return result


    def _clean_json_response(self, text: str) -> str:
        text = re.sub(r'^[^{]*','',text)
        text = re.sub(r'[^}]*$','',text)
        return text

    def _combine_risk_factors(
        self,
        base_score: float,
        fraud_indicators: List[str],
        compliance_issues: List[str],
        ai_assessment: Dict[str, Any]
    ) -> float:
        """
        Combines multiple risk components (base, fraud, compliance, and AI analysis)
        into a single normalized risk score between 0.0 and 1.0.
    
        Weighting strategy:
            - Base Score: foundation derived from deterministic checks
            - Fraud Indicators: +0.1 per flag (max +0.3)
            - Compliance Issues: +0.05 per issue (max +0.2)
            - AI Risk Score: contributes 40–50% of total weight
    
        Returns:
            float: final risk score clamped to [0, 1]
        """
        try:
            # Extract and normalize AI risk
            ai_score = float(ai_assessment.get("risk_score", 0.0))
            ai_score = max(0.0, min(ai_score, 1.0))
    
            # --- Weighted contributions ---
            fraud_contrib = min(len(fraud_indicators) * 0.1, 0.3)
            compliance_contrib = min(len(compliance_issues) * 0.05, 0.2)
            ai_contrib = 0.5 * ai_score if ai_score > 0 else 0.2 * base_score
    
            combined = base_score + fraud_contrib + compliance_contrib + ai_contrib
    
            # Cap at 1.0 for safety
            final_score = round(min(combined, 1.0), 3)
    
            self.logger.logger.info(
                f"[RiskAgent] Combined risk computed: base={base_score}, "
                f"fraud_flags={len(fraud_indicators)}, compliance_flags={len(compliance_issues)}, "
                f"ai_score={ai_score}, final={final_score}"
            )
    
            return final_score
    
        except Exception as e:
            self.logger.logger.error(f"[RiskAgent] Error combining risk factors: {e}", exc_info=True)
            return min(base_score + 0.2, 1.0)  # fallback conservative estimate


    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score<0.3:
            return RiskLevel.LOW
        elif risk_score<0.6:
            return RiskLevel.MEDIUM
        elif risk_score<0.8:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def _generate_recommendation(
        self,
        risk_level: RiskLevel,
        fraud_indicators: List[str],
        compliance_issues: List[str],
        validation_result
    ) -> Dict[str, Any]:
        """
        Generate a structured recommendation (approve, escalate, or reject)
        based on overall risk, fraud, and compliance outcomes.
    
        Decision Logic:
            - HIGH / CRITICAL risk → escalate for human review
            - INVALID validation → reject
            - Medium risk with minor issues → escalate
            - Otherwise → approve
    
        Returns:
            Dict[str, Any]: {
                'action': str,              # 'approve', 'escalate', or 'reject'
                'reason': str,              # Explanation summary
                'requires_human_review': bool
            }
        """
        try:
            # --- Determine key flags ---
            has_fraud = bool(fraud_indicators)
            has_compliance_issues = bool(compliance_issues)
            validation_invalid = (
                validation_result and validation_result.validation_status == ValidationStatus.INVALID
            )
    
            # --- Decision Logic ---
            if validation_invalid:
                action = "reject"
                requires_review = True
                reason = "Validation failed: " + "; ".join(fraud_indicators + compliance_issues or ["Invalid invoice data"])
    
            elif risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                action = "escalate"
                requires_review = True
                reason = f"High risk level detected ({risk_level.value}). Issues: " + "; ".join(fraud_indicators + compliance_issues or ["Potential anomalies"])
    
            elif has_fraud or has_compliance_issues:
                action = "escalate"
                requires_review = True
                reason = "Minor irregularities found: " + "; ".join(fraud_indicators + compliance_issues)
    
            else:
                action = "approve"
                requires_review = False
                reason = "All checks passed; invoice appears valid and compliant."
    
            # --- Structured Output ---
            recommendation = {
                "action": action,
                "reason": reason,
                "requires_human_review": requires_review,
            }
    
            self.logger.logger.info(
                f"[DecisionAgent] Recommendation generated: {recommendation}"
            )
            return recommendation
    
        except Exception as e:
            self.logger.logger.error(f"[DecisionAgent] Error generating recommendation: {e}", exc_info=True)
            # Safe fallback
            return {
                "action": "escalate",
                "reason": "Error during recommendation generation",
                "requires_human_review": True,
            }


    def _record_execution(self, success: bool, duration_ms: float):
        self.execution_history.append({
            # "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "duration_ms": duration_ms,
        })
        # Keep recent N only
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)

    async def health_check(self) -> Dict[str, Any]:
        total_runs = len(self.execution_history)
        if total_runs == 0:
            return {
                "Agent": "Risk Agent ⚠️",
                "Executions": 0,
                "Success Rate (%)": 0.0,
                "Avg Duration (ms)": 0.0,
                "Total Failures": 0,
                "Status": "idle",
                # "Timestamp": datetime.utcnow().isoformat()
            }
        metrics_data = {}
        executions = 0
        success_rate = 0.0
        avg_duration = 0.0
        failures = 0
        last_run = None

        # 1. Try to get live metrics from state
        # print("(self.state)-------",self.metrics)
        # print("self.state.agent_metrics-------", self.state.agent_metrics)
        if self.metrics:
            executions = self.metrics["processed"]
            avg_duration = self.metrics["avg_latency_ms"]
            failures = self.metrics["errors"]
            last_run = self.metrics["last_run_at"]
            success_rate = (executions - failures) / (executions+1e-8)

        # 2. API connectivity check
        gemini_ok = bool(self.api_key)
        api_status = "🟢 Active" if gemini_ok else "🔴 Missing Key"

        # 3. Health logic
        overall_status = "🟢 Healthy"
        if not gemini_ok or failures > 3:
            overall_status = "🟠 Degraded"
        if executions > 0 and success_rate < 0.5:
            overall_status = "🔴 Unhealthy"

        successes = sum(1 for e in self.execution_history if e["success"])
        failures = total_runs - successes
        avg_duration = round(mean(e["duration_ms"] for e in self.execution_history), 2)
        success_rate = round((successes / (total_runs+1e-8)) * 100, 2)
        
        return {
            "Agent": "Risk Agent ⚠️",
            "Executions": total_runs,
            "Success Rate (%)": success_rate,
            "Avg Duration (ms)": avg_duration,
            "API Status": api_status,
            "Total Failures": failures,
            "Last Run": str(last_run) if last_run else "Not applicable",
            # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Overall Health": overall_status,
        }

            
