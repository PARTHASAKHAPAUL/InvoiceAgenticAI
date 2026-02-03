
"""Payment Agent for Invoice Processing"""

# TODO: Implement agent

import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import time
import requests

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, PaymentDecision, PaymentStatus,
    RiskLevel, ValidationStatus, ProcessingStatus, RiskAssessment
)
from utils.logger import StructuredLogger

load_dotenv()


class PaymentAgent(BaseAgent):
    """Agent responsible for payment processing decisions and execution"""
    # Persistent in-memory history (like validation agent)
    health_history = []
    
    def __init__(self, config: Dict[str, Any] = None):
        # pass
        super().__init__("payment_agent", config)
        self.logger = StructuredLogger("PaymentAgent")
        self.approved_vendor_list = ["Acme Corporation", "TechNova Ltd", "SupplyCo"]
        self.retry_limit = 3
        # Health metrics tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_duration = 0.0
        self.last_transaction_id = None
        self.last_run = None

    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        # pass
        if workflow_type == "expedited":
            return bool(state.validation_result.validation_status.VALID and state.invoice_data)
        else:
            return bool(state.risk_assessment and state.invoice_data)

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        return bool(state.payment_decision)

    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        # pass
        start_time = time.time()
        try:
            if not self._validate_preconditions(state, workflow_type):
                state.overall_status = ProcessingStatus.FAILED
                self._log_decision(state, "Payment Agent Failed", "Preconditions not met", confidence=0.0)
                return state
            
            invoice_data = state.invoice_data
            validation_result = state.validation_result
            if workflow_type == "expedited":
                risk_assessment = RiskAssessment(
                risk_level = RiskLevel.LOW,
                risk_score = 0.3,
                fraud_indicators = None,
                compliance_issues = None,
                recommendation = None,
                reason = "Expedited Workflow Called",
                requires_human_review = "Not needed due to Expedited Workflow"
                )
                payment_decision = PaymentDecision(
                decision = "auto_pay",
                status = PaymentStatus.APPROVED,
                approved_amount = invoice_data.total,
                transaction_id = f"TXN-{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}",
                payment_method = self._select_payment_method(invoice_data.total),
                approval_chain = ["system_auto_approval"],
                rejection_reason = None,
                scheduled_date = self._calculate_payment_date(invoice_data.due_date, "ACH")
                )
                payment_result = await self._execute_payment(invoice_data, payment_decision)
                payment_decision = self._update_payment_decision(payment_decision, payment_result)
        
                justification = await self._generate_payment_justification(
                    invoice_data, payment_decision, validation_result, risk_assessment
                )

                state.payment_decision = payment_decision
                state.overall_status = ProcessingStatus.COMPLETED
                state.current_agent = "payment_agent"
                # success criteria
                if payment_decision.status == PaymentStatus.APPROVED:
                    self.successful_executions += 1
                else:
                    self.failed_executions += 1

                self.last_transaction_id = payment_decision.transaction_id
                self._log_decision(state, payment_decision.status, justification, 95.0, state.process_id)
                return state
            else:
                risk_assessment = state.risk_assessment
        
                payment_decision = await self._make_payment_decision(
                    invoice_data, validation_result, risk_assessment, state
                )
                if payment_decision.decision == "auto_pay":
                    state.approval_chain = [
                        {
                            "approved_by":"system_auto_approval in payment_agent",
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        ]
                else:
                    state.approval_chain = [{"payment_agent":"Failed or Rejected"}]

        
                payment_result = await self._execute_payment(invoice_data, payment_decision)
                payment_decision = self._update_payment_decision(payment_decision, payment_result)
        
                justification = await self._generate_payment_justification(
                    invoice_data, payment_decision, validation_result, risk_assessment
                )

                state.payment_decision = payment_decision
                state.overall_status = ProcessingStatus.COMPLETED
                state.current_agent = "payment_agent"
                # success criteria
                if payment_decision.status == PaymentStatus.APPROVED:
                    print("self.successful_executions---", self.successful_executions)
                    self.successful_executions += 1
                else:
                    self.failed_executions += 1

                self.last_transaction_id = payment_decision.transaction_id
                self._log_decision(state, payment_decision.status, justification, 95.0, state.process_id)
                return state
            
        except Exception as e:
            self.failed_executions += 1
            self.logger.logger.error(f"[PaymentAgent] Execution failed: {e}")
            state.overall_status = ProcessingStatus.FAILED
            return state

        finally:
            duration = (time.time() - start_time) * 1000  # in ms
            print("self.total_executions---", self.total_executions)
            self.last_run = datetime.utcnow().isoformat()
            self.total_executions += 1
            self.total_duration += duration
            self._record_health_metrics(duration)

    async def _make_payment_decision(self, invoice_data, validation_result,
                                   risk_assessment, state: InvoiceProcessingState) -> PaymentDecision:
        # pass
        amount = invoice_data.total or invoice_data.total_amount or 0.0
        risk_level = risk_assessment.risk_level
        validation_status = validation_result.validation_status

        if risk_level == RiskLevel.CRITICAL or validation_status == ValidationStatus.INVALID:
            decision = PaymentDecision(
                decision = "reject",
                status = PaymentStatus.FAILED,
                approved_amount = 0.0,
                transaction_id = None,
                payment_method = None,
                approval_chain = [],
                rejection_reason = "Critical Risk or Invalid Validation",
                scheduled_date = None
            )
        elif risk_level == RiskLevel.LOW or amount < 5000:
            decision = PaymentDecision(
                decision = "auto_pay",
                status = PaymentStatus.APPROVED,
                approved_amount = amount,
                transaction_id = f"TXN-{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}",
                payment_method = self._select_payment_method(amount),
                approval_chain = ["system_auto_approval"],
                rejection_reason = None,
                scheduled_date = self._calculate_payment_date(invoice_data.due_date, "ACH")
            )
        elif risk_level == RiskLevel.MEDIUM or validation_status == ValidationStatus.PARTIAL_MATCH:
            decision = PaymentDecision(
                decision = "hold",
                status = PaymentStatus.PENDING_APPROVAL,
                approved_amount = amount,
                transaction_id = None,
                payment_method = self._select_payment_method(amount),
                approval_chain = ["system_auto_approval", "finance_manager_approval"],
                rejection_reason = None,
                scheduled_date = self._calculate_payment_date(invoice_data.due_date, "ACH")
            )
        else:
            decision = PaymentDecision(
                decision = "manual_approval",
                status = PaymentStatus.PENDING_APPROVAL,
                approved_amount = amount,
                transaction_id = None,
                payment_method = self._select_payment_method(amount),
                approval_chain = ["system_auto_approval", "executive_approval"],
                rejection_reason = None,
                scheduled_date = self._calculate_payment_date(invoice_data.due_date, "WIRE")
            )

        return decision

    def _select_payment_method(self, amount: float) -> str:
        # pass
        if amount < 5000:
            return "ACH"
        elif amount < 25000:
            return "WIRE"
        return "MANUAL"

    def _calculate_payment_date(self, due_date_str: Optional[str], payment_method: str) -> datetime:
        # pass
        due_date = self._parse_date(due_date_str)
        if not due_date:
            due_date = datetime.utcnow().date() + timedelta(days=3)
        offset = 1 if payment_method == "ACH" else 2
        return datetime.combine(due_date, datetime.min.time()) + timedelta(days=offset)


    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        # pass
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None

    # async def _execute_payment(self, invoice_data, payment_decision: PaymentDecision) -> Dict[str, Any]:
    #     # pass
    #     await self._async_sleep(1)
    #     response = requests.post("http://localhost:8000", data=PaymentRequest)
    #     if payment_decision.status == PaymentStatus.FAILED:
    #         return {"status": "failed", "message": "Payment rejected by policy."}
    #     return {"status": "success", "transaction_id": payment_decision.transaction_id or f"TXN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}", "message": "Payment executed."}

    async def _execute_payment(self, invoice_data, payment_decision: PaymentDecision) -> Dict[str, Any]:
        """Send payment request to web API and return response with transaction_id"""
        import asyncio
        await asyncio.sleep(1)
    
        payment_payload = {
            "order_id": invoice_data.invoice_number or f"INV-{int(datetime.utcnow().timestamp())}",
            "customer_name": invoice_data.customer_name or "Unknown Vendor",
            "amount": float(invoice_data.total),
            "currency": "USD",
            # "method": payment_decision.payment_method.lower(),
            "recipient_account": "auto_generated_account",
            "due_date": str(invoice_data.due_date or datetime.utcnow().date())
        }
    
        try:
            response = requests.post("http://localhost:8001/initiate_payment", json=payment_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print("res from apiii =======", result)
                return {
                    "status": "success" if result["status"] == "SUCCESS" else "failed",
                    "transaction_id": result["transaction_id"],
                    "message": result["message"]
                }
            else:
                print("res from apiii111111 =======", result)
                return {"status": "failed", "message": f"HTTP {response.status_code}: {response.text}"}
    
        except Exception as e:
            print("res from apiii111111222222222222 =======", result)
            return {"status": "failed", "message": f"Payment API error: {e}"}

    async def _async_sleep(self, seconds: int):
        # pass
        import asyncio
        await asyncio.sleep(seconds)

    def _update_payment_decision(self, payment_decision: PaymentDecision,
                               payment_result: Dict[str, Any]) -> PaymentDecision:
        # pass
        if payment_result.get("status") == "success":
            payment_decision.status = PaymentStatus.APPROVED
            payment_decision.transaction_id = payment_result.get("transaction_id")
        else:
            payment_decision.status = PaymentStatus.FAILED
            payment_decision.rejection_reason = payment_result.get("message")
        return payment_decision


    async def _generate_payment_justification(self, invoice_data, payment_decision: PaymentDecision,
                                            validation_result, risk_assessment) -> str:
        # pass
        reason = f"Payment Decision: {payment_decision.status}. "
        if payment_decision.status == PaymentStatus.FAILED:
            reason += f"Reason: {payment_decision.rejection_reason}"
        reason += f"Risk level: {risk_assessment.risk_level}. Validation: {validation_result.validation_status}."
        return reason

    def _record_health_metrics(self, duration: float):
        """Update and record health statistics"""
        success_rate = (
            (self.successful_executions / self.total_executions) * 100
            if self.total_executions else 0
        )
        avg_duration = (
            self.total_duration / self.total_executions
            if self.total_executions else 0
        )
        overall_status = "🟢 Healthy"
        if success_rate < 70:
            overall_status = "🟠 Degraded"
        if success_rate < 60:
            overall_status = "🔴 Unhealthy"

        metrics = {
            "Agent": "Payment Agent 💳",
            "Executions": self.total_executions,
            "Success Rate (%)": round(success_rate, 2),
            "Avg Duration (ms)": round(avg_duration, 2),
            "Total Failures": self.failed_executions,
            "Last Transaction ID": self.last_transaction_id or "N/A",
            # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Last Run": self.last_run,
            "Overall Health": overall_status,
        }

        PaymentAgent.health_history.append(metrics)
        PaymentAgent.health_history = PaymentAgent.health_history[-50:]  # keep last 50

    async def health_check(self) -> Dict[str, Any]:
        """Return the current or last known health state"""
        await self._async_sleep(0.05)
        if not PaymentAgent.health_history:
            return {
                "Agent": "Payment Agent 💳",
                "Executions": 0,
                "Success Rate (%)": 0.0,
                "Avg Duration (ms)": 0.0,
                "Total Failures": 0,
                "Last Transaction ID": "N/A",
            }
        return PaymentAgent.health_history[-1]
