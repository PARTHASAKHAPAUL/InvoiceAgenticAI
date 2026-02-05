
"""Escalation Agent for Invoice Processing"""

# TODO: Implement agent

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ProcessingStatus, PaymentStatus,
    RiskLevel, ValidationStatus
)
from utils.logger import StructuredLogger

load_dotenv()


class EscalationAgent(BaseAgent):
    """Agent responsible for escalation management and human-in-the-loop workflows"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("escalation_agent",config)
        self.logger = StructuredLogger("EscalationAgent")

        self.escalation_triggers = {
            'high_risk' : {'route_to':'risk_manager','sla_hours':4},
            'validation_failure': {'route_to':'finance_manager','sla_hours':8},
            'high_value': {'route_to':'cfo','sla_hours':24},
            'fraud_suspicion': {'route_to':'fraud_team','sla_hours':2},
            'new_vendor':{'route_to':'procurement','sla_hours':48}
        }

    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        # pass
        return hasattr(state,'invoice_data') and hasattr(state,'risk_assessment')

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        return hasattr(state,'escalation_details')

    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        # pass
        self.logger.logger.info('Executing Escalation Agent...')
        if not self._validate_preconditions(state, workflow_type):
            self.logger.logger.error("Preconditions not meet for Escalation handling")
            state.status = ProcessingStatus.FAILED
            self._log_decision(state, "Escalation Agent Failed", "Preconditions not met", confidence=0.0)
            return state

        escalation_type = self._determine_escalation_type(state)
        if not escalation_type:
            self.logger.logger.info("No escalation required for this invoice.")
            state.escalation_required = False
            state.overall_status = 'completed'
            return state
        
        priority_level = self._calculate_priority_level(state)
        approver_info = self._route_to_approver(state, escalation_type,priority_level)
        summary = await self._generate_escalation_summary(state,escalation_type,approver_info)

        escalation_record = await self._create_escalation_record(state, escalation_type, priority_level, approver_info,summary)
        await self._send_escalation_notifications(state,escalation_record,approver_info)
        await self._setup_sla_monitoring(state,escalation_record,priority_level)

        state.escalation_required = True
        state.human_review_required = True
        state.escalation_details = escalation_record
        state.human_review_required = summary
        state.escalation_reason = escalation_record["escalation_reason"]
        state.current_agent = 'escalation_agent'
        state.overall_status = 'escalated'
        self._log_decision(
            state,
            "Escalation Successful",
            "PDF successfully escalated to Human for review",
            "N/A",
            state.process_id
        )
        self.logger.logger.info('Escalation record successfully created and routed.')
        return state

    def _determine_escalation_type(self, state: InvoiceProcessingState) -> str:
        # pass
        risk = getattr(state,'risk_assessment',{})
        validation = getattr(state,'validation_result',{})
        invoice = getattr(state,'invoice_data',{})
        risk_level = getattr(risk,'risk_level',{})
        amount = getattr(invoice,'total',0)
        vendor = getattr(invoice,'customer_name','')
        # fraud_indicators = risk.get('fraud_indicators',[])
        fraud_indicators = getattr(risk,'fraud_indicators',[])

        if risk_level in ['high','critical']:
            return 'high_risk'
        elif state.validation_status == 'invalid' or state.validation_status == 'missing_po':
            return 'validation_failure'
        elif amount and amount>250000:
            return 'high_value'
        elif len(fraud_indicators) > 3:
            return 'fraud_suspicion'
        elif vendor and 'new' in vendor.lower():
            return 'new_vendor'
        else:
            return None

    def _calculate_priority_level(self, state: InvoiceProcessingState) -> str:
        # pass
        # risk = getattr(state,'risk_assessment',{}).get('risk_level','low').lower()
        # amount = getattr(state,'invoice_data',{}).get('total',0)
        risk_assessment = getattr(state,'risk_assessment',{})
        invoice_data = getattr(state,'invoice_data',{})
        risk = getattr(risk_assessment,'risk_level','low').lower()
        amount = getattr(invoice_data,'total',0)
        if risk == 'critical' or amount > 50000:
            return 'urgent'
        elif risk == 'high' or amount > 25000:
            return 'high'
        else:
            return 'medium'

    def _route_to_approver(self, state: InvoiceProcessingState,
                          escalation_type: str, priority_level: str) -> Dict[str, Any]:
        # pass
        # print(self.escalation_triggers)
        route_info = self.escalation_triggers.get(escalation_type,{})
        # print("route_info..................", route_info)
        assigned_to = route_info.get('route_to','finance_manager')
        sla_hours = route_info.get('sla_hours',8)
        approvers = ['finance_manager']
        if assigned_to == 'cfo':
            approvers.append('cfo')
        return {
            'assigned_to':assigned_to,
            'sla_hours':sla_hours,
            'approval_required_from':approvers
        }


    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        # pass
        try:
            return datetime.strptime(date_str,"%Y-%m-%d").date()
        except Exception:
            return None

    async def _generate_escalation_summary(self, state: InvoiceProcessingState,
                                         escalation_type: str, approver_info: Dict[str, Any]) -> str:
        # pass

        risk = getattr(state,'risk_assessment',{})
        invoice = getattr(state,'invoice_data',{})
        risk_level = getattr(risk,'risk_level',{})
        amount = getattr(invoice,'total',0)
        # invoice = state.invoice_data
        # risk = state.risk_assessment
        reason = ""

        if escalation_type == 'high_risk':
            reason = f"Invoice marked as high risk ({risk_level})."
        elif escalation_type == 'validation_failure':
            reason = 'Validation discrepancies require finance approval.'
        elif escalation_type == 'high_value':
            reason = f"High-value invoice ({amount}) requires CFO approval."
        elif escalation_type == 'fraud_suspicion':
            reason = 'Fraud suspicion based on anomalies detected'
        elif escalation_type == 'new_vendor':
            reason = 'Vendor is new and not yet in approved list.'
        return f"{reason} Routed to {approver_info['assigned_to']} for review."


    async def _create_escalation_record(self, state: InvoiceProcessingState,
                                      escalation_type: str, priority_level: str,
                                      approver_info: Dict[str, Any], summary: str) -> Dict[str, Any]:
        # pass
        timestamp = datetime.utcnow()
        sla_deadline = timestamp+timedelta(hours=approver_info['sla_hours'])
        return {
            'escalation_type':escalation_type,
            'severity':priority_level,
            'assigned_to':approver_info['assigned_to'],
            'escalation_time':timestamp.isoformat()+'Z',
            'sla_deadline':sla_deadline.isoformat()+'Z',
            'notification_sent':True,
            'approval_required_from':approver_info['approval_required_from'],
            'escalation_reason':summary
        }


    async def _send_escalation_notifications(self, state: InvoiceProcessingState,
                                           escalation_record: Dict[str, Any],
                                           approver_info: Dict[str, Any]) -> Dict[str, Any]:
        # pass
        try:
            subject = f"[Escalation Alert] Invoice requires {approver_info['assigned_to']} review"
            body = f"""
            Escalation Type: {escalation_record['escalation_type']}
            severity: {escalation_record['severity']}
            SLA Deadline: {escalation_record['sla_deadline']}
            reason: {escalation_record['escalation_reason']}
            """
            to_email = f"{approver_info['assigned_to']}@company.com"
            self._send_email(to_email,subject,body)
            self.logger.logger.info(f"Escalation notification send to {to_email}.")
            return {'status':'send','to':to_email}
        except Exception as e:
            self.logger.logger.error(f'Failed to send notification: {e}')
            return {'status':'failed','error':str(e)}

    def _send_email(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        # pass
        try:
            sender = os.getenv('EMAIL_SENDER','noreply@invoicesystem.com')
            msg = MIMEMultipart()
            msg['From'] = send
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body,'plain'))
            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
            return {'sent':True}
        except Exception as e:
            return {'sent':False, 'error':str(e)}


    async def _setup_sla_monitoring(self, state: InvoiceProcessingState,
                                  escalation_record: Dict[str, Any], priority_level: str):
        # pass
        self.logger.logger.debug(
            f"SLA monitoring initialized for {escalation_record['escalation_type']}"
            f"with deadline {escalation_record['sla_deadline']}"
        )

    async def resolve_escalation(self, escalation_id: str, resolution: str,
                               resolver: str) -> Dict[str, Any]:
        # pass
        return {
            'escalation_id':escalation_id,
            'resolved_by':resolver,
            'resolution_notes':resolution,
            'resolved_at':datetime.utcnow().isoformat()+'Z',
            'status':'resolved'
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Performs a detailed health check for the Escalation Agent.
        Includes operational metrics, configuration validation, and reliability stats.
        """

        start_time = datetime.utcnow()
        self.logger.logger.info("Performing health check for EscalationAgent...")

        executions = 0
        avg_duration = 0.0
        failures = 0
        last_run = None
        success_rate = 0.0
        
        try:
            if self.metrics:
                executions = self.metrics["processed"]
                avg_duration = self.metrics["avg_latency_ms"]
                failures = self.metrics["errors"]
                last_run = self.metrics["last_run_at"]
                success_rate = (executions - failures) / (executions + 1e-8) * 100.0 if executions > 0 else 0.0

            total_executions = executions
            total_failures = failures
            avg_duration_ms = avg_duration

            # Email and trigger configuration validation
            email_configured = bool(os.getenv('EMAIL_SENDER'))
            missing_triggers = [k for k, v in self.escalation_triggers.items() if not v.get("route_to")]

            # Duration calculation
            # duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            # last_run = self.metrics["last_run_at"]

            health_report = {
                "Agent": "Escalation Agent 🚨",
                "Executions": total_executions,
                "Success Rate (%)": round(success_rate, 2),
                "Avg Duration (ms)": round(avg_duration_ms, 2) if avg_duration_ms else "Not Called",
                "Total Failures": total_failures,
                # "Email Configured": email_configured,
                # "Available Triggers": list(self.escalation_triggers.keys()),
                "Missing Routes": missing_triggers,
                "Last Run": self.metrics["last_run_at"],
                "Overall Health": "🟢 Healthy" if (success_rate > 70 or total_executions == 0) else "Degraded ⚠️",
                # "Response Time (ms)": round(duration_ms, 2)
                # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }

            self.logger.logger.info("EscalationAgent health check completed successfully.")
            return health_report

        except Exception as e:
            error_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.logger.error(f"Health check failed: {e}")

            # Return degraded health if something goes wrong
            return {
                "Agent": "EscalationAgent ❌",
                "Overall Health": "Degraded",
                "Error": str(e),
                "Timestamp": datetime.utcnow().isoformat() + "Z"
            }
