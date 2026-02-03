
"""Audit Agent for Invoice Processing"""

# TODO: Implement agent

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import time
from statistics import mean

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ProcessingStatus, PaymentStatus,
    ValidationStatus, RiskLevel
)
from utils.logger import StructuredLogger

load_dotenv()


class AuditAgent(BaseAgent):
    """Agent responsible for audit trail generation, compliance tracking, and reporting"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("audit_agent",config)
        self.logger = StructuredLogger("AuditAgent")
        # --- Health tracking ---
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 50  # store last 50 runs

    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        """
        Ensure that the state object is properly initialized before invoice processing begins.
        Checks for presence of critical fields like process_id, file_name, and timestamps.
        """
        if not state:
            return False
    
        # Must have valid process id and file name
        if not getattr(state, "process_id", None) or not getattr(state, "file_name", None):
            return False
    
        # Must have timestamps and valid status
        if not getattr(state, "created_at", None) or not getattr(state, "overall_status", None):
            return False
    
        # Should not already be marked complete
        if state.overall_status in ("failed", "pending"):
            return False
    
        return True


    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """
        Validate that all expected outputs and audit data are present after processing.
        Ensures that critical workflow components completed successfully.
        """
        if not state:
            return False
    
        # Must have processed invoice data and validation results
        if not state.invoice_data or not state.validation_result:
            return False
    
        # Must have at least one audit entry for traceability
        if not state.audit_trail or len(state.audit_trail) == 0:
            return False
    
        # Risk or payment results may be optional, but check consistency
        if state.risk_assessment and state.risk_assessment.risk_score > 1.0:
            return False  # sanity check for invalid scores
    
        # Final status should not be pending anymore
        if state.overall_status == "pending":
            return False
    
        return True


    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        """Main audit generation workflow"""
        self.logger.logger.info("Starting audit trail generation")
        start_time = time.time()
        success = False
        try:
            if not self._validate_preconditions(state, workflow_type):
                self.logger.logger.error("Preconditions not met for audit generation")
                state.overall_status = ProcessingStatus.FAILED
                self._log_decision(state, "Audit Failed", "Preconditions not met", confidence=0.0)
                return state
    
            audit_record = await self._generate_audit_record(state)
            print("audit_record---------", audit_record)
            compliance_results = await self._perform_compliance_checks(state,audit_record)
            print("compliance_results---------", compliance_results)
            audit_summary = await self._generate_audit_summary(state,audit_record,compliance_results)
            print("audit_summary---------", audit_summary)
            await self._save_audit_records(state,audit_record,audit_summary,compliance_results)
    
            reportable_events = await self._identify_reportable_events(state,audit_record)
            print("reportable_events---------", reportable_events)

            await self._generate_audit_alerts(state,reportable_events)
    
            state.audit_trail = audit_record.get("audit_trail",[])
            print("state.audit_trail---------", state.audit_trail)
            state.compliance_report = compliance_results
            state.current_agent = "audit_agent"
            state.overall_status = "completed"
    
            self.logger.logger.info("Audit trail and compliance generated successfully")
            success = True
            self._log_decision(
                state,
                "Auditing Successful",
                "Auditing Processed",
                100.0,
                state.process_id
            )
            state.audit_trail[-1]
            return state
            
        except Exception as e:
            self.logger.logger.error(f"Audit agent execution failed: {e}")
            state.overall_status = ProcessingStatus.FAILED
            return state

        finally:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._record_execution(success, duration_ms, state)

    async def _generate_audit_record(self, state: InvoiceProcessingState) -> Dict[str, Any]:
        """
        Aggregate and structure all agent-level logs into a consistent audit report.
        Uses the state's existing audit_trail list and agent_metrics for detailed tracking.
        """
        self.logger.logger.debug("Generating audit record")
    
        if not isinstance(state, InvoiceProcessingState):
            raise ValueError("Invalid state object passed to _generate_audit_record")
    
        audit_trail_records = []
        for entry in getattr(state, "audit_trail", []):
            record = {
                "process_id": getattr(entry, "process_id", state.process_id),
                "timestamp": getattr(entry, "timestamp", datetime.utcnow().isoformat() + "Z"),
                "agent_name": getattr(entry, "agent_name", "unknown"),
                "action": getattr(entry, "action", "undefined"),
                # "status": getattr(entry, "status", "completed"),
                "details": getattr(entry, "details", {}),
                # "duration_ms": getattr(entry, "details", {}).get("duration_ms", 0),
                # "error_message": getattr(entry, "details", {}).get("error_message", None),
            }
            audit_trail_records.append(record)
    
        # Include agent metrics summary for full traceability
        metrics_summary = {
            agent: {
                "executions": getattr(m, "processed_count", 0),
                "success_rate": getattr(m, "success_rate", 0),
                "failures": getattr(m, "errors", 0),
                "avg_duration_ms": getattr(m, "avg_latency_ms", 0.0),
                "last_run_at": getattr(m, "last_run_at", None),
            }
            for agent, m in getattr(state, "agent_metrics", {}).items()
        }
    
        audit_report = {
            "process_id": state.process_id,
            "created_at": state.created_at.isoformat() + "Z",
            "updated_at": state.updated_at.isoformat() + "Z",
            "total_entries": len(audit_trail_records),
            "audit_trail": audit_trail_records,
            "metrics_summary": metrics_summary,
        }
    
        self.logger.logger.info(
            f"Audit record generated with {len(audit_trail_records)} entries for process {state.process_id}"
        )
    
        return audit_report

    async def _perform_compliance_checks(
        self, state: InvoiceProcessingState, audit_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform SOX, GDPR, and financial compliance validations.
        Aggregates results from internal compliance check methods and produces
        a structured compliance report.
        """
        self.logger.logger.debug("Performing compliance checks for process %s", state.process_id)
    
        # Defensive: ensure proper structures
        if not isinstance(state, InvoiceProcessingState):
            raise ValueError("Invalid state object passed to _perform_compliance_checks")
        if not isinstance(audit_record, dict):
            raise ValueError("Invalid audit record structure")
    
        # Run all compliance sub-checks safely
        sox = self._check_sox_compliance(state, audit_record) or {}
        privacy = self._check_data_privacy_compliance(state, audit_record) or {}
        financial = self._check_financial_controls(state, audit_record) or {}
        completeness = self._check_audit_trail_completeness(state, audit_record) or {}
    
        # Normalize results for consistency
        sox_issues = sox.get("issues", [])
        privacy_issues = privacy.get("issues", [])
        financial_issues = financial.get("issues", [])
        is_complete = completeness.get("complete", True)
    
        # Compose structured compliance summary
        compliance_report = {
            "process_id": state.process_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sox_compliance": "compliant" if not sox_issues else "non_compliant",
            "gdpr_compliance": "compliant" if not privacy_issues else "non_compliant",
            "financial_controls": "passed" if not financial_issues else "failed",
            "audit_trail_complete": is_complete,
            "retention_policy": getattr(self.config, "retention_policy", "7_years"),
            "encryption_status": "encrypted",
            "issues": {
                "sox": sox_issues,
                "privacy": privacy_issues,
                "financial": financial_issues,
            },
        }
    
        # Optional: attach compliance report to the state for future use
        setattr(state, "compliance_report", compliance_report)
        state.updated_at = datetime.utcnow()
    
        self.logger.logger.info(
            f"Compliance checks completed for process {state.process_id}: "
            f"SOX={compliance_report['sox_compliance']}, "
            f"GDPR={compliance_report['gdpr_compliance']}, "
            f"Financial={compliance_report['financial_controls']}"
        )
    
        return compliance_report


    def _check_sox_compliance(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Intelligent SOX compliance verification.
        Checks that all approval steps, segregation of duties,
        and key sign-offs are properly recorded and timestamped.
        """
        issues = []
    
        approval_chain = getattr(state, "approval_chain", [])
        if not approval_chain:
            issues.append("Missing approval chain records")
        else:
            # Verify each approval step includes signer and timestamp
            for step in approval_chain:
                if not step.get("approved_by") or not step.get("timestamp"):
                    issues.append(f"Incomplete approval step: {step}")
            # Optional: check segregation of duties
            approvers = [a.get("approved_by") for a in approval_chain if a.get("approved_by")]
            if len(set(approvers)) < len(approvers):
                issues.append("Potential conflict of interest: repeated approver detected")
        
        VALID_ACTIONS = {
        "Extraction Successful",
        "Validation Successful",
        "Risk Assessment Successful",
        "Agent Successfully Executed",
        "approved"
        }
        has_final_approval = all(
            any(keyword in entry.get("action", "") for keyword in VALID_ACTIONS)
            for entry in audit_record.get("audit_trail", [])
        )

        if not has_final_approval:
            issues.append("Some approval event yet to successful in audit trail")
    
        return {"issues": issues}


    def _check_data_privacy_compliance(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate GDPR / Data Privacy compliance.
        Ensures that no unmasked personal or financial data is logged or stored.
        """
        issues = []
        text_repr = str(audit_record).lower()
    
        # PII patterns to scan for (we can expand this list)
        suspicious_patterns = ["@gmail.com", "@yahoo.com", "ssn", "credit card", "bank_account"]
    
        for pattern in suspicious_patterns:
            if pattern in text_repr:
                issues.append(f"Unmasked PII detected: '{pattern}'")
    
        # Ensure encryption and retention policy
        # if getattr(state, "config", {}).get("encryption_status") != "encrypted":
        #     issues.append("Data encryption not confirmed")
    
        # if "retention_policy" not in getattr(state, "config", {}):
        #     issues.append("Retention policy not defined")
    
        return {"issues": issues}


    def _check_financial_controls(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate financial control compliance.
        Ensures that transactions, approvals, and risk assessments
        are properly recorded before payment release.
        """
        issues = []
    
        # Check for missing financial artifacts
        if not getattr(state, "payment_decision", None):
            issues.append("Missing payment decision records")
    
        if not getattr(state, "validation_result", None):
            issues.append("Missing validation result for payment control")
    
        if state.validation_result and state.validation_result.validation_status == "invalid":
            issues.append("Invoice marked invalid but payment decision exists")
    
        # Cross-check audit trail for financial actions
        actions = [a.get("action", "").lower() for a in audit_record.get("audit_trail", [])]
        if not any("approved" in a for a in actions):
            issues.append("No payment-related activity recorded in audit trail")
    
        return {"issues": issues}

    def _check_audit_trail_completeness(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensure all mandatory agents and workflow stages were executed and logged.
        Validates sequence integrity and timestamp order.
        """
        required_agents = ["document_agent", "validation_agent", "risk_agent", "payment_agent"]
        logged_agents = [x.get("agent_name") for x in audit_record.get("audit_trail", [])]
        missing = [a for a in required_agents if a not in logged_agents]
    
        complete = len(missing) == 0
        
        timestamps = []
        for e in audit_record.get("audit_trail", []):
            ts = e.get("timestamp")
            if ts:
                try:
                    if isinstance(ts, datetime):
                        timestamps.append(ts)
                    else:
                        # Normalize 'Z' and try parsing
                        ts_str = str(ts).replace("Z", "+00:00")
                        try:
                            timestamps.append(datetime.fromisoformat(ts_str))
                        except Exception:
                            try:
                                timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f"))
                            except Exception:
                                timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    self.logger.logger.warning(f"Invalid timestamp format in audit trail: {ts}")



        if timestamps and timestamps != sorted(timestamps):
            missing.append("Non-sequential timestamps detected in audit trail")
    
        # Check for duplicate agent entries
        if len(logged_agents) != len(set(logged_agents)):
            missing.append("Duplicate agent entries found in audit trail")
    
        return {"complete": complete, "missing": missing}


    async def _generate_audit_summary(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any],
        compliance_results: Dict[str, Any]
    ) -> str:
        """
        Generate a structured textual audit summary report.
        Combines audit record data and compliance results into a concise, test-friendly JSON summary.
        """
        self.logger.logger.debug("Generating audit summary for process %s", state.process_id)
    
        # Defensive: ensure valid input types
        if not isinstance(state, InvoiceProcessingState):
            raise ValueError("Invalid state object passed to _generate_audit_summary")
        if not isinstance(audit_record, dict):
            raise ValueError("Invalid audit record structure")
        if not isinstance(compliance_results, dict):
            raise ValueError("Invalid compliance results structure")
    
        # Extract audit trail count safely
        total_actions = len(audit_record.get("audit_trail", []))
    
        # Safely extract compliance keys
        sox_status = compliance_results.get("sox_compliance", "unknown")
        gdpr_status = compliance_results.get("gdpr_compliance", "unknown")
        financial_status = compliance_results.get("financial_controls", "unknown")
        retention_policy = compliance_results.get("retention_policy", "7_years")
    
        # Build structured summary
        summary_data = {
            "process_id": state.process_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_actions": total_actions,
            "overall_status": getattr(state, "overall_status", "UNKNOWN"),
            "compliance": {
                "SOX": sox_status,
                "GDPR": gdpr_status,
                "Financial": financial_status,
            },
            "retention_policy": retention_policy,
        }
    
        # Attach to state for post-validation
        setattr(state, "audit_summary", summary_data)
        state.updated_at = datetime.utcnow()
    
        # Log completion
        self.logger.logger.info(
            f"Audit summary generated for process {state.process_id}: "
            f"Actions={total_actions}, SOX={sox_status}, GDPR={gdpr_status}, Financial={financial_status}"
        )
    
        # Return formatted JSON for easy test validation or storage
        return json.dumps(summary_data, indent=2)


    async def _save_audit_records(self, state: InvoiceProcessingState,
                                audit_record: Dict[str, Any],
                                audit_summary: str,
                                compliance_results: Dict[str, Any]):
        """Save audit log to file"""
        os.makedirs("logs/audit",exist_ok=True)
        file_path = f"logs/audit/audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path,"w") as f:
            json.dump({
                "audit_trail": audit_record["audit_trail"],
                "summary": json.loads(audit_summary),
                "compliance":compliance_results
            },f,indent=2, default=str)
        self.logger.logger.info(f"Audit record saved:{file_path}")

    async def _identify_reportable_events(
        self,
        state: InvoiceProcessingState,
        audit_record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify reportable anomalies or irregularities from the audit trail for compliance auditors.
        Includes failed actions, high latency events, and repeated errors.
        """
        self.logger.logger.debug("Analyzing audit trail for reportable events...")
    
        reportable: List[Dict[str, Any]] = []
        audit_trail = audit_record.get("audit_trail", [])
    
        if not audit_trail:
            self.logger.logger.warning("No audit trail found for process %s", state.process_id)
            return []
    
        # Group by agent to detect repeated failures
        failure_counts = {}
    
        for entry in audit_trail:
            # Defensive: ensure entry is a dict
            if not isinstance(entry, dict):
                continue
    
            status = str(entry.get("status", "")).lower()
            error_message = entry.get("error_message")
            duration_ms = entry.get("duration_ms", 0)
            agent = entry.get("agent_name", "unknown")
    
            # Track failures for later aggregation
            if status == "failed":
                failure_counts[agent] = failure_counts.get(agent, 0) + 1
    
            # Identify anomalies
            anomaly_detected = (
                status == "failed"
                or bool(error_message)
                or duration_ms > 5000  # 5-second latency threshold
            )
    
            if anomaly_detected:
                reportable.append({
                    "process_id": state.process_id,
                    "agent_name": agent,
                    "timestamp": entry.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                    "status": status,
                    "duration_ms": duration_ms,
                    "error_message": error_message,
                    "details": entry.get("details", {}),
                    "anomaly_reason": (
                        "Failure"
                        if status == "failed"
                        else "High latency"
                        if duration_ms > 5000
                        else "Error message logged"
                    ),
                })
    
        # Add summary-level anomaly if multiple failures detected
        for agent, count in failure_counts.items():
            if count > 2:
                reportable.append({
                    "process_id": state.process_id,
                    "agent_name": agent,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "status": "repeated_failures",
                    "details": {"failure_count": count},
                    "anomaly_reason": f"{count} repeated failures detected for {agent}",
                })
    
        # Log summary for visibility
        if reportable:
            self.logger.logger.info(
                "Detected %d reportable events for process %s",
                len(reportable),
                state.process_id,
            )
        else:
            self.logger.logger.debug("No reportable events found for process %s", state.process_id)
    
        # Attach to state for traceability
        setattr(state, "reportable_events", reportable)
        state.updated_at = datetime.utcnow()
    
        return reportable


    async def _generate_audit_alerts(
        self,
        state: InvoiceProcessingState,
        reportable_events: List[Dict[str, Any]]
    ) -> None:
        """
        Generate and dispatch alerts for detected audit anomalies.
        Alerts are categorized based on severity (warning or critical)
        and logged for traceability. Optionally integrates with external
        alerting channels (e.g., Slack, PagerDuty, email).
        """
        if not reportable_events:
            self.logger.logger.debug("No audit alerts to generate for process %s", state.process_id)
            return
    
        self.logger.logger.warning(
            "[AuditSystem] %d reportable audit events detected for process %s",
            len(reportable_events),
            state.process_id,
        )
    
        alerts_summary = []
        critical_events = 0
    
        for event in reportable_events:
            agent = event.get("agent_name", "unknown")
            reason = event.get("anomaly_reason", "unspecified")
            status = str(event.get("status", "")).lower()
            duration = event.get("duration_ms", 0)
            timestamp = event.get("timestamp", datetime.utcnow().isoformat() + "Z")
    
            # Classify severity
            severity = "critical" if status == "failed" or "repeated" in status else "warning"
            if severity == "critical":
                critical_events += 1
    
            alert_message = (
                f"[{severity.upper()} ALERT] Agent: {agent} | Reason: {reason} | "
                f"Status: {status} | Duration: {duration} ms | Time: {timestamp}"
            )
    
            # Log structured alert
            if severity == "critical":
                self.logger.logger.error(alert_message)
            else:
                self.logger.logger.warning(alert_message)
    
            alerts_summary.append({
                "severity": severity,
                "agent_name": agent,
                "reason": reason,
                "status": status,
                "duration_ms": duration,
                "timestamp": timestamp,
            })
    
            # Optionally send to external alerting channels (mocked)
            try:
                await self._send_alert_notification(alerts_summary[-1])
            except Exception as e:
                self.logger.logger.error(f"Failed to dispatch alert notification: {e}")
    
        # Attach alerts summary to state for later review
        setattr(state, "audit_alerts", alerts_summary)
        state.updated_at = datetime.utcnow()
    
        # Log summary
        self.logger.logger.info(
            "Audit alert generation completed: %d total (%d critical)",
            len(alerts_summary),
            critical_events,
        )

    def _record_execution(self, success: bool, duration_ms: float, state: Optional[InvoiceProcessingState] = None):
        compliance = getattr(state, "compliance_report", {}) if state else {}
        compliant_flags = [
            compliance.get("sox_compliance") == "compliant",
            compliance.get("gdpr_compliance") == "compliant",
            compliance.get("financial_controls") in ("passed", "compliant")
        ]
        compliance_score = round((sum(compliant_flags) / len(compliant_flags)) * 100, 2) if compliant_flags else 0
    
        self.execution_history.append({
            # "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "duration_ms": duration_ms,
            "compliance_score": compliance_score,
            "reportable_events": len(getattr(state, "reportable_events", [])) if state else 0,
        })
    
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)

    async def health_check(self) -> Dict[str, Any]:
        total_runs = len(self.execution_history)
        if total_runs == 0:
            return {
                "Agent": "Audit Agent 🧮",
                "Executions": 0,
                "Success Rate (%)": 0.0,
                "Avg Duration (ms)": 0.0,
                "Total Failures": 0,
                "Avg Compliance (%)": 0.0,
                "Avg Reportable Events": 0.0,
                "Status": "idle",
                # "Timestamp": datetime.utcnow().isoformat()
            }
    
        successes = sum(1 for e in self.execution_history if e["success"])
        failures = total_runs - successes
        avg_duration = round(mean(e["duration_ms"] for e in self.execution_history), 2)
        success_rate = round((successes / (total_runs+1e-8)) * 100, 2)
        avg_compliance = round(mean(e["compliance_score"] for e in self.execution_history), 2)
        avg_events = round(mean(e["reportable_events"] for e in self.execution_history), 2)
    
        # Dynamic health status logic
        print("self.execution_history------", self.execution_history)
        print(avg_compliance)
        if success_rate >= 85 and avg_compliance >= 90:
            overall_status = "🟢 Healthy"
        elif success_rate >= 60:
            overall_status = "🟠 Degraded"
        else:
            overall_status = "🔴 Unhealthy"
    
        return {
            "Agent": "Audit Agent 🧮",
            "Executions": total_runs,
            "Success Rate (%)": success_rate,
            "Avg Duration (ms)": avg_duration,
            "Total Failures": failures,
            "Avg Compliance (%)": avg_compliance,
            "Avg Reportable Events": avg_events,
            # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Overall Health": overall_status,
            "Last Run": self.metrics["last_run_at"],
        }
