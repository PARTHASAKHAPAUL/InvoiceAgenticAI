
"""State models and enumerations"""
# TODO: Define state models
from __future__ import annotations
import uuid  # extra import

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, root_validator
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"



class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PaymentStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PAID = "paid"
    FAILED = "failed"



class ItemDetail(BaseModel):
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    # description: Optional[str] = None
    quantity: int = Field(..., ge=0)
    rate: float = Field(..., ge=0.0)
    # total: Optional[float] = None
    # unit: Optional[str] = None
    amount: float = Field(..., ge=0.0)
    category: Optional[str] = None


class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    order_id: Optional[str] = None
    file_name: Optional[str] = None
    customer_name: Optional[str] = None
    invoice_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    currency: Optional[str] = "USD"
    total: Optional[float] = None
    # line_items: List[ItemDetail] = Field(default_factory=list)
    raw_text: Optional[str] = None
    item_details: Optional[list] = None
    # confidence_scores: Dict[str, float] = Field(default_factory=dict)
    extraction_confidence: Optional[float] = None

class ValidationStatus(str, Enum):
    NOT_STARTED = "not_started"
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL_MATCH = "partial_match"
    MISSING_PO = "missing_po"
    
class ValidationResult(BaseModel):
    po_found: bool = False
    quantity_match: bool = False
    rate_match: bool = False
    amount_match: bool = False
    validation_status: ValidationStatus = ValidationStatus.NOT_STARTED
    validation_result: Optional[str] = None
    discrepencies: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    # expected_amount: Optional[float] = None
    po_data: Optional[Dict[str, Any]] = None


class RiskAssessment(BaseModel):
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.LOW
    signals: List[str] = Field(default_factory=list)
    vendor_status: Optional[str] = None
    compliance_violations: List[str] = Field(default_factory=list)


class PaymentDecision(BaseModel):
    decision: Optional[Literal["auto_pay", "manual_approval", "hold", "reject"]]
    status: PaymentStatus = PaymentStatus.NOT_STARTED
    scheduled_date: Optional[datetime] = None
    transaction_id: Optional[str] = None
    attempts: int = 0
    reason: Optional[str] = None


class AuditTrail(BaseModel):
    process_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_name: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class AgentMetrics(BaseModel):
    processed_count: int = 0
    avg_latency_ms: Optional[float] = None
    last_run_at: Optional[datetime] = None
    errors: int = 0
    success_rate: Optional[float] = None


class InvoiceProcessingState(BaseModel):
    # Core identifiers
    process_id: str = Field(
        default_factory=lambda: f"proc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    file_name: Optional[str] = None

    # Processing status
    overall_status: ProcessingStatus = ProcessingStatus.PENDING
    current_agent: Optional[str] = None
    workflow_type: str = "Standard"

    # Agent Outputs
    invoice_data: Optional[InvoiceData] = None
    validation_result: Optional[ValidationResult] = None
    validation_status: Optional[str] = None
    risk_assessment: Optional[RiskAssessment] = None
    payment_decision: Optional[PaymentDecision] = None
    approval_chain: Optional[List[Dict[str, Any]]] = None

    # Audit and Tracking
    agent_name: Optional[str] = None
    audit_trail: List[AuditTrail] = Field(default_factory=list)
    agent_metrics: Dict[str, AgentMetrics] = Field(default_factory=dict)
    compliance_report: Optional[Dict[str, Any]] = None
    audit_summary: Optional[Dict[str, Any]] = None
    reportable_events: Optional[List[Dict[str, Any]]] = None

    # Escalation
    escalation_required: bool = False
    human_review_required: bool = False
    escalation_details: Optional[str] = None
    escalation_reason: Optional[str] = None

    # Workflow Control
    retry_count: int = 0
    completed_agents: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Convenience Methods
    def add_audit_entry(self, agent_name: str, action: str, status: Optional[ProcessingStatus] = None, details: Optional[Dict[str, Any]] = None, process_id: Optional[str] = None) -> None:
        entry = AuditTrail(agent_name=agent_name, action=action, status = status or self.overall_status, details=details or {}, process_id=process_id)
        print("entry.....", entry)
        print("self.audit_trail...", self.audit_trail)
        self.audit_trail.append(entry)
        self.updated_at = datetime.utcnow()

    def add_agent_metric(self, agent: str, processed: int = 0, latency_ms: Optional[float] = None, errors: int = 0) -> None:
        metrics = self.agent_metrics.get(agent) or AgentMetrics()
        metrics.processed_count += processed
        metrics.errors += errors

        if latency_ms is not None:
            if metrics.avg_latency_ms is None:
                metrics.avg_latency_ms = latency_ms
            else:
                metrics.avg_latency_ms = (metrics.avg_latency_ms + latency_ms) / 2.0
        metrics.last_run_at = datetime.utcnow()

        if metrics.processed_count > 0:
            metrics.success_rate = max(0.0, 1.0 - (metrics.errors / max(1, metrics.processed_count)))
        self.agent_metrics[agent] = metrics
        self.updated_at = datetime.utcnow()

    def update_agent_metrics(self, agent_name: str, success: bool, duration_ms: float):
        """
        Update or create performance metrics for an agent.
        Expected structure aligns with test_9_agent_metrics: attributes like executions, success_count, failure_count.
        """
        # Ensure agent_metrics dict exists
        if self.agent_metrics is None:
            self.agent_metrics = {}
    
        # Get or initialize metrics object
        metrics = self.agent_metrics.get(agent_name)
    
        # If existing metrics is a dict or None, replace with a new AgentMetrics-like object
        if isinstance(metrics, dict) or metrics is None:
            metrics = type("DynamicMetrics", (), {})()
            metrics.executions = 0
            metrics.successes = 0
            metrics.failure_count = 0
            metrics.total_duration_ms = 0.0
            metrics.avg_duration_ms = 0.0
    
        # Update fields
        metrics.executions += 1
        if success:
            metrics.successes += 1
        else:
            metrics.failure_count += 1
    
        metrics.total_duration_ms += duration_ms
        metrics.avg_duration_ms = round(metrics.total_duration_ms / metrics.executions, 2)
    
        # Save back
        self.agent_metrics[agent_name] = metrics
        self.updated_at = datetime.utcnow()


    def mark_agent_completed(self, agent: str) -> None:
        if agent not in self.completed_agents:
            self.completed_agents.append(agent)
            self.updated_at = datetime.utcnow()

    def requires_escalation(self, risk_threshold: float = 0.6, confidence_threshold: float = 0.7) -> bool:
        if self.validation_result and self.validation_result.validation_status == ValidationStatus.INVALID:
            return True
        if self.risk_assessment and self.risk_assessment.risk_score >= risk_threshold:
            return True
        if self.validation_result and self.validation_result.confidence_score is not None and self.validation_result.confidence_score < confidence_threshold:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @root_validator(pre=True)
    def ensure_timestamps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "created_at" not in values or values.get("created_at") is None:
            values["created_at"] = datetime.utcnow()
        if "updated_at" not in values or values.get("updated_at") is None:
            values["updated_at"] = datetime.utcnow()
        return values


class WorkflowConfig(BaseModel):
    name: str
    auto_approve_threshold: float = 0.3
    auto_approve_amount_limit: Optional[float] = None
    tolerance_percent: float = 5.0
    escalation_rules: Dict[str, Any] = Field(default_factory=dict)


WORKFLOW_CONFIGS: Dict[str, WorkflowConfig] = {
    "standard": WorkflowConfig(
        name="standard",
        auto_approve_threshold=0.3,
        auto_approve_amount_limit=10000.0,
        tolerance_percent=5.0,
        escalation_rules={"sla_hours": 24},
    ),
    "high_value": WorkflowConfig(
        name="high_value",
        auto_approve_threshold=0.1,
        auto_approve_amount_limit=5000.0,
        tolerance_percent=2.0,
        escalation_rules={"require_cfo": True, "sla_hours": 12},
    ),
    "expedited": WorkflowConfig(
        name="expedited",
        auto_approve_threshold=0.5,
        auto_approve_amount_limit=5000.0,
        tolerance_percent=10.0,
        escalation_rules={"skip_manual_for_low_risk": True},
    ),
}
