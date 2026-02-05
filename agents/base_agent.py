
"""Base Agent Class for Invoice Processing System"""

# TODO: Implement agent

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from state import InvoiceProcessingState, ProcessingStatus, AuditTrail
from utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all invoice processing agents"""

    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.logger = get_logger(agent_name)
        self.metrics: Dict[str,Any] = {
            "processed" : 0,
            "errors" : 0,
            "avg_latency_ms" : None,
            "last_run_at" : None
        }
        self.start_time: Optional[float] = None

    @abstractmethod
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        raise NotImplementedError

    async def run(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        self.start_time = time.time()
        self.logger.logger.info(f"Starting {self.agent_name} execution.")
        if not self._validate_preconditions(state, workflow_type):
            self.logger.logger.warning(f"Preconditions not met for {self.agent_name}.")
            self.metrics["processed"] = int(self.metrics.get("processed", 0)) + 1
            self.metrics["last_run_at"] = datetime.utcnow().isoformat()

            # optional but very good:
            state.add_agent_metric(self.agent_name, processed=1, latency_ms=0, errors=0)

            state.add_audit_entry(
                self.agent_name,
                "precondition_failed",
                {"note": "Preconditions not met, agent skipped."}
            )
            return state
        state.current_agent = self.agent_name
        state.agent_name = self.agent_name
        state.overall_status = ProcessingStatus.IN_PROGRESS

        try:
            updated_state = await self.execute(state, workflow_type)

            try:
                self._validate_postconditions(updated_state)
            except Exception as post_exc:
                self.logger.logger.warning(f"Postcondition check raised for {self.agent_name}:{post_exc}")
            
            state.mark_agent_completed(self.agent_name)
            latency_ms = (time.time()-self.start_time)*1000
            self.metrics["processed"] = int(self.metrics.get("processed",0)) + 1
            prev_avg = self.metrics.get("avg_latency_ms")

            if prev_avg is None:
                self.metrics["avg_latency_ms"] = latency_ms
            else:
                self.metrics["avg_latency_ms"] = (prev_avg+latency_ms)/2.0

            self.metrics["last_run_at"] = datetime.utcnow().isoformat()
            print(
                f"Agent: {self.agent_name} | "
                f"id: {id(self)} | "
                f"last_run_at: {self.metrics['last_run_at']}"
            )

            print("self.metrics[last_run_at]", self.metrics["last_run_at"])
            state.add_agent_metric(self.agent_name,processed=1,latency_ms=latency_ms)
            state.add_audit_entry(self.agent_name, action="Agent Successfully Executed", status=ProcessingStatus.COMPLETED, details={"latency_ms":latency_ms}, process_id=state.process_id)
            
            self.logger.logger.info(f"{self.agent_name}completed successfully in {latency_ms:.2f}ms.")
            return updated_state
            
        except Exception as e:
            latency_ms = (time.time()-self.start_time)*1000 if self.start_time else 0.0
            # self._update_metrics(latency_ms=latency_ms,error=True)
            self.metrics["processed"] = int(self.metrics.get("processed",0))+1
            self.metrics["errors"] = int(self.metrics.get("errors",0))+1
            prev_avg = self.metrics.get("avg_latency_ms")

            if prev_avg is None:
                self.metrics["avg_latency_ms"] = latency_ms
            else:
                self.metrics["avg_latency_ms"] = (prev_avg+latency_ms)/2.0
            self.metrics["last_run_at"] = datetime.utcnow().isoformat() 
            state.add_agent_metric(self.agent_name, processed = 1, latency_ms = latency_ms, errors = 1)
            state.add_audit_entry(self.agent_name,"Error in Execution",{"error":str(e)})
            state.overall_status = ProcessingStatus.FAILED
            self.logger.logger.exception(f"{self.agent_name} failed: {e}")
            return state

    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        "override to add custom preconditions for agent execution"
        return True

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        "override to verify expected outcomes after agent execution"
        return True


    def get_metrics(self) -> Dict[str, Any]:
        # pass
        return dict(self.metrics)

    def reset_metrics(self):
        # pass
        self.metrics = {"processed":0,
                        "errors":0,
                        "avg_latency_ms":None,
                        "last_run_at":None}

    async def health_check(self) -> Dict[str, Any]:
        # pass
        """perform a basic health check for the agent"""
        return {
            "agent":self.agent_name,
            "status":"Healthy",
            "Last Run":self.metrics.get("last_run_at"),
            "errors":self.metrics.get("errors", 0)
        }

    def _extract_business_context(self, state: InvoiceProcessingState) -> Dict[str, Any]:
        # pass
        """Extract relevant invoice or PO context for resaoning logs"""
        context: Dict[str,Any] = {}
        if state.invoice_data:
            context["vendor"] = state.invoice_data.vendor_name
            context["invoice_id"] = state.invoice_data.invoice_id
            context["amount"] = state.invoice_data.total_amount
        if state.validation_result:
            try:
                context["validation_status"] = state.validation_result.validation_status.value
            except Exception:
                context["validation_status"] = str(state.validation_result.validation_status)
        if state.risk_assessment:
            context["risk_score"] = state.risk_assessment.risk_score
            context["risk_level"] = state.risk_assessment.risk_level.value if hasattr(state.risk_assessment.risk_level, "value") else str(state.risk_assessment.risk_level)
        return context


    def _should_escalate(self, state: InvoiceProcessingState, reason: str = None) -> bool:
        # pass
        """Determine whether the workflow should escalate."""
        try:
            result = state.requires_escalation()
        except Exception:
            result = True
        if result:
            self.logger.logger.warning(f"Escalation triggered by {self.agent_name}:{reason or 'auto'}")
            state.escalation_required = True
            state.human_review_required = True
            state.add_audit_entry(self.agent_name,"Escalation Triggered", None, {"reason":reason or "auto"})
        return result

    def _log_decision(self, state: InvoiceProcessingState, decision: str,
                     reasoning: str, confidence: float = None, process_id: str = None):
        # pass
        """Log and record an agent decision into audit trail."""
        details:Dict[str,Any] = {
            "decision":decision,
            "reasoning":reasoning,
            "confidence":confidence,
            # "timestamp":datetime.utcnow().isoformat()
        }
        self.logger.logger.info(f"{self.agent_name} decision:{decision}(confidence = {confidence})")
        state.add_audit_entry(self.agent_name, decision, None, details, process_id)

class AgentRegistry:
    """Registry for managing agent instances"""

    def __init__(self):
        # pass
        self._agents:Dict[str,BaseAgent] = {}

    def register(self, agent: BaseAgent):
        # pass
        if agent.agent_name in self._agents:
            print(f"{agent.agent_name} already registered - skipping")
            return
        self._agents[agent.agent_name] = agent

    def get(self, agent_name: str) -> Optional[BaseAgent]:
        # pass
        return self._agents.get(agent_name)

    def list_agents(self) -> List[str]:
        # pass
        return list(self._agents.keys())

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        # pass
        return {name:agent.get_metrics() for name, agent in self._agents.items()}

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        # pass
        result:Dict[str,Dict[str,Any]] = {}
        for name, agent in self._agents.items():
            result[name] = await agent.health_check()
        return result



# Global agent registry instance
agent_registry = AgentRegistry()
print("Registry instance ID in base:", id(agent_registry))
