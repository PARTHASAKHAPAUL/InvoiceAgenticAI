"""LangGraph workflow orchestrator"""
# TODO: Implement graph workflow

import asyncio
import uuid # extra import
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import (
    InvoiceProcessingState, ProcessingStatus, ValidationStatus,
    RiskLevel, PaymentStatus, WORKFLOW_CONFIGS
)
from agents.base_agent import agent_registry
from agents.document_agent import DocumentAgent
from agents.validation_agent import ValidationAgent
from agents.risk_agent import RiskAgent
from agents.payment_agent import PaymentAgent
from agents.audit_agent import AuditAgent
from agents.escalation_agent import EscalationAgent
from utils.logger import StructuredLogger


class InvoiceProcessingGraph:
    """Graph orchestrator"""

    def __init__(self, config: Dict[str, Any] = None):
        self.logger = StructuredLogger("InvoiceProcessingGraph")
        self.config = config or {}
        #Simple in-memory store for process states (process_id -> InvoiceProcessingState)
        self._process_store: Dict[str, InvoiceProcessingState] = {}
        #Register and initialize agents
        self._initialize_agents()
        try:
            self.graph = self._create_workflow_graph()
            self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
            self.logger.logger.info("InvoiceProcessingGraph initialized successfully with compiled graph.")
        except Exception as e:
            self.logger.logger.warning(f"Failed to fully build graph nodes: {e} — exposing empty StateGraph")
            self.graph = StateGraph("invoice_processing_graph_fallback")

    def _initialize_agents(self):
        """Instantiate and register agent instances in the global registry"""
        #create agent instances (idempotent - replace if already registered)
        agents = [
            DocumentAgent(),
            ValidationAgent(),
            RiskAgent(),
            PaymentAgent(),
            AuditAgent(),
            EscalationAgent(),
        ]
        for agent in agents:
            agent_registry.register(agent)
        self.logger.logger.info(f"Registered agents: {agent_registry.list_agents()}")

    def _create_workflow_graph(self) -> StateGraph:
        """
        Build a LangGraph StateGraph with conditional routing:
        Each node executes its corresponding agent and determines
        the next node based on runtime logic (risk, validation, etc.)
        """

        graph = StateGraph("invoice_processing_graph")

        # NODE DEFINITIONS 
        async def node_document(state: InvoiceProcessingState):
            state = await self._document_agent_node(state)
            next_node = self._route_after_document(state)
            return next_node, state

        async def node_validation(state: InvoiceProcessingState):
            state = await self._validation_agent_node(state)
            next_node = self._route_after_validation(state)
            return next_node, state

        async def node_risk(state: InvoiceProcessingState):
            state = await self._risk_agent_node(state)
            next_node = self._route_after_risk(state)
            return next_node, state

        async def node_payment(state: InvoiceProcessingState):
            state = await self._payment_agent_node(state)
            next_node = self._route_after_payment(state)
            return next_node, state

        async def node_audit(state: InvoiceProcessingState):
            state = await self._audit_agent_node(state)
            next_node = self._route_after_audit(state)
            return next_node, state

        async def node_escalation(state: InvoiceProcessingState):
            state = await self._escalation_agent_node(state)
            next_node = self._route_after_escalation(state)
            return next_node, state

        async def node_human_review(state: InvoiceProcessingState):
            state = await self._human_review_node(state)
            next_node = self._route_after_human_review(state)
            return next_node, state

        async def node_end(state: InvoiceProcessingState):
            self.logger.logger.info(f"Invoice {state.invoice_id} completed at {state.updated_at}")
            return "end", state

        # REGISTER NODES 
        for name, func in {
            "document": node_document,
            "validation": node_validation,
            "risk": node_risk,
            "payment": node_payment,
            "audit": node_audit,
            "escalation": node_escalation,
            "human_review": node_human_review,
            "end": node_end,
        }.items():
            try:
                graph.add_node(name, func)
            except Exception:
                # fallback if add_node signature differs
                setattr(graph, name, func)

        # ADD EDGES (DEFAULT PATHS)
        try:
            graph.add_edge("document", "validation")
            graph.add_edge("validation", "risk")
            graph.add_edge("risk", "payment")
            graph.add_edge("payment", "audit")
            graph.add_edge("audit", "end")
            # Alternative / exception flows
            graph.add_edge("document", "escalation")
            graph.add_edge("validation", "escalation")
            graph.add_edge("risk", "escalation")
            graph.add_edge("escalation", "human_review")
            graph.add_edge("human_review", "end")

            graph.set_entry_point("document")
        except Exception as ex:
            self.logger.logger.warning(f"Edge registration failed: {ex}")

        self.logger.logger.info("Conditional workflow graph built successfully.")
        return graph


    async def _document_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: DocumentAgent = agent_registry.get("document_agent")
        print("agent from doc", agent)
        if not agent:
            agent = DocumentAgent()
            agent_registry.register(agent)
        print("Registry instance ID in graph:", id(agent_registry))

        return await agent.run(state, workflow_type)

    async def _validation_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: ValidationAgent = agent_registry.get("validation_agent")
        print("agent from val", agent)
        if not agent:
            agent = ValidationAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    async def _risk_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: RiskAgent = agent_registry.get("risk_agent")
        if not agent:
            agent = RiskAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    async def _payment_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: PaymentAgent = agent_registry.get("payment_agent")
        if not agent:
            agent = PaymentAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    async def _audit_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: AuditAgent = agent_registry.get("audit_agent")
        if not agent:
            agent = AuditAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    async def _escalation_agent_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        agent: EscalationAgent = agent_registry.get("escalation_agent")
        if not agent:
            agent = EscalationAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    async def _human_review_node(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        #Reusing escalation agent's human-in-the-loop or simply marking for manual review
        agent: EscalationAgent = agent_registry.get("escalation_agent")
        if not agent:
            agent = EscalationAgent()
            agent_registry.register(agent)
        return await agent.run(state, workflow_type)

    def _route_after_document(self, state: InvoiceProcessingState) -> Literal["validation", "escalation", "end"]:
        """Route decision after document extraction"""
        #if extraction yielded no invoice_data or low confidence -> escalate
        if not state.invoice_data:
            return "escalation"
        #if extraction confidence exists and is low -> escalate
        conf = getattr(state.invoice_data, "extraction_confidence", None)
        if conf is not None and conf<0.6:
            return "escalation"
        return "validation"


    def _route_after_validation(self, state: InvoiceProcessingState) -> Literal["risk", "escalation", "end"]:
        """Route decision after document validation"""
        vr = state.validation_result
        if not vr:
            return "escalation"
        #if missing PO or invalid -> escalate
        try:
            status = vr.validation_status
            #ValidationStatus maybe enum or str
            if isinstance(status,ValidationStatus):
                status_val = status
            else:
                status_val = ValidationStatus(status) if isinstance(status,str) else None
            if status_val == ValidationStatus.NO_MATCH or status_val == ValidationStatus.PARTIAL_MATCH and (not vr.amount_match):
                return "escalation"
        except Exception:
            #fallback: if discrepancies exist -> escalation
            if vr and getattr(vr,"discrepancies",None):
                return "escalation"
        return "risk"

    def _route_after_risk(self, state: InvoiceProcessingState) -> Literal["payment", "escalation", "human_review", "end"]:
        """Route decision after risk assessment"""
        ra = state.risk_assessment
        if not ra:
            return "escalation"
        #ra.risk_level is an enum RiskLevel
        rl = getattr(ra,"risk_level",None)
        #handle strings or enums
        rl_val = rl.value if hasattr(rl,"value") else str(rl).lower()
        try:
            if rl_val in (RiskLevel.CRITICAL.value, RiskLevel.HIGH.value):
                #For critical-> human review; for high->escalate
                if rl_val == RiskLevel.CRITICAL.value:
                    return "human_review"
                return "escalation"
            else:
                #low or medium -> payment
                return "payment"
        except Exception:
            return "payment"

    def _route_after_payment(self, state: InvoiceProcessingState) -> Literal["audit", "escalation", "end"]:
        pd = getattr(state,"payment_decision",None)
        if not pd:
            return "escalation"
        #If approved (or scheduled) -> audit
        try:
            status = pd.payment_status
            #Accept enum or str
            status_val = status if isinstance(status,str) else getattr(status,"value",str(status))
            if status_val in (PaymentStatus.APPROVED.value, PaymentStatus.SCHEDULED.value, PaymentStatus.PENDING_APPROVAL.value):
                return "audit"
            else:
                return "escalation"
        except Exception:
            return "audit"

    def _route_after_audit(self, state: InvoiceProcessingState) -> Literal["escalation", "end"]:
        cr = getattr(state, "compliance_report",None)
        if not cr:
            return "end"
        #If any compliance issues ->escalate
        issues = cr.get("issues",{}) if isinstance(cr, dict) else {}
        has_issues = any(issues.get(k) for k in issues)
        return "escalation" if has_issues else "end"

    async def _handle_escalation_chain(self, state: "InvoiceProcessingState", workflow_type):
        """Common handler for escalation → human review → complete"""
        state = await self._escalation_agent_node(state, workflow_type)
        self._process_store[state.process_id] = state
        state = await self._human_review_node(state, workflow_type)
        state.overall_status = ProcessingStatus.COMPLETED
        self._process_store[state.process_id] = state
        return state

    async def process_invoice(self, file_name: str, workflow_type: str = "standard",
                          config: Dict[str, Any] = None) -> InvoiceProcessingState:
        """
        Orchestrate processing for a single invoice file.
        Supports 3 workflow types: standard, high_value, and expedited.
        """
        process_id = f"proc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        initial_state = InvoiceProcessingState(
            process_id=process_id,
            file_name=file_name,
            overall_status=ProcessingStatus.PENDING,
            current_agent=None,
            workflow_type=workflow_type,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
        self._process_store[process_id] = initial_state
        start_ts = datetime.utcnow()
        state = initial_state
        worked_agents = []
        try:
            # STEP 1️ Document Extraction
            state = await self._document_agent_node(state, workflow_type)
            self._process_store[process_id] = state
            route = self._route_after_document(state)
            print("state agent anme ::::::::::::::", state.agent_name)
            worked_agents.append(state.agent_name)
            if route == "escalation":
                state = await self._handle_escalation_chain(state, workflow_type)
                worked_agents.append(state.agent_name)
                return state, worked_agents
    
            # ---- Workflow branching ----
            if workflow_type == "expedited":
                # Fast lane - skip validation if AI confidence is high
                # if getattr(state, "extraction_confidence", 0.0) < 0.85:
                state = await self._validation_agent_node(state, workflow_type)
                self._process_store[process_id] = state
                worked_agents.append(state.agent_name)
                route = self._route_after_validation(state)
                if route == "escalation":
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # Directly go to Payment, minimal audit
                state = await self._payment_agent_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                self._process_store[process_id] = state
                if getattr(state, "payment_decision", {}).decision == "auto-pay":
                    state = await self._audit_agent_node(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    self._process_store[process_id] = state

            elif workflow_type == "high_value":
                # 2️ Validation (twice for accuracy)
                state = await self._validation_agent_node(state, workflow_type)
                self._process_store[process_id] = state
                state = await self._validation_agent_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                route = self._route_after_validation(state)
                if route == "escalation":
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # 3️ Risk
                state = await self._risk_agent_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                self._process_store[process_id] = state
                route = self._route_after_risk(state)
                if route in ["escalation", "human_review"]:
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # 4 Audit
                state = await self._audit_agent_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                self._process_store[process_id] = state

                # 5 Mandatory human review for high-value invoices
                state = await self._human_review_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                self._process_store[process_id] = state
    
            else:  # STANDARD workflow
                # 2️ Validation
                state = await self._validation_agent_node(state, workflow_type)
                self._process_store[process_id] = state
                worked_agents.append(state.agent_name)
                route = self._route_after_validation(state)
                if route == "escalation":
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # 3️ Risk
                state = await self._risk_agent_node(state, workflow_type)
                self._process_store[process_id] = state
                worked_agents.append(state.agent_name)
                route = self._route_after_risk(state)
                if route in ["escalation", "human_review"]:
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # 4️ Payment
                state = await self._payment_agent_node(state, workflow_type)
                self._process_store[process_id] = state
                worked_agents.append(state.agent_name)
                route = self._route_after_payment(state)
                if route == "escalation":
                    state = await self._handle_escalation_chain(state, workflow_type)
                    worked_agents.append(state.agent_name)
                    return state, worked_agents
    
                # 5️ Audit
                state = await self._audit_agent_node(state, workflow_type)
                worked_agents.append(state.agent_name)
                self._process_store[process_id] = state
    
            # Success completion
            state.overall_status = ProcessingStatus.COMPLETED
            state.updated_at = datetime.utcnow()
            elapsed = (datetime.utcnow() - start_ts).total_seconds()
            self.logger.logger.info(f"Process {process_id} ({workflow_type}) completed in {elapsed:.2f}s")
            self._process_store[process_id] = state
            # print("from graph worked agents::::", worked_agents)
            return state, worked_agents
    
        except Exception as e:
            self.logger.logger.exception(f"Error processing invoice {file_name}: {e}")
            state.overall_status = ProcessingStatus.FAILED
            self._process_store[process_id] = state
            return state, worked_agents


    # async def process_batch(self, file_names: List[str], workflow_type: str = "standard",
    #                       max_concurrent: int = 5) -> List[InvoiceProcessingState]:
    #     """Process a batch of files with limit concurrency"""
    #     sem = asyncio.Semaphore(max_concurrent)
    #     results: List[InvoiceProcessingState] = []

    #     async def _worker(fn: str):
    #         async with sem:
    #             return await self.process_invoice(fn, workflow_type=workflow_type)

    #     tasks = [asyncio.create_task(_worker(f)) for f in file_names]
    #     completed = await asyncio.gather(*tasks)
    #     for st in completed:
    #         results.append(st)
    #     return results
    async def process_batch(self, file_names: List[str], workflow_type: str = "standard",
                        max_concurrent: int = 5):
        sem = asyncio.Semaphore(max_concurrent)
        results = []   # will store: {"state": ..., "worked_agents": [...]}

        async def _worker(fn: str):
            async with sem:
                return await self.process_invoice(fn, workflow_type=workflow_type)

        tasks = [asyncio.create_task(_worker(f)) for f in file_names]
        completed = await asyncio.gather(*tasks)

        for result in completed:
            state, worked_agents = result  # unpack the tuple
            results.append({
                "state": state,
                "worked_agents": worked_agents
            })

        return results


    async def get_workflow_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored workflow status dictionary for a given process_id"""
        state = self._process_store.get(process_id)
        if not state:
            return None
        return {"process_id":process_id, "status":state.overall_status, "updated_at": getattr(state,"updated_at",None), "state":state.model_dump()}

    async def health_check(self) -> Dict[str, Any]:
        """Aggregate health check across agents and the orchestrator itself"""
        agents_health = await agent_registry.health_check_all()
        return {"orchestrator":"Healthy","timestamp":datetime.utcnow().isoformat(),"agent":agents_health}

    def _extract_final_state(self, result, initial_state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Compatibility helper (returns invoice processing state)"""
        return result


invoice_workflow: Optional[InvoiceProcessingGraph] = None

def get_workflow(config: Dict[str, Any] = None) -> InvoiceProcessingGraph:
    global invoice_workflow
    if invoice_workflow is None:
        invoice_workflow = InvoiceProcessingGraph(config=config)
    return invoice_workflow