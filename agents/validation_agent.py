
"""Validation Agent for Invoice Processing"""

# TODO: Implement agent
import asyncio
import os
import pandas as pd
from typing import Dict, Any, List, Tuple
from fuzzywuzzy import fuzz
import numpy as np
import time
from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ValidationResult, ValidationStatus,
    ProcessingStatus
)
from datetime import datetime, timedelta

from utils.logger import StructuredLogger
from difflib import SequenceMatcher

class ValidationAgent(BaseAgent):
    """Agent responsible for validating invoice data against purchase orders"""
    
    health_history: List[Dict[str, Any]] = []  # global history for metrics

    def __init__(self, config: Dict[str, Any] = None):
        # pass
        super().__init__(agent_name="validation_agent",config=config or {})
        self.logger = StructuredLogger(__name__)
        self.po_file = self.config.get("po_file","data/purchase_orders.csv")
        self.tolerance = self.config.get("tolerance",0.05)
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_duration = 0.0
        self.total_executions = 0
        self.last_run = None
        # self.match_threshold = self.config.get("match_threshold",80)

    def _validate_preconditions(self, state: InvoiceProcessingState, workflow_type) -> bool:
        # pass
        if not state.invoice_data:
            self.logger.logger.error("No invoice data available for validation.")
            return False
        return True

    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        # pass
        return hasattr(state,'validation_result') and state.validation_result is not None

    async def execute(self, state: InvoiceProcessingState, workflow_type) -> InvoiceProcessingState:
        # pass
        self.logger.logger.info(f"[ValidationAgent] Starting validation for {state.file_name}")
        start_time = time.time()
        try:
            if not self._validate_preconditions(state, workflow_type):
                state.status = ProcessingStatus.FAILED
                self._log_decision(state,"Validation Failed","Precondition not met",confidence = 0.0)
                return state
            invoice_data = state.invoice_data
            matching_pos = await self._find_matching_pos(invoice_data)
            validation_result = await self._validate_against_pos(invoice_data,matching_pos)
            state.validation_result = validation_result
            state.current_agent = "validation_agent"
            state.overall_status = ProcessingStatus.IN_PROGRESS
    
            if self._should_escalate_validation(validation_result, invoice_data):
                state.escalation_required = True
            self._validate_postconditions(state)
            self.successful_executions += 1
            self.last_run = datetime.utcnow().isoformat()
            # print("ValidationResult().confidence_score", state.validation_result.confidence_score)
            self._log_decision(
                state,
                "Validation Successful",
                "PDF text successfully validated and checked by AI",
                state.validation_result.confidence_score,
                state.process_id
            )
            return state
        except Exception as e:
            self.logger.logger.error(f"[ValidationAgent] Execution failed: {e}")
            self.failed_executions += 1
            state.overall_status = ProcessingStatus.FAILED
            return state

        finally:
            duration = (time.time() - start_time) * 1000  # ms
            self.total_executions += 1
            self.total_duration += duration
            self._record_health_metrics(duration)
    
    def _load_purchase_orders(self) -> pd.DataFrame:
        # pass
        """load po data from csv"""
        try:
            df = pd.read_csv(self.po_file)
            self.logger.logger.info(f"[ValidationAgent] Loaded {len(df)} purchase orders")
            return df
        except Exception as e:
            self.logger.logger.error(f"[ValidationAgent] failed to load purchase order: {e}")
            raise

    async def _find_matching_pos(self, invoice_data) -> List[Dict[str, Any]]:
        """find POs matching invoice order_id or fuzzy customer/items"""
        po_df = self._load_purchase_orders()
        matches = []
        for _,po in po_df.iterrows():
            customer_score = fuzz.token_sort_ratio(po["customer_name"], invoice_data.customer_name)
            order_id_score = fuzz.token_sort_ratio(po["order_id"], invoice_data.order_id)
            for item in invoice_data.item_details:
                item_score = fuzz.token_sort_ratio(po["item_name"],item.item_name)
                print(f"Compairing PO item {po['item_name']} with invoice item {item.item_name}: score = {item_score}")

            if (customer_score >= 80) and (item_score >=80) and (order_id_score >=80) and (po['invoice_number'] == int(invoice_data.invoice_number)):
                matches.append(po.to_dict())

        print("matches.....", matches)
        return matches


    async def _validate_against_pos(self, invoice_data, matching_pos: List[Dict[str, Any]]) -> ValidationResult:
        # pass

        if not matching_pos:
            return ValidationResult(po_found=False, validation_status='missing_po',validation_result='No matching purchase order found',
            discrepancies = [],
            confidence_score = 0.0)
        po_data = matching_pos[0]
        discrepancies = self._validate_item_against_po(invoice_data,po_data)
        discrepancies += self._validate_totals(invoice_data,po_data)
        actual_amount = [item.amount for item in invoice_data.item_details][0]
        actual_quantity = [item.quantity for item in invoice_data.item_details][0]
        actual_rate = [item.rate for item in invoice_data.item_details][0]
        amount_diff = abs(actual_amount - po_data.get('expected_amount',0))
        tolerance_limit = po_data.get('expected_amount',0)*self.tolerance
        amount_match = amount_diff <= tolerance_limit
        
        validation_result = ValidationResult(
            po_found=True,
            quantity_match=actual_quantity == po_data.get('quantity'),
            rate_match=abs(actual_rate - po_data.get('rate', 0)) <= tolerance_limit,
            amount_match=amount_match,
            validation_status=ValidationStatus.NOT_STARTED,  # temporary
            validation_result="; ".join(discrepancies) if discrepancies else "All checks passed",
            discrepencies=discrepancies,
            confidence_score=0.0,  # temporary
            expected_amount=po_data.get('amount'),
            po_data=po_data
        )
        validation_result.validation_status = self._determine_validation_status(validation_result)
        validation_result.confidence_score = self._calculate_validation_confidence(validation_result, matching_pos, invoice_data)
        return validation_result

    def _validate_item_against_po(self, item, po_data: Dict[str, Any]) -> List[str]:
        # pass
        # print("itemmmmmmmmm", item.item_details.quantity)
        print("po_-------------", po_data)
        discrepancies = []
        for item in item.item_details:
            if item.quantity != po_data.get('quantity'):
                discrepancies.append(f"Quantity mismatch: Expected {po_data['quantity']}, Found {item.quantity}")
            if abs(item.rate - po_data.get('rate',0)) > po_data.get('rate',0)*self.tolerance:
                discrepancies.append(f"Rate mismatch: Expected {po_data['rate']}, Found {item.rate}")
            return discrepancies

    def _validate_totals(self, invoice_data, po_data: Dict[str, Any]) -> List[str]:
        # pass
        discrepancies = []
        expected = po_data.get('expected_amount',0)
        actual = [item.amount for item in invoice_data.item_details][0]
        diff = abs(expected-actual)
        if diff > expected*self.tolerance:
            discrepancies.append(f"Total amount mismatch: Expected {expected}, Actual {actual} (Difference:{diff:.2f})")
        return discrepancies

    def _calculate_validation_confidence(self, validation_result: ValidationResult,
                                         matching_pos: List[Dict[str, Any]], invoice_data) -> float:
        """
        Compute an intelligent, weighted confidence score across 7 key dimensions:
        invoice_number, order_id, customer_name, item_name, amount, rate, quantity.
        Each field contributes based on importance.
        """
    
        if not validation_result.po_found or not matching_pos:
            return 0.0
    
        po_data = matching_pos[0]
    
        # Extract PO (expected) values
        expected = {
            "invoice_number": po_data.get("invoice_number", ""),
            "order_id": po_data.get("order_id", ""),
            "customer_name": po_data.get("customer_name", ""),
            "item_name": po_data.get("item_name", ""),
            "amount": float(po_data.get("expected_amount", po_data.get("amount", 0))),
            "rate": float(po_data.get("rate", 0)),
            "quantity": float(po_data.get("quantity", 0))
        }
    
        # Extract actual (from invoice)
        actual = {
            "invoice_number": invoice_data.invoice_number,
            "order_id": invoice_data.order_id,
            "customer_name": invoice_data.customer_name,
        }
    
        # Handle line-item level (assuming single dominant item)
        if invoice_data.item_details:
            item = invoice_data.item_details[0]
            actual.update({
                "item_name": item.item_name,
                "amount": float(item.amount or 0),
                "rate": float(item.rate or 0),
                "quantity": float(item.quantity or 0)
            })
    
        # Define weights intelligently (sum = 1)
        weights = {
            "invoice_number": 0.20,
            "order_id": 0.15,
            "customer_name": 0.05,
            "item_name": 0.05,
            "amount": 0.25,
            "rate": 0.15,
            "quantity": 0.15
        }
    
        # --- Similarity functions ---
        def numeric_similarity(expected_val, actual_val):
            if expected_val == 0:
                return 1.0 if actual_val == 0 else 0.0
            diff_ratio = abs(expected_val - actual_val) / (abs(expected_val) + 1e-6)
            return max(0.0, 1.0 - diff_ratio)
    
        def text_similarity(a, b):
            return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    
        # --- Compute weighted similarities ---
        weighted_scores = []
        for field, weight in weights.items():
            exp_val, act_val = expected.get(field), actual.get(field)
    
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                score = numeric_similarity(exp_val, act_val)
            else:
                score = text_similarity(exp_val, act_val)
    
            weighted_scores.append(weight * score)
    
        # Combine to final confidence
        confidence = sum(weighted_scores)
        confidence = round(confidence * 100, 2)  # convert to %
        confidence = max(0.0, min(confidence, 100.0))  # clamp 0–100
    
        self.logger.logger.debug(f"Validation Confidence (weighted): {confidence}%")
        return confidence



    def _determine_validation_status(self, validation_result: ValidationResult) -> ValidationStatus:
        """
        Determine the final validation status based on PO existence, discrepancies, and amount match.
        """
        if not validation_result.po_found:
            return ValidationStatus.MISSING_PO
    
        discrepancies_count = len(validation_result.discrepencies)
    
        if discrepancies_count == 0 and validation_result.amount_match:
            return ValidationStatus.VALID
    
        if validation_result.amount_match and discrepancies_count <= 2:
            return ValidationStatus.PARTIAL_MATCH
    
        return ValidationStatus.INVALID


    def _should_escalate_validation(self, validation_result: ValidationResult, invoice_data) -> bool:
        # pass
        return validation_result.validation_status in ['invalid','missing_po']

    def _record_health_metrics(self, duration: float):
        """Record the health metrics after each execution"""
        success_rate = (
            (self.successful_executions / self.total_executions) * 100
            if self.total_executions > 0 else 0
        )
        avg_duration = (
            self.total_duration / self.total_executions
            if self.total_executions > 0 else 0
        )

        metrics = {
            "Agent": "Validation Agent ✅",
            "Executions": self.total_executions,
            "Success Rate (%)": round(success_rate, 2),
            "Avg Duration (ms)": round(avg_duration, 2),
            "Total Failures": self.failed_executions,
            # "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        metrics_data = {}
        executions = 0
        success_rate = 0.0
        avg_duration = 0.0
        failures = 0
        last_run = None

        if self.metrics:
            print("self.metrics from validation agent", self.metrics)
            executions = self.metrics["processed"]
            print("executions.....", executions)
            avg_duration = self.metrics["avg_latency_ms"]
            failures = self.metrics["errors"]
            last_run = self.metrics["last_run_at"]
            print("last_run.....", last_run)
            success_rate = (executions - failures) / (executions + 1e-6)

        # if last_run == None:
        last_run = self.last_run

        # 3. Health logic
        overall_status = "🟢 Healthy"
        if failures > 3:
            overall_status = "🟠 Degraded"
        if executions > 0 and success_rate < 0.5:
            overall_status = "🔴 Unhealthy"
    
        print("metrics from val---....1", metrics)
    
        metrics.update({
            "Last Run": str(last_run) if last_run else "Not applicable",
            "Overall Health": overall_status,
        })
        print("metrics from val---....", metrics)
        # maintain up to last 50 records
        ValidationAgent.health_history.append(metrics)
        # ValidationAgent.health_history = ValidationAgent.health_history[-50:]

    async def health_check(self) -> Dict[str, Any]:
        """
        Returns the health metrics summary for UI display.
        """
        await asyncio.sleep(0.05)
        if not ValidationAgent.health_history:
            return {
                "Agent": "Validation Agent ✅",
                "Executions": 0,
                "Success Rate (%)": 0.0,
                "Avg Duration (ms)": 0.0,
                "Total Failures": 0,
            }


        latest = ValidationAgent.health_history[-1]
        print("latest.....", latest)
        return latest
