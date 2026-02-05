
"""
Smart Explainer Agent (Enhanced + Gemini-powered)
- Produces a detailed, human-readable explanation for a single InvoiceProcessingState.
- Uses Gemini for natural summarization if API key is present.
- Defensive, HTML-enhanced, and fully dashboard-ready.
"""

from state import InvoiceProcessingState, ValidationStatus, PaymentStatus, RiskLevel
from datetime import datetime
import google.generativeai as genai
import json
import os


class SmartExplainerAgent:
    def __init__(self):
        # Configure Gemini only if available
        self.api_key = os.environ.get("GEMINI_API_KEY_4")
        self.use_gemini = bool(self.api_key)
        if self.use_gemini:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")

    # ---------- Helper functions ----------
    def _safe_invoice_dict(self, state: InvoiceProcessingState) -> dict:
        if not state or not getattr(state, "invoice_data", None):
            return {}
        return (
            state.invoice_data.model_dump(exclude_none=True)
            if hasattr(state.invoice_data, "model_dump")
            else state.invoice_data.dict()
        )

    def _safe_validation(self, state: InvoiceProcessingState) -> dict:
        if not state or not getattr(state, "validation_result", None):
            return {}
        return (
            state.validation_result.model_dump(exclude_none=True)
            if hasattr(state.validation_result, "model_dump")
            else state.validation_result.dict()
        )

    def _safe_risk(self, state: InvoiceProcessingState) -> dict:
        if not state or not getattr(state, "risk_assessment", None):
            return {}
        return (
            state.risk_assessment.model_dump(exclude_none=True)
            if hasattr(state.risk_assessment, "model_dump")
            else state.risk_assessment.dict()
        )

    # ---------- Core explain logic ----------
    def explain(self, state) -> str:
        """
        Generate a detailed HTML + markdown explanation for a given invoice.
        Falls back gracefully if data or Gemini is unavailable.
        """
    
        # --- Defensive normalization ---
        if state is None:
            return "<p>⚠️ No invoice state provided.</p>"
    
        if isinstance(state, dict):
            try:
                state = InvoiceProcessingState(**state)
            except Exception:
                pass
    
        # --- Extract fields safely ---
        invoice = self._safe_invoice_dict(state) or {}
        validation = self._safe_validation(state) or {}
        risk = self._safe_risk(state) or {}
        payment = (
            state.payment_decision.model_dump(exclude_none=True)
            if getattr(state, "payment_decision", None)
            and hasattr(state.payment_decision, "model_dump")
            else getattr(state, "payment_decision", {}) or {}
        )
    
        discrepancies = validation.get("discrepencies", [])  # per schema
    
        inv_id = invoice.get("invoice_number") or invoice.get("file_name") or "<unknown>"
        vendor = invoice.get("customer_name") or invoice.get("vendor_name") or "Unknown"
        total = invoice.get("total") or invoice.get("amount") or 0
    
        status = getattr(state, "overall_status", "unknown")
        status_val = status.value if hasattr(status, "value") else str(status)
    
        # --- Interpret status fields ---
        risk_level = risk.get("risk_level")
        if hasattr(risk_level, "value"):
            risk_level = risk_level.value
        risk_score = risk.get("risk_score", 0) or 0.0
    
        val_status = validation.get("validation_status")
        if hasattr(val_status, "value"):
            val_status = val_status.value
    
        payment_status = payment.get("status")
        if hasattr(payment_status, "value"):
            payment_status = payment_status.value
    
        # --- Badge colors ---
        colors = {
            "VALIDATION": "#ffc107",
            "RISK": (
                "#ff1744" if str(risk_level).lower() == "critical"
                else "#ff9800" if str(risk_level).lower() == "medium"
                else "#4caf50"
            ),
            "PAYMENT": "#4caf50",
            "AUDIT": "#2196f3",
        }
    
        # --- Header layout ---
        header_html = f"""
        <div style="display:flex;justify-content:center;margin-bottom:1rem;">
            <div style="flex:1;text-align:center;padding:0.8rem;
                        border-radius:10px;background:{colors['VALIDATION']};
                        color:white;margin:0 4px;">
                <b>Validation</b>
            </div>
            <div style="flex:1;text-align:center;padding:0.8rem;
                        border-radius:10px;background:{colors['RISK']};
                        color:white;margin:0 4px;">
                <b>Risk</b>
            </div>
            <div style="flex:1;text-align:center;padding:0.8rem;
                        border-radius:10px;background:{colors['PAYMENT']};
                        color:white;margin:0 4px;">
                <b>Payment</b>
            </div>
            <div style="flex:1;text-align:center;padding:0.8rem;
                        border-radius:10px;background:{colors['AUDIT']};
                        color:white;margin:0 4px;box-shadow:0 0 10px rgba(0,255,0,0.7);">
                <b>Audit</b>
            </div>
        </div>
        """
    
        # --- Formatter ---
        def _fmt(val):
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return f"${val:,.2f}"
            return str(val)
    
        # --- Base explanation (structured) ---
        lines = [
            f"<p><b>Invoice:</b> {inv_id}</p>",
            f"<p><b>Vendor:</b> {vendor}</p>",
            f"<p><b>Amount:</b> {_fmt(total)}</p>",
            f"<p><b>Status:</b> {status_val}</p>",
            "<hr>",
            f"<p><b>Validation:</b> {val_status or 'unknown'}</p>",
            f"<p><b>Risk Level:</b> {risk_level or 'low'} ({risk_score})</p>",
            f"<p><b>Payment:</b> {payment.get('decision', 'N/A')} ({payment_status or 'pending'})</p>",
        ]
    
        if discrepancies:
            lines.append("<p><b>Discrepancies Found:</b></p><ul>")
            for d in discrepancies:
                field = d.get("field", "unknown")
                expected = d.get("expected", "")
                actual = d.get("actual", "")
                lines.append(f"<li>{field}: expected <code>{expected}</code>, got <code>{actual}</code></li>")
            lines.append("</ul>")
    
        # --- Recommendations ---
        advice = []
        if str(val_status).lower() == "invalid":
            advice.append("❌ Invoice failed validation — requires manual review.")
        elif str(val_status).lower() in ("partial", "partial_match"):
            advice.append("⚠️ Partial validation — check mismatched fields.")
        if str(risk_level).lower() == "critical":
            advice.append("🚨 Critical risk detected — immediate escalation required.")
        elif str(risk_level).lower() == "medium":
            advice.append("⚠️ Medium risk — consider manual review.")
        if not advice:
            advice.append("✅ No major issues detected. Proceed as usual.")
    
        lines.append("<p><b>Recommendation:</b></p><ul>")
        for a in advice:
            lines.append(f"<li>{a}</li>")
        lines.append("</ul>")
    
        explanation_html = header_html + "\n".join(lines)
    
        # --- Gemini polishing (using your API key) ---
        if self.use_gemini:
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel("models/gemini-2.0-flash")
    
                prompt = f"""
    You are a professional financial analyst.
    Here is structured invoice data and an auto-generated explanation.
    
    Invoice summary:
    {json.dumps(invoice, indent=2)}
    
    Validation details: {json.dumps(validation, indent=2)}
    Risk assessment: {json.dumps(risk, indent=2)}
    Payment info: {json.dumps(payment, indent=2)}
    
    Rewrite the following explanation to sound executive-level, clear, and concise.
    Use HTML for sections but do not remove any factual details.
    
    Existing summary:
    {explanation_html}
    """
                response = model.generate_content(prompt)
                if response and getattr(response, "text", None):
                    return response.text.strip()
            except Exception as e:
                return explanation_html + f"<p><i>Gemini explanation failed: {e}</i></p>"
    
        return explanation_html
