
# agents/insights_agent.py
"""
Insight Agent
-------------
Generates analytical and visual insights from processed invoices.
"""

import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from state import InvoiceProcessingState


class InsightAgent:
    def __init__(self):
        pass

    def _extract_invoice_records(self, results: List[InvoiceProcessingState]) -> pd.DataFrame:
        """Extract flat invoice info for analysis"""
        records = []
        for r in results:
            if isinstance(r, dict):
                # Convert dict to InvoiceProcessingState if needed
                try:
                    r = InvoiceProcessingState(**r)
                except Exception:
                    continue

            inv = getattr(r, "invoice_data", None)
            risk = getattr(r, "risk_assessment", None)
            val = getattr(r, "validation_result", None)
            pay = getattr(r, "payment_decision", None)

            records.append({
                "file_name": getattr(inv, "file_name", None),
                "invoice_number": getattr(inv, "invoice_number", None),
                "customer_name": getattr(inv, "customer_name", None),
                "invoice_date": getattr(inv, "invoice_date", None),
                "total": getattr(inv, "total", None),
                "validation_status": getattr(val, "validation_status", None),
                "risk_score": getattr(risk, "risk_score", None),
                "risk_level": getattr(risk, "risk_level", None),
                "payment_status": getattr(pay, "status", None),
                "decision": getattr(pay, "decision", None),
            })

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame()

        # Clean up data
        df["customer_name"] = df["customer_name"].fillna("Unknown Vendor")
        df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
        return df

    def generate_insights(self, results: List[InvoiceProcessingState]) -> Dict[str, Any]:
        """Generate charts and textual summary."""
        df = self._extract_invoice_records(results)
        if df.empty:
            return {"summary": "No data available for insights.", "charts": []}

        charts = []

        # 🔹 Total spend per customer
        if "customer_name" in df.columns:
            spend_chart = px.bar(
                df.groupby("customer_name", as_index=False)["total"].sum(),
                x="customer_name",
                y="total",
                title="Total Spend per Customer"
            )
            charts.append(spend_chart)

        # 🔹 Risk distribution
        if "risk_level" in df.columns:
            risk_chart = px.pie(
                df,
                names="risk_level",
                title="Risk Level Distribution"
            )
            charts.append(risk_chart)

        # 🔹 Validation status counts
        if "validation_status" in df.columns:
            val_chart = px.bar(
                df.groupby("validation_status", as_index=False).size(),
                x="validation_status",
                y="size",
                title="Validation Status Overview"
            )
            charts.append(val_chart)

        # 🔹 Summary text
        total_spend = df["total"].sum()
        high_risk = (df["risk_score"] >= 0.7).sum()
        valid_invoices = (df["validation_status"].astype(str).str.lower() == "valid").sum()

        summary = (
            f"💰 **Total Spend:** ₹{total_spend:,.2f}\n\n"
            f"📄 **Invoices Processed:** {len(df)}\n\n"
            f"✅ **Valid Invoices:** {valid_invoices}\n\n"
            f"⚠️ **High Risk Invoices:** {high_risk}\n\n"
        )

        return {"summary": summary, "charts": charts}
