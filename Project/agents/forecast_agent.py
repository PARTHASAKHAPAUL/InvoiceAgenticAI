
# agents/forecast_agent.py
"""
Forecast Agent (robust)
- Accepts a list of invoice states (dicts or InvoiceProcessingState models).
- Produces monthly historical spend and a simple forecast (moving average).
- Performs lightweight anomaly detection.
- Returns a dict containing a Plotly chart and numeric summary.
"""
from typing import List, Dict, Any, Union
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import os

# keep the type import only for hints; we do NOT require reconstructing models
try:
    from state import InvoiceProcessingState
except Exception:
    InvoiceProcessingState = None  # type: ignore


class ForecastAgent:
    def __init__(self):
        pass

    # ---- Internal: normalize input states -> DataFrame ----
    def _normalize_states_to_df(self, states: List[Union[dict, object]]) -> pd.DataFrame:
        """
        Accepts list of dicts or model instances.
        Produces a cleaned DataFrame with columns:
        ['file_name','invoice_date','due_date','total','vendor','risk_score','status']
        """
        rows = []
        for s in states:
            try:
                # 1) obtain a plain dict representation without constructing pydantic models
                if isinstance(s, dict):
                    raw = dict(s)
                else:
                    # model-like object: try model_dump, to_dict, or __dict__
                    if hasattr(s, "model_dump"):
                        raw = s.model_dump(exclude_none=False)
                    elif hasattr(s, "dict"):
                        raw = s.dict()
                    else:
                        # best effort: convert attributes to dict
                        raw = {
                            k: getattr(s, k)
                            for k in dir(s)
                            if not k.startswith("_") and not callable(getattr(s, k))
                        }

                # 2) sanitize well-known problematic fields that break pydantic elsewhere
                if "human_review_required" in raw and isinstance(raw["human_review_required"], str):
                    v = raw["human_review_required"].strip().lower()
                    raw["human_review_required"] = v in ("true", "yes", "1", "required")
                if "escalation_details" in raw and isinstance(raw["escalation_details"], dict):
                    # convert to string summary so downstream code doesn't expect a dict
                    try:
                        raw["escalation_details"] = str(raw["escalation_details"])
                    except Exception:
                        raw["escalation_details"] = ""

                # 3) pull invoice_data safely (may be None, dict, or model)
                inv = {}
                if raw.get("invoice_data") is None:
                    inv = {}
                else:
                    inv_raw = raw.get("invoice_data")
                    if isinstance(inv_raw, dict):
                        inv = dict(inv_raw)
                    else:
                        # model-like invoice_data
                        if hasattr(inv_raw, "model_dump"):
                            inv = inv_raw.model_dump(exclude_none=False)
                        elif hasattr(inv_raw, "dict"):
                            inv = inv_raw.dict()
                        else:
                            # fallback: read attributes
                            inv = {
                                k: getattr(inv_raw, k)
                                for k in dir(inv_raw)
                                if not k.startswith("_") and not callable(getattr(inv_raw, k))
                            }

                # 4) turnout the row items we care about
                total = inv.get("total") or inv.get("amount") or raw.get("total") or 0.0
                # risk may be under risk_assessment.risk_score or top-level
                risk_src = raw.get("risk_assessment") or {}
                if isinstance(risk_src, dict):
                    risk_score = risk_src.get("risk_score") or 0.0
                else:
                    # model-like risk_assessment
                    if hasattr(risk_src, "model_dump"):
                        try:
                            risk_score = risk_src.model_dump().get("risk_score", 0.0)
                        except Exception:
                            risk_score = 0.0
                    else:
                        risk_score = getattr(risk_src, "risk_score", 0.0)

                # dates: prefer due_date then invoice_date - they could be strings or datetimes
                due = inv.get("due_date") or inv.get("invoice_date") or raw.get("due_date") or raw.get("invoice_date")
                vendor = inv.get("customer_name") or inv.get("vendor_name") or raw.get("vendor") or raw.get("customer_name") or "Unknown"
                file_name = inv.get("file_name") or raw.get("file_name") or "unknown"

                rows.append(
                    {
                        "file_name": file_name,
                        "due_date": due,
                        "invoice_date": inv.get("invoice_date") or raw.get("invoice_date"),
                        "total": total,
                        "vendor": vendor,
                        "risk_score": risk_score,
                        "status": raw.get("overall_status") or inv.get("status") or "unknown",
                    }
                )
            except Exception:
                # skip malformed state
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # coerce and normalize
        df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        # if due_date missing, fallback to invoice_date
        df["date"] = df["due_date"].fillna(df["invoice_date"])
        df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
        df["vendor"] = df["vendor"].fillna("Unknown")
        return df

    # ---- Public: predict monthly cashflow and return a plotly chart ----
    def predict_cashflow(self, states: List[Union[dict, object]], months: int = 6) -> Dict[str, Any]:
        """
        Produces a monthly historical spend + simple forecast for `months` into the future.
        Returns:
        {
            "chart": plotly_figure,
            "average_monthly_spend": float,
            "total_forecast": float,
            "forecast_values": {month_str: float, ...},
            "historical": pandas.Series,
            "forecast_start_month": str,
            "forecast_end_month": str
        }
        """
        df = self._normalize_states_to_df(states)
        if df.empty or df["date"].dropna().empty:
            return {"message": "No data to forecast", "chart": None}

        # create monthly buckets (period start)
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly_hist = df.groupby("month")["total"].sum().sort_index()

        # compute average monthly spend from available historical months
        average_month = float(monthly_hist.mean()) if not monthly_hist.empty else 0.0

        # build forecast months (next `months` starting from the next month after last historical)
        last_hist_month = monthly_hist.index.max()
        if pd.isnull(last_hist_month):
            start_month = pd.Timestamp.now().to_period("M").to_timestamp()
        else:
            # next month
            start_month = (last_hist_month + pd.offsets.MonthBegin(1)).normalize()

        forecast_index = pd.date_range(start=start_month, periods=months, freq="MS")
        # simple forecast: repeat the historical mean (interpretable and safe)
        forecast_vals = [average_month for _ in range(len(forecast_index))]

        # build plot dataframe (historical + forecast)
        hist_df = monthly_hist.reset_index().rename(columns={"month": "date", "total": "amount"})
        hist_df["type"] = "Historical"
        fc_df = pd.DataFrame({"date": forecast_index, "amount": forecast_vals})
        fc_df["type"] = "Forecast"
        plot_df = pd.concat([hist_df, fc_df], ignore_index=True).sort_values("date")

        # prepare a plotly figure with clear styling
        fig = go.Figure()
        # historical - solid line
        hist_plot = plot_df[plot_df["type"] == "Historical"]
        if not hist_plot.empty:
            fig.add_trace(go.Scatter(
                x=hist_plot["date"],
                y=hist_plot["amount"],
                mode="lines+markers",
                name="Historical Spend",
                line=dict(dash="solid"),
            ))
        # forecast - dashed line
        fc_plot = plot_df[plot_df["type"] == "Forecast"]
        if not fc_plot.empty:
            fig.add_trace(go.Scatter(
                x=fc_plot["date"],
                y=fc_plot["amount"],
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash"),
                marker=dict(symbol="circle-open")
            ))

        fig.update_layout(
            title="Monthly Spend (Historical + Forecast)",
            xaxis_title="Month",
            yaxis_title="Total Spend (USD)",
            hovermode="x unified",
            template="plotly_dark",
        )

        forecast_series = pd.Series(forecast_vals, index=[d.strftime("%Y-%m") for d in forecast_index])
        total_forecast = float(forecast_series.sum())

        result = {
            "chart": fig,
            "average_monthly_spend": round(average_month, 2),
            "total_forecast": round(total_forecast, 2),
            "forecast_values": forecast_series.to_dict(),
            "historical": monthly_hist,
            "forecast_start_month": forecast_index[0].strftime("%Y-%m"),
            "forecast_end_month": forecast_index[-1].strftime("%Y-%m"),
        }
        return result

    # ---- Public: detect anomalies on sanitized data ----
    def detect_anomalies(self, states: List[Union[dict, object]]) -> pd.DataFrame:
        """
        Returns DataFrame of anomalies:
         - total > 2 * mean(total)
         - OR risk_score >= 0.7
        Columns returned: ['file_name','date','vendor','total','risk_score','anomaly_reason']
        """
        df = self._normalize_states_to_df(states)
        if df.empty:
            return pd.DataFrame()

        mean_spend = df["total"].mean()
        cond = (df["total"] > mean_spend * 2) | (df["risk_score"] >= 0.7)
        anomalies = df.loc[cond, ["file_name", "date", "vendor", "total", "risk_score"]].copy()
        if anomalies.empty:
            return pd.DataFrame()
        anomalies = anomalies.rename(columns={"date": "invoice_date"})
        anomalies["anomaly_reason"] = anomalies.apply(
            lambda r: "High Spend" if r["total"] > mean_spend * 2 else "High Risk",
            axis=1,
        )
        return anomalies.reset_index(drop=True)
