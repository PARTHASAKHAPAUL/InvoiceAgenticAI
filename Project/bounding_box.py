
import fitz  # PyMuPDF
import pandas as pd
import os
import re
 
# === File paths ===
DATA_DIR = os.path.join(os.getcwd(), "data")
PDF_PATH = os.path.join(DATA_DIR, "invoices/Invoice-26.pdf")  # Update for new PDF if needed
CSV_PATH = os.path.join(DATA_DIR, "purchase_orders.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "annotated_invoice.pdf")
 
# === Field coordinate map (from your data) ===
FIELD_BOXES = {
    "invoice_number": (525, 55, 575, 75),
    "order_id": (45, 470, 230, 490),
    "customer_name": (40, 135, 100, 155),
    "quantity": (370, 235, 385, 250),
    "rate": (450, 235, 500, 250),
    "expected_amount": (520, 360, 570, 375),
}
 
# === Step 1: Open PDF and extract text ===
pdf = fitz.open(PDF_PATH)
page = pdf[0]
pdf_text = page.get_text()
 
# === Step 2: Helper to extract fields ===
def extract_field(pattern, text, group=1):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group).strip() if match else None
 
# Extract key identifiers
invoice_number_pdf = extract_field(r"#\s*(\d+)", pdf_text)
order_id_pdf = extract_field(r"Order ID\s*[:\-]?\s*(\S+)", pdf_text)
customer_name_pdf = extract_field(r"Bill To:\s*(.*)", pdf_text)
 
# === Step 3: Read CSV and match correct row ===
po_df = pd.read_csv(CSV_PATH)
 
matched_row = po_df[
    (po_df['invoice_number'].astype(str) == str(invoice_number_pdf))
    | (po_df['order_id'] == order_id_pdf)
]
 
if matched_row.empty:
    raise ValueError(f"No matching CSV row found for Invoice {invoice_number_pdf} / Order {order_id_pdf}")
 
expected = matched_row.iloc[0].to_dict()
expected = {k.lower(): str(v).strip() for k, v in expected.items()}
 
print("✅ Loaded expected data from CSV for this PDF:")
for k, v in expected.items():
    print(f"   {k}: {v}")
 
# === Step 4: Extract fields from PDF ===
invoice_data = {
    "invoice_number": invoice_number_pdf,
    "customer_name": customer_name_pdf,
    "order_id": order_id_pdf,
}
 
# Numeric fields
amounts = re.findall(r"\$?([\d,]+\.\d{2})", pdf_text)
invoice_data["expected_amount"] = amounts[-1] if amounts else None
 
# Extract first item (quantity, rate)
item_lines = re.findall(
    r"([A-Za-z0-9 ,\-]+)\s+(\d+)\s+\$?([\d,]+\.\d{2})\s+\$?([\d,]+\.\d{2})",
    pdf_text,
)
if item_lines:
    invoice_data["quantity"] = item_lines[0][1]
    invoice_data["rate"] = item_lines[0][2]
 
print("\n✅ Extracted data from PDF:")
for k, v in invoice_data.items():
    print(f"   {k}: {v}")
 
# === Step 5: Compare PDF vs CSV ===
discrepancies = []
 
def add_discrepancy(field, expected_val, found_val):
    discrepancies.append({"field": field, "expected": expected_val, "found": found_val})
 
# Compare string fields
for field in ["invoice_number", "order_id", "customer_name"]:
    if str(invoice_data.get(field, "")).strip() != str(expected.get(field, "")).strip():
        add_discrepancy(field, expected.get(field, ""), invoice_data.get(field, ""))
 
# Compare numeric fields
for field in ["quantity", "rate", "expected_amount"]:
    try:
        found_val = float(str(invoice_data.get(field, 0)).replace(",", "").replace("$", ""))
        expected_val = float(str(expected.get(field, 0)).replace(",", "").replace("$", ""))
        if round(found_val, 2) != round(expected_val, 2):
            add_discrepancy(field, expected_val, found_val)
    except:
        if str(invoice_data.get(field, "")) != str(expected.get(field, "")):
            add_discrepancy(field, expected.get(field, ""), invoice_data.get(field, ""))
 
# === Step 6: Annotate mismatched fields using fixed coordinates ===
for d in discrepancies:
    field = d["field"]
    if field not in FIELD_BOXES:
        print(f"⚠️ No coordinates found for field '{field}' — skipping annotation.")
        continue
 
    rect_coords = FIELD_BOXES[field]
    rect = fitz.Rect(rect_coords)
    expected_text = (
        f"{float(d['expected']):,.2f}"
        if field in ["quantity", "rate", "expected_amount"]
        else str(d["expected"])
    )
 
    # Draw red bounding box
    page.draw_rect(rect, color=(1, 0, 0), width=1.5)
 
    # Add expected value below box
    page.insert_text(
        (rect.x0, rect.y1 + 10),
        expected_text,
        fontsize=9,
        color=(1, 0, 0),
    )
 
pdf.save(OUTPUT_PATH)
pdf.close()
 
print("\n✅ Annotated invoice saved at:", OUTPUT_PATH)
 
if discrepancies:
    print("\n⚠️ Mismatches found:")
    for d in discrepancies:
        print(f" - {d['field']}: expected {d['expected']}, found {d['found']}")
else:
    print("\n✅ No mismatches found! Invoice matches CSV.")
