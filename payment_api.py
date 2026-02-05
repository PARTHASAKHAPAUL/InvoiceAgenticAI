"""Payment API service"""
# TODO: Implement payment endpoints

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
import random
import uvicorn
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Invoice Payment API",
    description="Payment processing simulation for invoice automation",
    version="1.0.0"
)

TRANSACTIONS: Dict[str, Dict[str, Any]] = {}
PAYMENT_METRICS = {
    'total_transactions': 0,
    'successful_payments': 0,
    'failed_payments': 0,
    'cancelled_payments': 0,
    'last_transaction_time': None
}


class PaymentRequest(BaseModel):
    order_id: str
    customer_name: str
    amount: float
    currency: str = 'USD'
    method: str = 'bank_transfer'
    recipient_account: str = 'default_account'
    due_date: str


class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    message: str
    processed_at: str


@app.get("/")
async def root():
    return {'message': 'Welcome to the Invoice Payment API', 'version': '1.0.0'}


@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime': 'OK'
    }


@app.post("/initiate_payment")
async def initiate_payment(payment_request: PaymentRequest):
    transaction_id = f"txn_{int(datetime.utcnow().timestamp())}_{random.randint(1000,9999)}"
    PAYMENT_METRICS['total_transactions'] += 1
    PAYMENT_METRICS['last_transaction_time'] = datetime.utcnow().isoformat()

    success_chance = 0.85
    if random.random() < success_chance:
        status = 'SUCCESS'
        message = f"Payment of {payment_request.amount} {payment_request.currency} successful."
        PAYMENT_METRICS['successful_payments'] += 1
    else:
        status = 'FAILED'
        message = 'Payment failed due to simulated gateway error.'
        PAYMENT_METRICS['failed_payments'] += 1

    transaction_data = {
        'transaction_id': transaction_id,
        'order_id': payment_request.order_id,
        'customer_name': payment_request.customer_name,
        'amount': payment_request.amount,
        'currency': payment_request.currency,
        'method': payment_request.method,
        'recipient_account': payment_request.recipient_account,
        'status': status,
        'message': message,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }

    TRANSACTIONS[transaction_id] = transaction_data
    logger.info(f"Processed payment transaction {transaction_id} with status: {status}")

    return PaymentResponse(
        transaction_id=transaction_id,
        status=status,
        message=message,
        processed_at=transaction_data['updated_at']
    )


@app.get("/transaction/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    transaction = TRANSACTIONS.get(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction


@app.post("/cancel_payment/{transaction_id}")
async def cancel_payment(transaction_id: str):
    transaction = TRANSACTIONS.get(transaction_id)
    if not transaction:
        raise HTTPException(status_code=400, detail="Cannot cancel a non-existent payment")

    if transaction['status'] in ['SUCCESS', 'FAILED']:
        raise HTTPException(status_code=400, detail="Cannot cancel a completed payment")

    transaction['status'] = 'CANCELLED'
    transaction['updated_at'] = datetime.utcnow().isoformat()
    PAYMENT_METRICS['cancelled_payments'] += 1
    logger.info(f"Cancelled payment transaction: {transaction_id}")
    return {
        'transaction_id': transaction_id,
        'status': 'CANCELLED',
        'message': 'Payment has been cancelled successfully'
    }


@app.get("/payment_methods")
async def get_payment_methods():
    return {
        'available_methods': [
            'bank_transfer',
            'credit_card',
            'paypal',
            'upi',
            'crypto'
        ],
        'default_method': 'bank_transfer'
    }


@app.get("/metrics")
async def get_metrics():
    return {
        'metrics': PAYMENT_METRICS,
        'active_transactions': len(TRANSACTIONS)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)