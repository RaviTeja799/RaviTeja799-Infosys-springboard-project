# Fraud Detection API - Test Commands

## Prerequisites
- Backend running: `python app.py`
- API running on: `http://127.0.0.1:5000`
- API Key: `super_secret_bfsi_key_123`

---

## Test 1: Legitimate Transaction

**Description:** Tests a normal, low-risk transaction from an established account with verified KYC.

**Expected Result:** `Legitimate` prediction with low risk score

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Headers @{
    "Content-Type" = "application/json"
    "X-API-Key" = "super_secret_bfsi_key_123"
} -Body '{
    "transaction_id": "T1001",
    "customer_id": "C123",
    "amount": 2000,
    "channel": "online",
    "timestamp": "2024-01-15T14:30:00",
    "account_age_days": 365,
    "kyc_verified": "Yes"
}'
```

**Expected Response:**
```json
{
  "transaction_id": "T1001",
  "prediction": "Legitimate",
  "risk_score": 0.15,
  "confidence": 0.85,
  "reason": "Low fraud risk. Normal transaction pattern for amount ₹2,000",
  "rule_flags": [],
  "model_version": "RandomForest"
}
```

---

## Test 2: Fraud - High Amount from New Account

**Description:** Tests a high-risk transaction with multiple fraud indicators:
- High amount (₹45,000)
- New account (5 days old)
- Unusual hour (3:00 AM)
- KYC not verified

**Expected Result:** `Fraud` prediction with high risk score and multiple rule flags

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Headers @{
    "Content-Type" = "application/json"
    "X-API-Key" = "super_secret_bfsi_key_123"
} -Body '{
    "transaction_id": "T1002",
    "customer_id": "C456",
    "amount": 45000,
    "channel": "mobile",
    "timestamp": "2024-01-15T03:00:00",
    "account_age_days": 5,
    "kyc_verified": "No"
}'
```

**Expected Response:**
```json
{
  "transaction_id": "T1002",
  "prediction": "Fraud",
  "risk_score": 0.89,
  "confidence": 0.95,
  "reason": "Rules triggered: High transaction amount (₹45,000) from new account (5 days old); KYC not verified with amount ₹45,000; Transaction at unusual hour (3:00) with amount ₹45,000; Very new account (5 days) without KYC verification",
  "rule_flags": [
    "HIGH_VALUE_NEW_ACCOUNT",
    "UNVERIFIED_KYC_HIGH_AMOUNT",
    "UNUSUAL_HOUR",
    "NEW_ACCOUNT_UNVERIFIED"
  ],
  "model_version": "RandomForest"
}
```

---

## Test 3: Get Result by Transaction ID

**Description:** Retrieves the stored prediction result for a specific transaction.

**Note:** Must be run after Test 2 to retrieve T1002 data.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/result/T1002" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}
```

**Expected Response:**
```json
{
  "transaction_id": "T1002",
  "prediction": "Fraud",
  "risk_score": 0.89,
  "confidence": 0.95,
  "reason": "Rules triggered: High transaction amount (₹45,000) from new account (5 days old); ...",
  "rule_flags": [
    "HIGH_VALUE_NEW_ACCOUNT",
    "UNVERIFIED_KYC_HIGH_AMOUNT",
    "UNUSUAL_HOUR",
    "NEW_ACCOUNT_UNVERIFIED"
  ],
  "model_version": "RandomForest",
  "customer_id": "C456",
  "amount": 45000,
  "channel": "mobile",
  "timestamp": "2024-01-15T03:00:00",
  "account_age_days": 5,
  "kyc_verified": "No",
  "processed_at": "2024-01-15T10:30:45.123456",
  "latency_ms": 45.32
}
```

---

## Test 4: Get Overall Statistics

**Description:** Retrieves aggregate statistics about all predictions made.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/stats" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}
```

**Expected Response:**
```json
{
  "total_predictions": 2,
  "fraud_count": 1,
  "legitimate_count": 1,
  "fraud_rate": 0.5,
  "avg_risk_score": 0.52
}
```

---

## Additional Test Endpoints

### Get All Results (with filters)

```powershell
# Get all results
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/results" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}

# Get only fraud transactions
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/results?fraud_only=true" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}

# Get last 50 results
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/results?limit=50" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}
```

---

## Business Rules Tested

The API implements 6 fraud detection rules:

1. **HIGH_VALUE_NEW_ACCOUNT** - Amount > ₹10,000 from account < 30 days old
2. **UNVERIFIED_KYC_HIGH_AMOUNT** - KYC not verified + amount > ₹5,000
3. **UNUSUAL_HOUR** - Transactions between 2 AM - 5 AM with amount > ₹3,000
4. **VERY_HIGH_AMOUNT** - Transactions > ₹50,000
5. **NEW_ACCOUNT_UNVERIFIED** - Account < 7 days + KYC not verified
6. **HIGH_ATM_WITHDRAWAL** - ATM/POS transactions > ₹20,000

---

## Error Cases

### Invalid API Key
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Headers @{
    "Content-Type" = "application/json"
    "X-API-Key" = "wrong_key"
} -Body '{
    "transaction_id": "T1003",
    "customer_id": "C789",
    "amount": 1000,
    "channel": "web"
}'
```
**Expected:** `401 Unauthorized` - "Invalid API Key"

### Missing Required Fields
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Headers @{
    "Content-Type" = "application/json"
    "X-API-Key" = "super_secret_bfsi_key_123"
} -Body '{
    "transaction_id": "T1004",
    "amount": 5000
}'
```
**Expected:** `400 Bad Request` - "Missing required fields"

### Transaction Not Found
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/result/INVALID_ID" -Method GET -Headers @{
    "X-API-Key" = "super_secret_bfsi_key_123"
}
```
**Expected:** `404 Not Found` - "No result found for transaction_id: INVALID_ID"

---

## Milestone 3 Deliverables Checklist

- ✅ `/api/predict` endpoint working
- ✅ Fraud detection combines ML model + business rules
- ✅ Fraud alerts stored in memory - ready for DB
- ✅ Results stored correctly with all metadata
- ✅ API returns JSON with prediction, risk_score, and reason
- ✅ Rule flags included in response
- ✅ Model version tracking
- ✅ Result retrieval by transaction ID
- ✅ Statistics endpoint for monitoring

---

## Notes
- Timestamps should be in ISO 8601 format
- Risk scores range from 0.0 (no risk) to 1.0 (high risk)
- Confidence scores are derived from prediction probabilities
- Results are stored in-memory (will be replaced with MongoDB)