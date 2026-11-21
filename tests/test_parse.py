import io
import zipfile

import pandas as pd

from app.config import settings
from app.services.extractors import detect_file_type, map_columns, rows_from_dataframe
from app.services.redaction import redact_sensitive


def test_redaction_masks_sensitive_data_and_keeps_merchant_details():
    text = (
        "Paid to Coffee Shop on 12/04/2024 for 1,250.00 INR. "
        "Acct 1234567890123456 Card 9876543210987 "
        "Email test.user@example.com Phone +91 9876543210 "
        "PAN ABCDE1234F Aadhaar 1234 5678 9012 IFSC HDFC0001234 UPI name@bank"
    )
    redacted = redact_sensitive(text)

    assert "1234567890123456" not in redacted
    assert "9876543210987" not in redacted
    assert "3456" in redacted  # account/card numbers keep only last 4
    assert "0987" in redacted
    assert "test.user@example.com" not in redacted
    assert "9876543210" not in redacted
    assert "ABCDE1234F" not in redacted
    assert "1234 5678 9012" not in redacted
    assert "HDFC0001234" not in redacted
    assert "name@bank" not in redacted

    # Dates, amounts and merchant/description text should remain intact.
    assert "12/04/2024" in redacted
    assert "1,250.00" in redacted
    assert "Coffee Shop" in redacted


def test_column_mapping_common_headers_and_debit_credit_derivation():
    df = pd.DataFrame(
        [
            {
                "Txn Date": "12/04/2024",
                "Narration": "Salary credit",
                "Debit": "",
                "Credit": "50,000.00",
            },
            {
                "Txn Date": "13/04/2024",
                "Narration": "Grocery store",
                "Debit": "1,250.50",
                "Credit": "",
            },
        ]
    )

    mapping = map_columns(list(df.columns))
    assert mapping["date"] == "Txn Date"
    assert mapping["description"] == "Narration"
    assert mapping["debit"] == "Debit"
    assert mapping["credit"] == "Credit"

    rows, mapped = rows_from_dataframe(df)
    assert mapped is True
    assert len(rows) == 2
    assert rows[0]["type"] == "income"
    assert rows[0]["amount"] == 50000.0
    assert rows[0]["date"] == "2024-04-12"
    assert rows[1]["type"] == "expense"
    assert rows[1]["amount"] == 1250.5
    assert rows[1]["date"] == "2024-04-13"


def test_type_detection_from_magic_bytes():
    assert detect_file_type(b"%PDF-1.4 test") == "pdf"
    assert detect_file_type(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert detect_file_type(b"\xff\xd8\xff\xdbrest") == "jpg"

    csv_bytes = b"date,description,amount\n12/04/2024,Coffee,120\n"
    assert detect_file_type(csv_bytes) == "csv"

    xlsx_buf = io.BytesIO()
    with zipfile.ZipFile(xlsx_buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
    assert detect_file_type(xlsx_buf.getvalue()) == "xlsx"


def test_parse_endpoint_auth_checks(client, monkeypatch):
    csv_bytes = b"date,description,amount,type\n12/04/2024,Coffee,120,expense\n"

    # Fail closed when INTERNAL_API_KEY is not configured.
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "")
    response = client.post(
        "/parse",
        headers={"X-Internal-Key": "secret"},
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 503

    # Missing or wrong key rejected.
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "secret")
    response = client.post("/parse", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert response.status_code == 401

    response = client.post(
        "/parse",
        headers={"X-Internal-Key": "wrong"},
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 401

    # Browser-facing origins are not allowed.
    response = client.post(
        "/parse",
        headers={"X-Internal-Key": "secret", "Origin": "http://evil.example"},
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 403

    # Correct internal key accepted for a valid CSV (deterministic path, no LLM).
    response = client.post(
        "/parse",
        headers={"X-Internal-Key": "secret"},
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "csv"
    assert body["used_llm"] is False
    assert len(body["transactions"]) == 1
    assert body["transactions"][0]["amount"] == 120.0
    assert body["transactions"][0]["type"] == "expense"
    assert body["transactions"][0]["date"] == "2024-04-12"
