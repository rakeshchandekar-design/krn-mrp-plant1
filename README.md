# KRN MRP Plant1 – v1.1

**What’s new**
- FIFO GRN stock & automatic consumption in Melting
- Slag entry; Actual output = Inputs − Slag; 3% theoretical loss monitoring
- QA holds & approvals on heats and lots
- Atomization lots from approved heats, 3,000 kg default
- Lot-by-lot PDF traceability (with KRN logo)
- Simple top navigation, role links (Stores/QA/RAP/Admin)
- Render/Neon ready. Uses SQLite by default; set `DATABASE_URL` to use Postgres.

## Quick start (Render)
1. Create a new Web Service from this repo.
2. (Optional) Add env var `DATABASE_URL` with your Neon Postgres connection string.
3. Deploy. Open `/setup` once, then go to `/grn`.

## Local
```
pip install -r requirements.txt
uvicorn krn_mrp_app.main:app --reload
```
Go to: http://127.0.0.1:8000/setup, then `/grn`.
