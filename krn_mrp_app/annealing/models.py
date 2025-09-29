# models_annealing.py
from app import db
from sqlalchemy import func

class AnnealDowntime(db.Model):
    __tablename__ = "anneal_downtime"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, index=True, nullable=False)
    minutes = db.Column(db.Integer, nullable=False)
    area = db.Column(db.String(64), nullable=False)  # e.g., "Furnace", "Cooling", "Conveyor"
    reason = db.Column(db.String(255), nullable=False)

class AnnealLot(db.Model):
    __tablename__ = "anneal_lots"
    id = db.Column(db.Integer, primary_key=True)
    lot_no = db.Column(db.String(32), unique=True, index=True, nullable=False)
    date = db.Column(db.Date, index=True, nullable=False)
    # Source RAP lots allocation summary
    src_alloc_json = db.Column(db.Text, nullable=False)  # {"RAP-20250921-001": 300.0, ...}
    # Produced
    grade = db.Column(db.String(16), nullable=False)     # KIP or KFS
    weight_kg = db.Column(db.Float, nullable=False)
    rap_cost_per_kg = db.Column(db.Float, nullable=False, default=0)  # weighted RAP cost
    cost_per_kg = db.Column(db.Float, nullable=False, default=0)
    ammonia_kg = db.Column(db.Float, nullable=False, default=0)  # recorded per lot 

    # QA fields
    qa_status = db.Column(db.String(16), nullable=False, default="PENDING")  # PENDING/APPROVED/HOLD/REJECTED
    c_pct = db.Column(db.Float)      # carry-forward or entered at QA
    si_pct = db.Column(db.Float)
    mn_pct = db.Column(db.Float)
    s_pct  = db.Column(db.Float)
    p_pct  = db.Column(db.Float)
    o_pct  = db.Column(db.Float)     # NEW oxygen %
    compressibility = db.Column(db.Float)  # NEW
    remarks = db.Column(db.String(255))

    # bookkeeping
    created_by = db.Column(db.String(64))
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
