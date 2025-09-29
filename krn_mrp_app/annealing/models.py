from app import db
from datetime import datetime

class AnnealLot(db.Model):
    __tablename__ = "anneal_lots"
    id = db.Column(db.Integer, primary_key=True)
    lot_number = db.Column(db.String(50), unique=True, nullable=False)
    source_rap_lot = db.Column(db.String(50), nullable=False)
    grade = db.Column(db.String(20), nullable=False)  # KRIP → KIP, KRFS → KFS
    ammonia_used = db.Column(db.Float, nullable=False, default=0.0)
    oxygen_percent = db.Column(db.Float, nullable=False)   # mandatory
    compressibility = db.Column(db.Float, nullable=False)  # mandatory
    remarks = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnnealDowntime(db.Model):
    __tablename__ = "anneal_downtime"
    id = db.Column(db.Integer, primary_key=True)
    reason = db.Column(db.String(200), nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    logged_at = db.Column(db.DateTime, default=datetime.utcnow)
