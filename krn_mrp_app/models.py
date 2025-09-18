from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
from .db import Base

class RMPrice(Base):
    __tablename__ = "rm_prices"
    id: Mapped[int] = mapped_column(primary_key=True)
    rm_type: Mapped[str] = mapped_column(String, unique=True, index=True)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GRN(Base):
    __tablename__ = "grns"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    supplier: Mapped[str] = mapped_column(String, default="")
    rm_type: Mapped[str] = mapped_column(String)
    qty_kg: Mapped[float] = mapped_column(Float)
    price_per_kg: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)

class Heat(Base):
    __tablename__ = "heats"
    id: Mapped[int] = mapped_column(primary_key=True)
    heat_no: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    grade: Mapped[str] = mapped_column(String)
    input_total_kg: Mapped[float] = mapped_column(Float)
    theoretical_out_kg: Mapped[float] = mapped_column(Float)
    actual_out_kg: Mapped[float] = mapped_column(Float, default=0.0)
    qa_status: Mapped[str] = mapped_column(String, default="PENDING")
    inputs = relationship("HeatInput", back_populates="heat", cascade="all, delete")

class HeatInput(Base):
    __tablename__ = "heat_inputs"
    id: Mapped[int] = mapped_column(primary_key=True)
    heat_id: Mapped[int] = mapped_column(ForeignKey("heats.id"))
    rm_type: Mapped[str] = mapped_column(String)
    qty_kg: Mapped[float] = mapped_column(Float)
    heat = relationship("Heat", back_populates="inputs")

class HeatQA(Base):
    __tablename__ = "heat_qas"
    id: Mapped[int] = mapped_column(primary_key=True)
    heat_id: Mapped[int] = mapped_column(ForeignKey("heats.id"), unique=True)
    c: Mapped[float] = mapped_column(Float, default=0.0)
    si: Mapped[float] = mapped_column(Float, default=0.0)
    s: Mapped[float] = mapped_column(Float, default=0.0)
    p: Mapped[float] = mapped_column(Float, default=0.0)
    cu: Mapped[float] = mapped_column(Float, default=0.0)
    ni: Mapped[float] = mapped_column(Float, default=0.0)
    mn: Mapped[float] = mapped_column(Float, default=0.0)
    fe: Mapped[float] = mapped_column(Float, default=0.0)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)

class Lot(Base):
    __tablename__ = "lots"
    id: Mapped[int] = mapped_column(primary_key=True)
    lot_no: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    grade: Mapped[str] = mapped_column(String)
    total_kg: Mapped[float] = mapped_column(Float)
    qa_status: Mapped[str] = mapped_column(String, default="PENDING")

class LotHeat(Base):
    __tablename__ = "lot_heats"
    id: Mapped[int] = mapped_column(primary_key=True)
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id"))
    heat_id: Mapped[int] = mapped_column(ForeignKey("heats.id"))

class LotQA(Base):
    __tablename__ = "lot_qas"
    id: Mapped[int] = mapped_column(primary_key=True)
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id"), unique=True)
    c: Mapped[float] = mapped_column(Float, default=0.0)
    si: Mapped[float] = mapped_column(Float, default=0.0)
    s: Mapped[float] = mapped_column(Float, default=0.0)
    p: Mapped[float] = mapped_column(Float, default=0.0)
    cu: Mapped[float] = mapped_column(Float, default=0.0)
    ni: Mapped[float] = mapped_column(Float, default=0.0)
    mn: Mapped[float] = mapped_column(Float, default=0.0)
    fe: Mapped[float] = mapped_column(Float, default=0.0)
    ad: Mapped[float] = mapped_column(Float, default=0.0)
    flow: Mapped[float] = mapped_column(Float, default=0.0)
    plus_212: Mapped[float] = mapped_column(Float, default=0.0)
    plus_180: Mapped[float] = mapped_column(Float, default=0.0)
    m180_p150: Mapped[float] = mapped_column(Float, default=0.0)
    m150_p75: Mapped[float] = mapped_column(Float, default=0.0)
    m75_p45: Mapped[float] = mapped_column(Float, default=0.0)
    m45: Mapped[float] = mapped_column(Float, default=0.0)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
