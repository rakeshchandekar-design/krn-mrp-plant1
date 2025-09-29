# annealing/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import date
from app import db
from .models import AnnealLot, AnnealDowntime

anneal_bp = Blueprint(
    "anneal",
    __name__,
    url_prefix="/anneal",
    template_folder="templates"
)

@anneal_bp.route("/")
def home():
    # Minimal dashboard: counts for a quick smoke test
    lots_today = AnnealLot.query.filter(AnnealLot.date == date.today()).count()
    nh3_today = db.session.query(db.func.coalesce(db.func.sum(AnnealLot.ammonia_kg), 0.0))\
                          .filter(AnnealLot.date == date.today()).scalar()
    return render_template("annealing_home.html", lots_today=lots_today, nh3_today=nh3_today)

@anneal_bp.route("/new_lot", methods=["GET", "POST"])
def new_lot():
    if request.method == "POST":
        try:
            lot_no = request.form["lot_no"].strip()
            src_json = request.form.get("src_alloc_json", "{}").strip() or "{}"
            grade = request.form["grade"].strip()            # "KIP" or "KFS"
            weight_kg = float(request.form["weight_kg"])
            cost_per_kg = float(request.form.get("cost_per_kg", "0") or 0)
            ammonia_kg = float(request.form.get("ammonia_kg", "0") or 0)

            # Minimal lot record for now (QA fields later)
            rec = AnnealLot(
                lot_no=lot_no,
                date=date.today(),
                src_alloc_json=src_json,
                grade=grade,
                weight_kg=weight_kg,
                cost_per_kg=cost_per_kg,
                ammonia_kg=ammonia_kg,
            )
            db.session.add(rec)
            db.session.commit()
            flash(f"Anneal lot {lot_no} saved.", "success")
            return redirect(url_for("anneal.list_lots"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}", "danger")
            return redirect(url_for("anneal.new_lot"))

    return render_template("annealing_lot_form.html")

@anneal_bp.route("/lots")
def list_lots():
    lots = AnnealLot.query.order_by(AnnealLot.date.desc(), AnnealLot.lot_no.desc()).all()
    return render_template("annealing_lot_list.html", lots=lots)

@anneal_bp.route("/qa/<int:lot_id>", methods=["GET", "POST"])
def qa(lot_id):
    lot = AnnealLot.query.get_or_404(lot_id)
    if request.method == "POST":
        try:
            o_pct = float(request.form["o_pct"])
            compress = float(request.form["compressibility"])
            if o_pct <= 0 or compress <= 0:
                raise ValueError("Oxygen % and Compressibility must be > 0")

            lot.o_pct = o_pct
            lot.compressibility = compress
            lot.qa_status = request.form.get("qa_status", "PENDING")
            lot.c_pct = float(request.form.get("c_pct")) if request.form.get("c_pct") else lot.c_pct
            lot.si_pct = float(request.form.get("si_pct")) if request.form.get("si_pct") else lot.si_pct
            lot.mn_pct = float(request.form.get("mn_pct")) if request.form.get("mn_pct") else lot.mn_pct
            lot.s_pct  = float(request.form.get("s_pct"))  if request.form.get("s_pct")  else lot.s_pct
            lot.p_pct  = float(request.form.get("p_pct"))  if request.form.get("p_pct")  else lot.p_pct
            lot.remarks = request.form.get("remarks", lot.remarks)

            db.session.commit()
            flash("QA updated.", "success")
            return redirect(url_for("anneal.list_lots"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}", "danger")
            return redirect(url_for("anneal.qa", lot_id=lot_id))

    return render_template("annealing_qa_form.html", lot=lot)

@anneal_bp.route("/downtime", methods=["GET", "POST"])
def downtime():
    if request.method == "POST":
        try:
            rec = AnnealDowntime(
                date=request.form["date"],
                minutes=int(request.form["minutes"]),
                area=request.form["area"].strip(),
                reason=request.form["reason"].strip()
            )
            db.session.add(rec)
            db.session.commit()
            flash("Downtime logged.", "success")
            return redirect(url_for("anneal.downtime"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}", "danger")
            return redirect(url_for("anneal.downtime"))

    logs = AnnealDowntime.query.order_by(AnnealDowntime.date.desc()).all()
    return render_template("annealing_downtime.html", logs=logs)
