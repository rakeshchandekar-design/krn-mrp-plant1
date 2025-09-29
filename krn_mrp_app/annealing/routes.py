from flask import Blueprint, render_template, request, redirect, url_for, flash
from app import db
from .models import AnnealLot, AnnealDowntime

anneal_bp = Blueprint("anneal", __name__, url_prefix="/anneal")

@anneal_bp.route("/")
def home():
    return render_template("annealing_home.html")

@anneal_bp.route("/new_lot", methods=["GET", "POST"])
def new_lot():
    if request.method == "POST":
        lot_number = request.form["lot_number"]
        grade = request.form["grade"]
        ammonia = float(request.form["ammonia_used"])
        oxygen = float(request.form["oxygen_percent"])
        compress = float(request.form["compressibility"])

        if oxygen <= 0 or compress <= 0:
            flash("Oxygen % and Compressibility must be > 0")
            return redirect(url_for("anneal.new_lot"))

        newlot = AnnealLot(
            lot_number=lot_number,
            source_rap_lot=request.form["source_rap_lot"],
            grade=grade,
            ammonia_used=ammonia,
            oxygen_percent=oxygen,
            compressibility=compress,
            remarks=request.form.get("remarks", "")
        )
        db.session.add(newlot)
        db.session.commit()
        flash("Anneal lot created successfully")
        return redirect(url_for("anneal.list_lots"))
    return render_template("annealing_lot_form.html")

@anneal_bp.route("/lots")
def list_lots():
    lots = AnnealLot.query.all()
    return render_template("annealing_lot_list.html", lots=lots)

@anneal_bp.route("/downtime", methods=["GET", "POST"])
def downtime():
    if request.method == "POST":
        reason = request.form["reason"]
        duration = int(request.form["duration_minutes"])
        d = AnnealDowntime(reason=reason, duration_minutes=duration)
        db.session.add(d)
        db.session.commit()
        flash("Downtime logged successfully")
        return redirect(url_for("anneal.downtime"))
    logs = AnnealDowntime.query.order_by(AnnealDowntime.logged_at.desc()).all()
    return render_template("annealing_downtime.html", logs=logs)
