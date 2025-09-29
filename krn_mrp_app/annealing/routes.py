# krn_mrp_app/annealing/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import date, timedelta
from sqlalchemy import text
import json, io, csv

from app import db
from .models import AnnealLot, AnnealDowntime

anneal_bp = Blueprint("anneal", __name__, url_prefix="/anneal", template_folder="templates")

TARGET_KG_PER_DAY = 6000
ANNEAL_ADD_COST = 10.0  # ₹/kg to add over weighted RAP cost

from sqlalchemy import text

def fetch_approved_rap_balance():
    sql = text("""
        SELECT
          rl.id            AS rap_row_id,
          l.id             AS lot_id,
          COALESCE(l.lot_no, CONCAT('LOT-', l.id)) AS lot_no,
          l.grade          AS grade,          -- KRIP/KRFS
          l.cost_per_kg    AS cost_per_kg,    -- change if your cost column name differs
          rl.available_qty AS available_kg
        FROM rap_lot rl
        JOIN lot l ON l.id = rl.lot_id
        WHERE rl.available_qty > 0
          AND (l.status = 'APPROVED' OR l.qa_status = 'APPROVED')
        ORDER BY l.date ASC, rl.id ASC
    """)
    return db.session.execute(sql).mappings().all()


@anneal_bp.route("/")
def home():
    # very small KPI to start
    today = date.today()
    lots_today = db.session.execute(
        text("SELECT COUNT(*) FROM anneal_lots WHERE date=:d"), {"d": today}
    ).scalar()
    nh3_today = db.session.execute(
        text("SELECT COALESCE(SUM(ammonia_kg),0) FROM anneal_lots WHERE date=:d"), {"d": today}
    ).scalar()
    return render_template("annealing_home.html",
                           lots_today=lots_today or 0,
                           nh3_today=nh3_today or 0.0,
                           target=TARGET_KG_PER_DAY)

@anneal_bp.route("/create", methods=["GET","POST"])
def create():
    if request.method == "POST":
        # collect allocations
        allocations = {}  # {rap_lot: qty}
        total_alloc = 0.0
        for k,v in request.form.items():
            if k.startswith("alloc_") and v.strip():
                lot = k.replace("alloc_", "")
                qty = float(v)
                if qty > 0:
                    allocations[lot] = qty
                    total_alloc += qty

        if total_alloc <= 0:
            flash("Enter at least one allocation.", "warning")
            return redirect(url_for("anneal.create"))

        ammonia_kg = float(request.form.get("ammonia_kg","0") or 0)

        # fetch RAP costs + grades for selected lots
        lots_tuple = tuple(allocations.keys())
        rows = db.session.execute(text("""
            SELECT lot_no, grade, cost_per_kg
            FROM rap_lots WHERE lot_no IN :lots
        """), {"lots": lots_tuple}).mappings().all()
        if not rows:
            flash("Selected RAP lots were not found.", "danger")
            return redirect(url_for("anneal.create"))

        # validate same family (KRIP or KRFS)
        families = set(r["grade"] for r in rows)
        if len(families) > 1:
            flash("Please allocate from the same RAP grade family (KRIP or KRFS).", "danger")
            return redirect(url_for("anneal.create"))

        # map grade
        rap_grade = list(families)[0]
        out_grade = "KIP" if rap_grade == "KRIP" else "KFS"

        # weighted RAP cost
        rap_cost_wsum = 0.0
        for r in rows:
            rap_cost_wsum += allocations.get(r["lot_no"], 0.0) * float(r["cost_per_kg"] or 0)
        rap_cost_per_kg = rap_cost_wsum / total_alloc if total_alloc else 0.0
        cost_per_kg = rap_cost_per_kg + ANNEAL_ADD_COST

        # generate lot number: ANL-YYYYMMDD-###
        prefix = "ANL-" + date.today().strftime("%Y%m%d") + "-"
        last = db.session.execute(text(
            "SELECT lot_no FROM anneal_lots WHERE lot_no LIKE :pfx ORDER BY lot_no DESC LIMIT 1"
        ), {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        lot_no = f"{prefix}{seq:03d}"

        # insert anneal lot
        rec = AnnealLot(
            lot_no=lot_no,
            date=date.today(),
            src_alloc_json=json.dumps(allocations),
            grade=out_grade,
            weight_kg=total_alloc,
            rap_cost_per_kg=rap_cost_per_kg,
            cost_per_kg=cost_per_kg,
            ammonia_kg=ammonia_kg
        )
        db.session.add(rec)

        # deduct from RAP available
        for rap_lot, qty in allocations.items():
            db.session.execute(text("""
                UPDATE rap_lots SET available_kg = available_kg - :q
                WHERE lot_no=:lot AND available_kg >= :q
            """), {"q": qty, "lot": rap_lot})

        db.session.commit()
        flash(f"Anneal lot {lot_no} created. Cost/kg = ₹{cost_per_kg:.2f} (RAP {rap_cost_per_kg:.2f} + 10).", "success")
        return redirect(url_for("anneal.lots"))

    rap = fetch_approved_rap_balance()
    return render_template("annealing_create.html", rap_rows=rap)

@anneal_bp.route("/lots")
def lots():
    rows = db.session.execute(text("""
      SELECT id, date, lot_no, grade, weight_kg, ammonia_kg, rap_cost_per_kg, cost_per_kg, qa_status
      FROM anneal_lots ORDER BY date DESC, lot_no DESC
    """)).mappings().all()
    return render_template("annealing_lot_list.html", lots=rows)

@anneal_bp.route("/qa/<int:lot_id>", methods=["GET","POST"])
def qa(lot_id):
    lot = AnnealLot.query.get_or_404(lot_id)
    if request.method == "POST":
        try:
            o = float(request.form["o_pct"])
            cpr = float(request.form["compressibility"])
            if o <= 0 or cpr <= 0:
                raise ValueError("Oxygen % and Compressibility must be > 0")

            lot.o_pct = o
            lot.compressibility = cpr
            lot.qa_status = request.form.get("qa_status","APPROVED")
            # optional chemistry carry-forward edits
            for f in ("c_pct","si_pct","mn_pct","s_pct","p_pct"):
                if request.form.get(f):
                    setattr(lot, f, float(request.form.get(f)))
            lot.remarks = request.form.get("remarks","")
            db.session.commit()
            flash("QA saved.", "success")
            return redirect(url_for("anneal.lots"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}", "danger")
            return redirect(url_for("anneal.qa", lot_id=lot_id))
    return render_template("annealing_qa_form.html", lot=lot)

@anneal_bp.route("/downtime", methods=["GET","POST"])
def downtime():
    if request.method == "POST":
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
    logs = AnnealDowntime.query.order_by(AnnealDowntime.date.desc()).all()
    return render_template("annealing_downtime.html", logs=logs)
