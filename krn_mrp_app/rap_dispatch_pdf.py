# rap_dispatch_pdf.py
import os
import datetime as dt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def _logo_path():
    base = os.path.dirname(os.path.abspath(_file_))
    p1 = os.path.join(base, "static", "KRN_Logo.png")
    p2 = os.path.join(base, "..", "static", "KRN_Logo.png")
    return p1 if os.path.exists(p1) else p2

def _header(c, title: str):
    width, height = A4
    logo = _logo_path()
    if os.path.exists(logo):
        c.drawImage(logo, 1.5*cm, height-3.0*cm, width=4*cm, height=3*cm, preserveAspectRatio=True, mask="auto")
    c.setFont("Helvetica-Bold", 24)
    c.drawString(7*cm, height-2.1*cm, "KRN Alloys Pvt. Ltd")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(7*cm, height-2.7*cm, title)
    c.line(1.5*cm, height-3.3*cm, width-1.5*cm, height-3.3*cm)

def _hline(c, y):
    width, _ = A4
    c.line(2*cm, y, width-2*cm, y)

def _ensure_space(c, y, need=36):
    """Start a new page if near bottom; returns fresh y."""
    width, height = A4
    if y < 3.5*cm + need:
        c.showPage()
        _header(c, "Dispatch Note – For Invoice Purpose Only")
        return height - 4.0*cm
    return y

# ------------------------------------------------------------------
# Helper: draw annexure (QA certificate + trace to GRN) for ONE lot
# Returns the updated y position.
# ------------------------------------------------------------------
def draw_lot_qa_annexure(c, lot, disp_qty, db=None, y=None):
    """
    Renders:
      - Lot line (lot, grade, QA status, dispatch qty)
      - CoA (Chemistry / Physical / PSD) if present
      - Heats with FIFO GRN consumption (supplier, qty)
    Uses _ensure_space(...) for pagination and returns updated y.
    """
    if y is None:
        # sensible default start if someone calls without y
        _, height = A4
        y = height - 4.0*cm

    # Lot overview
    y = _ensure_space(c, y, 16)
    c.setFont("Helvetica", 10)
    c.drawString(
        2*cm, y,
        f"Lot: {lot.lot_no}  |  Grade: {lot.grade or ''}  |  QA: {getattr(lot,'qa_status','') or ''}  |  Dispatch Qty: {float(disp_qty or 0):.1f} kg"
    )
    y -= 12

    # CoA blocks
    chem = getattr(lot, "chemistry", None)
    phys = getattr(lot, "phys", None)
    psd  = getattr(lot, "psd", None)

    if chem or phys or psd:
        y = _ensure_space(c, y, 14)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.2*cm, y, "CoA (QA):")
        c.setFont("Helvetica", 10)
        y -= 12

    if chem:
        line = []
        for k in ("c","si","s","p","cu","ni","mn","fe"):
            v = getattr(chem, k, None)
            if v not in (None, ""):
                line.append(f"{k.upper()}:{v}")
        if line:
            y = _ensure_space(c, y, 12)
            c.drawString(2.6*cm, y, "Chemistry: " + ", ".join(line))
            y -= 12

    if phys and (getattr(phys, "ad", None) or getattr(phys, "flow", None)):
        y = _ensure_space(c, y, 12)
        c.drawString(2.6*cm, y, f"Physical: AD={getattr(phys,'ad','')}, Flow={getattr(phys,'flow','')}")
        y -= 12

    if psd and (getattr(psd,"p212",None) or getattr(psd,"p180",None) or getattr(psd,"n180p150",None)
                or getattr(psd,"n150p75",None) or getattr(psd,"n75p45",None) or getattr(psd,"n45",None)):
        y = _ensure_space(c, y, 12)
        parts = []
        for label, attr in [
            ("+212", "p212"), ("+180", "p180"), ("-180+150", "n180p150"),
            ("-150+75", "n150p75"), ("-75+45", "n75p45"), ("-45", "n45")
        ]:
            v = getattr(psd, attr, None)
            if v not in (None, ""):
                parts.append(f"{label}:{v}")
        if parts:
            c.drawString(2.6*cm, y, "PSD: " + ", ".join(parts))
            y -= 12

    # Traceability: heats + GRN FIFO
    for lh in getattr(lot, "heats", []):
        h = lh.heat
        y = _ensure_space(c, y, 14)
        c.drawString(
            2.2*cm, y,
            f"Heat {h.heat_no}: Alloc to lot {float(getattr(lh,'qty',0) or 0):.1f} kg, "
            f"Heat Out {float(getattr(h,'actual_output',0) or 0):.1f} kg, QA {getattr(h,'qa_status','') or ''}"
        )
        y -= 12
        for cons in getattr(h, "rm_consumptions", []):
            g = cons.grn
            supp = g.supplier if g else ""
            y = _ensure_space(c, y, 12)
            c.drawString(2.8*cm, y, f"- {cons.rm_type}: GRN #{cons.grn_id} ({supp}) {float(cons.qty or 0):.1f} kg")
            y -= 12

    return y - 6  # small gap after each lot block

# ------------------------------------------------------------------
# Main PDF builder
# ------------------------------------------------------------------
def draw_dispatch_note(c, disp, items, db):
    """
    Renders the dispatch note.
    - disp: RAPDispatch ORM instance
    - items: list of RAPDispatchItem (each has .lot and .qty, .cost)
    - db: Session (only used if needed)
    """
    width, height = A4
    _header(c, "Dispatch Note – For Invoice Purpose Only")

    # Top meta block
    y = height - 4.0*cm
    now = dt.datetime.now()
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Dispatch ID: {disp.id}"); y -= 14
    c.drawString(2*cm, y, f"Date: {disp.date.isoformat()}  Time: {now.strftime('%H:%M')}"); y -= 14
    c.drawString(2*cm, y, f"Customer: {disp.customer}"); y -= 14
    c.drawString(2*cm, y, f"Grade: {disp.grade}"); y -= 10
    _hline(c, y); y -= 10

    # Items table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2.0*cm, y, "Lot")
    c.drawString(8.5*cm, y, "Qty (kg)")
    c.drawString(11.5*cm, y, "Cost/kg")
    c.drawString(14.3*cm, y, "Amount")
    y -= 12
    c.setFont("Helvetica", 10)

    total = 0.0
    for it in items:
        lot = it.lot
        qty = float(it.qty or 0.0)
        unit = float(lot.unit_cost or 0.0)
        amt = qty * unit
        total += amt

        y = _ensure_space(c, y, 16)
        c.drawString(2.0*cm, y, f"{lot.lot_no}")
        c.drawRightString(10.6*cm, y, f"{qty:.1f}")
        c.drawRightString(13.8*cm, y, f"{unit:.2f}")
        c.drawRightString(width-2.0*cm, y, f"{amt:.2f}")
        y -= 12

    _hline(c, y); y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(width-2.0*cm, y, f"Total Amount: {total:.2f}")
    y -= 18

    # Manual rate placeholder + note (exact phrasing requested)
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Final Sell Rate (manual): __________________  ₹/kg"); y -= 16
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2*cm, y, "This Document is Dispatch Note use for Invoice Purpose Only"); y -= 14

    # -------------------------------
    # Annexure – QA & Traceability
    # -------------------------------
    y = _ensure_space(c, y, 22)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Annexure – QA Certificate & Traceability")
    y -= 12
    c.setFont("Helvetica", 10)

    # NEW: use helper for each item
    for it in items:
        y = draw_lot_qa_annexure(c, it.lot, it.qty, db=db, y=y)

    # Signature block
    y = _ensure_space(c, y, 40)
    _hline(c, y); y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Prepared By: _______________________")
    c.drawString(9.5*cm, y, "Approved By: _______________________")
