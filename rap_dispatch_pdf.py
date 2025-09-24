# rap_dispatch_pdf.py
import os
import datetime as dt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def _logo_path():
    base = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(base, "static", "KRN_Logo.png")
    p2 = os.path.join(base, "..", "static", "KRN_Logo.png")
    return p1 if os.path.exists(p1) else p2

def _header(c, title: str):
    width, height = A4
    logo = _logo_path()
    if os.path.exists(logo):
        c.drawImage(
            logo,
            1.5 * cm,
            height - 3.0 * cm,
            width=4 * cm,
            height=3 * cm,
            preserveAspectRatio=True,
            mask="auto",
        )
    c.setFont("Helvetica-Bold", 24)
    c.drawString(7 * cm, height - 2.1 * cm, "KRN Alloys Pvt. Ltd")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(7 * cm, height - 2.7 * cm, title)
    c.line(1.5 * cm, height - 3.3 * cm, width - 1.5 * cm, height - 3.3 * cm)

def _hline(c, y):
    width, _ = A4
    c.line(2 * cm, y, width - 2 * cm, y)

def _ensure_space(c, y, need=36):
    """Start a new page if near bottom; returns fresh y."""
    width, height = A4
    if y < 3.5 * cm + need:
        c.showPage()
        _header(c, "Dispatch Note – For Invoice Purpose Only")
        return height - 4.0 * cm
    return y

def _draw_qa_table(c, x, y, lot):
    """
    Draw a compact CoA-style table for Chemistry / Physical / PSD.
    Shows headings and blanks if values are missing (as requested).
    Returns updated y.
    """
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "CoA (QA)"); y -= 12
    c.setFont("Helvetica", 9)

    # Chemistry
    chem = getattr(lot, "chemistry", None)
    chem_fields = [("C", "c"), ("Si", "si"), ("S", "s"), ("P", "p"),
                   ("Cu", "cu"), ("Ni", "ni"), ("Mn", "mn"), ("Fe", "fe")]
    c.drawString(x + 4, y, "Chemistry:"); y -= 12
    line = []
    for label, attr in chem_fields:
        v = getattr(chem, attr, "") if chem is not None else ""
        line.append(f"{label}:{v if v not in (None, '') else '-'}")
    c.drawString(x + 12, y, ", ".join(line)); y -= 12

    # Physical
    phys = getattr(lot, "phys", None)
    ad = getattr(phys, "ad", "") if phys is not None else ""
    flow = getattr(phys, "flow", "") if phys is not None else ""
    c.drawString(x + 4, y, "Physical:"); y -= 12
    c.drawString(x + 12, y, f"AD:{ad if ad not in (None, '') else '-'}, Flow:{flow if flow not in (None, '') else '-'}"); y -= 12

    # PSD
    psd = getattr(lot, "psd", None)
    psd_pairs = [
        ("+212", "p212"), ("+180", "p180"), ("-180+150", "n180p150"),
        ("-150+75", "n150p75"), ("-75+45", "n75p45"), ("-45", "n45"),
    ]
    c.drawString(x + 4, y, "PSD:"); y -= 12
    parts = []
    for label, attr in psd_pairs:
        v = getattr(psd, attr, "") if psd is not None else ""
        parts.append(f"{label}:{v if v not in (None, '') else '-'}")
    c.drawString(x + 12, y, ", ".join(parts)); y -= 12

    return y

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
    y = height - 4.0 * cm
    now = dt.datetime.now()
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Dispatch ID: {disp.id}"); y -= 14
    c.drawString(2 * cm, y, f"Date: {disp.date.isoformat()}  Time: {now.strftime('%H:%M')}"); y -= 14
    c.drawString(2 * cm, y, f"Customer: {disp.customer}"); y -= 14
    c.drawString(2 * cm, y, f"Grade: {disp.grade}"); y -= 10
    _hline(c, y); y -= 10

    # Items table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2.0 * cm, y, "Lot")
    c.drawString(8.5 * cm, y, "Qty (kg)")
    c.drawString(11.5 * cm, y, "Cost/kg")
    c.drawString(14.3 * cm, y, "Amount")
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
        c.drawString(2.0 * cm, y, f"{lot.lot_no}")
        c.drawRightString(10.6 * cm, y, f"{qty:.1f}")
        c.drawRightString(13.8 * cm, y, f"{unit:.2f}")
        c.drawRightString(width - 2.0 * cm, y, f"{amt:.2f}")
        y -= 12

    _hline(c, y); y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(width - 2.0 * cm, y, f"Total Amount: {total:.2f}")
    y -= 18

    # Manual rate placeholder + note (exact phrasing requested)
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Final Sell Rate (manual): __________________  ₹/kg"); y -= 16
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2 * cm, y, "This Document is Dispatch Note use for Invoice Purpose Only"); y -= 14

    # Annexure: QA & Trace to GRN (per lot)
    y = _ensure_space(c, y, 22)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Annexure – QA Certificate & Traceability")
    y -= 12
    c.setFont("Helvetica", 10)

    for it in items:
        lot = it.lot
        disp_qty = float(it.qty or 0.0)

        y = _ensure_space(c, y, 60)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, y, f"Lot: {lot.lot_no}  |  Grade: {lot.grade or ''}  |  Dispatch Qty: {disp_qty:.1f} kg")
        y -= 12
        c.setFont("Helvetica", 10)

        # CoA table (always visible, with blanks if missing)
        y = _draw_qa_table(c, 2 * cm, y, lot)
        y -= 8

        # Trace: show heats + GRNs down to supplier
        y = _ensure_space(c, y, 14)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, y, "Traceability: Heats & GRN (FIFO)")
        y -= 12
        c.setFont("Helvetica", 10)

        for lh in getattr(lot, "heats", []):
            h = lh.heat if hasattr(lh, "heat") else None
            if not h:
                # fallback if relationship is by id
                h = getattr(db.get(type(lh).__class__.__mro__[-2], "Heat"), "id", None)  # defensive; usually not needed
            if not h:
                continue
            y = _ensure_space(c, y, 14)
            c.drawString(
                2.2 * cm,
                y,
                f"Heat {getattr(h, 'heat_no', '')}: Alloc to lot {float(getattr(lh, 'qty', 0) or 0):.1f} kg, "
                f"Heat Out {float(getattr(h, 'actual_output', 0) or 0):.1f} kg, "
                f"QA {getattr(h, 'qa_status', '') or ''}",
            )
            y -= 12
            for cons in getattr(h, "rm_consumptions", []):
                g = cons.grn
                supp = g.supplier if g else ""
                y = _ensure_space(c, y, 12)
                c.drawString(
                    2.8 * cm,
                    y,
                    f"- {cons.rm_type}: GRN #{cons.grn_id} ({supp}) {float(cons.qty or 0):.1f} kg",
                )
                y -= 12

        y -= 4

    # Signature block
    y = _ensure_space(c, y, 40)
    _hline(c, y)
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Prepared By: _______________________")
    c.drawString(9.5 * cm, y, "Approved By: _______________________")
