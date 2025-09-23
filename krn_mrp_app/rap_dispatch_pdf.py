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

# ---------- NEW: compact CoA table helper ----------
def _draw_coa_table(c, lot, start_y):
    """
    Draws CoA (QA) as tables for Chemistry, Physical and PSD.
    Always renders headers; empty values shown as '—'.
    Returns new y.
    """
    width, height = A4
    x_left = 2*cm
    y = start_y

    def cell(x, y, w, h, text="", bold=False):
        c.rect(x, y-h, w, h)           # box
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawCentredString(x + w/2.0, y - h + 3.5 + 6, str(text))

    # 1) Heat/QA header mini-table
    y = _ensure_space(c, y, 38)
    c.setFont("Helvetica-Bold", 12); c.drawString(x_left, y, "QA snapshot — Lot " + (lot.lot_no or "")); y -= 10
    c.setFont("Helvetica", 10); y -= 2

    col_w = [3.5*cm, 3.0*cm, 8.0*cm]
    header = ["Heat", "QA", "Notes"]
    x = x_left; h = 16
    for i, t in enumerate(header):
        cell(x, y, col_w[i], h, t, bold=True); x += col_w[i]
    y -= h
    # rows
    for lh in getattr(lot, "heats", []):
        hobj = getattr(lh, "heat", None)
        heat_no = getattr(hobj, "heat_no", "")
        qa = getattr(hobj, "qa_status", "") or "—"
        notes = getattr(hobj, "qa_notes", "") if hasattr(hobj, "qa_notes") else "—"
        x = x_left
        for w, txt in zip(col_w, [heat_no, qa, notes or "—"]):
            cell(x, y, w, h, txt); x += w
        y -= h

    y -= 12

    # 2) Chemistry table
    y = _ensure_space(c, y, 70)
    c.setFont("Helvetica-Bold", 12); c.drawString(x_left, y, "Chemistry"); y -= 6
    c.setFont("Helvetica", 10); y -= 4
    chem = getattr(lot, "chemistry", None)

    cols = [("C","c"),("Si","si"),("S","s"),("P","p"),("Cu","cu"),("Ni","ni"),("Mn","mn"),("Fe","fe")]
    cw = 2.0*cm; h = 16
    # header row
    x = x_left
    for label, _ in cols:
        cell(x, y, cw, h, label, bold=True); x += cw
    y -= h
    # value row
    x = x_left
    for _, attr in cols:
        val = getattr(chem, attr, None) if chem else None
        cell(x, y, cw, h, (val if (val not in (None, "")) else "—")); x += cw
    y -= h

    y -= 12

    # 3) Physical table
    y = _ensure_space(c, y, 48)
    c.setFont("Helvetica-Bold", 12); c.drawString(x_left, y, "Physical"); y -= 6
    c.setFont("Helvetica", 10); y -= 4
    phys = getattr(lot, "phys", None)

    cols = [("AD (g/cc)", "ad"), ("Flow (s/50g)", "flow")]
    cw = 3.5*cm; h = 16
    x = x_left
    for label, _ in cols:
        cell(x, y, cw, h, label, bold=True); x += cw
    y -= h
    x = x_left
    for _, attr in cols:
        val = getattr(phys, attr, None) if phys else None
        cell(x, y, cw, h, (val if (val not in (None, "")) else "—")); x += cw
    y -= h

    y -= 12

    # 4) PSD table
    y = _ensure_space(c, y, 70)
    c.setFont("Helvetica-Bold", 12); c.drawString(x_left, y, "PSD"); y -= 6
    c.setFont("Helvetica", 10); y -= 4
    psd = getattr(lot, "psd", None)

    cols = [("+212","p212"),("+180","p180"),("-180+150","n180p150"),("-150+75","n150p75"),("-75+45","n75p45"),("-45","n45")]
    cw = 2.7*cm; h = 16
    x = x_left
    for label, _ in cols:
        cell(x, y, cw, h, label, bold=True); x += cw
    y -= h
    x = x_left
    for _, attr in cols:
        val = getattr(psd, attr, None) if psd else None
        cell(x, y, cw, h, (val if (val not in (None, "")) else "—")); x += cw
    y -= h

    return y - 8
# ---------- /NEW ----------


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

    # Manual rate placeholder + note
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Final Sell Rate (manual): __________________  ₹/kg"); y -= 16
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2*cm, y, "This Document is Dispatch Note use for Invoice Purpose Only"); y -= 14

    # Annexure – QA & Trace
    y = _ensure_space(c, y, 22)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Annexure – QA Certificate & Traceability"); y -= 12
    c.setFont("Helvetica", 10)

    for it in items:
        lot = it.lot
        y = _ensure_space(c, y, 16)
        c.drawString(2*cm, y, f"Lot: {lot.lot_no}  |  Grade: {lot.grade or ''}  |  QA: {getattr(lot, 'qa_status', '') or ''}  |  Dispatch Qty: {float(it.qty or 0):.1f} kg")
        y -= 12

        # Trace: heats + GRNs
        for lh in getattr(lot, "heats", []):
            h = lh.heat
            y = _ensure_space(c, y, 14)
            c.drawString(2.2*cm, y, f"Heat {h.heat_no}: Alloc to lot {float(lh.qty or 0):.1f} kg, Heat Out {float(h.actual_output or 0):.1f} kg, QA {h.qa_status or ''}")
            y -= 12
            for cons in h.rm_consumptions:
                g = cons.grn
                supp = g.supplier if g else ""
                y = _ensure_space(c, y, 12)
                c.drawString(2.8*cm, y, f"- {cons.rm_type}: GRN #{cons.grn_id} ({supp}) {float(cons.qty or 0):.1f} kg")
                y -= 12

        # NEW: start the formal CoA table for this lot (on a fresh page if needed)
        y = _ensure_space(c, y, 220)
        y = _draw_coa_table(c, lot, y)

        y -= 8

    # Signature block
    y = _ensure_space(c, y, 40)
    _hline(c, y); y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Prepared By: _______________________")
    c.drawString(9.5*cm, y, "Approved By: _______________________")
