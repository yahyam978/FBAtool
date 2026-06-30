"""
Amazon FBA Weekly Sales & Restock Analyzer
-------------------------------------------
Reads the Amazon "FBA Inventory / Sales" CSV report (the standard FBA
inventory health / restock report exported from Seller Central) and produces:

1. Restock recommendations for the next bi-weekly (نصف شهرية) shipment.
2. Price action suggestions (increase / decrease / keep) based on sales
   velocity vs. weeks of cover.
3. An interactive dashboard you can re-run every week with a fresh export.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
    pointing at app.py. No secrets/API keys are required since you upload
    the CSV manually each week.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="تحليل مبيعات ومخزون أمازون", layout="wide")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

NUMERIC_COLS = [
    "available", "inbound-quantity", "inbound-working", "inbound-shipped",
    "inbound-received", "units-shipped-t7", "units-shipped-t30",
    "units-shipped-t60", "units-shipped-t90", "your-price", "sales-price",
    "lowest-price-new-plus-shipping", "sell-through", "days-of-supply",
    "weeks-of-cover-t30", "weeks-of-cover-t90", "Total Reserved Quantity",
    "unfulfillable-quantity",
]


@st.cache_data
def load_report(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_metrics(df: pd.DataFrame, shipment_cycle_days: int, lead_time_days: int,
                     safety_days: int, low_cover_weeks: float, high_cover_weeks: float,
                     price_up_pct: float, price_down_pct: float) -> pd.DataFrame:
    d = df.copy()

    # --- Sales velocity (units/day), weighting recent (t7) and stable (t30) windows
    d["units-shipped-t7"] = d["units-shipped-t7"].fillna(0)
    d["units-shipped-t30"] = d["units-shipped-t30"].fillna(0)
    d["units-shipped-t90"] = d["units-shipped-t90"].fillna(0)

    daily_t7 = d["units-shipped-t7"] / 7
    daily_t30 = d["units-shipped-t30"] / 30
    daily_t90 = d["units-shipped-t90"] / 90

    # blended velocity: favor t7 (recent trend) but stabilize with t30/t90
    d["avg-daily-sales"] = (0.5 * daily_t7 + 0.35 * daily_t30 + 0.15 * daily_t90)

    # --- Inventory currently available or already on the way
    d["available"] = d["available"].fillna(0).clip(lower=0)
    for c in ["inbound-quantity", "inbound-working", "inbound-shipped", "inbound-received"]:
        if c in d.columns:
            d[c] = d[c].fillna(0)
    d["total-inbound"] = d[["inbound-quantity", "inbound-working", "inbound-shipped", "inbound-received"]].max(axis=1)
    d["on-hand-and-incoming"] = d["available"] + d["total-inbound"]

    # --- Target coverage: time until this shipment is replaced by the next one
    # (shipment cycle) + transit/processing lead time + safety buffer
    target_days = shipment_cycle_days + lead_time_days + safety_days
    d["target-days-of-supply"] = target_days

    # --- Restock recommendation
    needed_units = d["avg-daily-sales"] * target_days - d["on-hand-and-incoming"]
    d["recommended-restock-qty"] = needed_units.apply(lambda x: int(np.ceil(x)) if x > 0 else 0)

    # don't recommend restocking dead-stock items (no sales at all in 90 days)
    d.loc[(d["units-shipped-t90"] == 0) & (d["units-shipped-t30"] == 0), "recommended-restock-qty"] = 0

    # current days of cover with what's on hand
    d["current-days-of-cover"] = np.where(
        d["avg-daily-sales"] > 0,
        d["on-hand-and-incoming"] / d["avg-daily-sales"],
        np.where(d["on-hand-and-incoming"] > 0, 9999, 0)
    )
    d["current-weeks-of-cover"] = d["current-days-of-cover"] / 7

    # --- Priority flag for the shipment
    def priority(row):
        if row["available"] <= 0 and row["avg-daily-sales"] > 0:
            return "🔴 نفذ من المخزون - أولوية قصوى"
        if row["recommended-restock-qty"] > 0 and row["current-weeks-of-cover"] < low_cover_weeks:
            return "🟠 أولوية عالية"
        if row["recommended-restock-qty"] > 0:
            return "🟡 يحتاج توريد"
        return "🟢 مخزون كافٍ"

    d["restock-priority"] = d.apply(priority, axis=1)

    # --- Price recommendation
    def price_action(row):
        if row["avg-daily-sales"] <= 0:
            if row["on-hand-and-incoming"] > 0:
                return "خفض السعر (راكد - لا مبيعات)"
            return "لا يوجد إجراء"
        cover = row["current-weeks-of-cover"]
        if cover < low_cover_weeks:
            return f"رفع السعر تجريبيًا (+{price_up_pct:.0f}%) - الطلب مرتفع والمخزون منخفض"
        if cover > high_cover_weeks:
            return f"خفض السعر (-{price_down_pct:.0f}%) - مخزون زائد وحركة بطيئة"
        return "إبقاء السعر كما هو"

    d["price-recommendation"] = d.apply(price_action, axis=1)

    def suggested_price(row):
        base = row.get("your-price", np.nan)
        if pd.isna(base) or base <= 0:
            return np.nan
        if "رفع" in row["price-recommendation"]:
            return round(base * (1 + price_up_pct / 100), 2)
        if "خفض" in row["price-recommendation"]:
            return round(base * (1 - price_down_pct / 100), 2)
        return base

    d["suggested-price"] = d.apply(suggested_price, axis=1)

    return d


def make_restock_sheet(d: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "sku": "SKU",
        "product-name": "اسم المنتج",
        "available": "المتاح حاليًا",
        "total-inbound": "في الطريق",
        "avg-daily-sales": "متوسط المبيعات اليومية",
        "current-weeks-of-cover": "أسابيع التغطية الحالية",
        "recommended-restock-qty": "الكمية الموصى بشحنها",
        "restock-priority": "الأولوية",
    }
    out = d[list(cols.keys())].rename(columns=cols)
    out["متوسط المبيعات اليومية"] = out["متوسط المبيعات اليومية"].round(2)
    out["أسابيع التغطية الحالية"] = out["أسابيع التغطية الحالية"].round(1)
    out = out.sort_values("الكمية الموصى بشحنها", ascending=False)
    return out


def make_price_sheet(d: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "sku": "SKU",
        "product-name": "اسم المنتج",
        "your-price": "السعر الحالي",
        "suggested-price": "السعر المقترح",
        "units-shipped-t7": "مبيعات آخر 7 أيام",
        "units-shipped-t30": "مبيعات آخر 30 يوم",
        "current-weeks-of-cover": "أسابيع التغطية",
        "price-recommendation": "التوصية",
    }
    out = d[list(cols.keys())].rename(columns=cols)
    out["أسابيع التغطية"] = out["أسابيع التغطية"].round(1)
    out = out[out["التوصية"] != "لا يوجد إجراء"]
    out = out.sort_values("مبيعات آخر 7 أيام", ascending=False)
    return out


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    engine = None
    for candidate in ("xlsxwriter", "openpyxl"):
        try:
            __import__(candidate)
            engine = candidate
            break
        except ImportError:
            continue
    if engine is None:
        raise RuntimeError(
            "لا توجد مكتبة لإنشاء ملفات Excel (xlsxwriter أو openpyxl). "
            "تأكد من وجودها في requirements.txt وأعد تشغيل التطبيق (Reboot app)."
        )
    with pd.ExcelWriter(buf, engine=engine) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("📦 تحليل المبيعات الأسبوعي ومخزون أمازون FBA")
st.caption("ارفع تقرير المخزون/المبيعات الأسبوعي من Amazon Seller Central للحصول على توصيات التوريد والتسعير")

with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    shipment_cycle_days = st.number_input(
        "دورة الشحن (أيام) - كل كم يوم ترسل شحنة جديدة", min_value=1, value=15, step=1,
        help="عندك شحنتين شهريًا تقريبًا، أي كل 15 يومًا"
    )
    lead_time_days = st.number_input(
        "مدة استلام الشحنة بمخازن أمازون (أيام)", min_value=0, value=5, step=1
    )
    safety_days = st.number_input(
        "أيام أمان إضافية (احتياطي)", min_value=0, value=5, step=1
    )
    st.markdown("---")
    low_cover_weeks = st.slider(
        "أقل عدد أسابيع تغطية (تحت هذا الرقم = ارفع السعر / أولوية شحن عالية)",
        min_value=0.5, max_value=8.0, value=2.0, step=0.5
    )
    high_cover_weeks = st.slider(
        "أعلى عدد أسابيع تغطية (فوق هذا الرقم = اخفض السعر)",
        min_value=4.0, max_value=20.0, value=8.0, step=0.5
    )
    st.markdown("---")
    price_up_pct = st.slider("نسبة رفع السعر %", min_value=1, max_value=30, value=5)
    price_down_pct = st.slider("نسبة خفض السعر %", min_value=1, max_value=30, value=8)

uploaded = st.file_uploader("ارفع ملف تقرير أمازون (CSV)", type=["csv"])

if uploaded is None:
    st.info("⬆️ ارفع ملف الـ CSV الأسبوعي من تقرير المخزون في Seller Central للبدء.")
    st.stop()

raw_df = load_report(uploaded)
st.success(f"تم تحميل التقرير: {len(raw_df)} منتج")

result = compute_metrics(
    raw_df, shipment_cycle_days, lead_time_days, safety_days,
    low_cover_weeks, high_cover_weeks, price_up_pct, price_down_pct
)

tab1, tab2, tab3 = st.tabs(["📦 خطة التوريد القادمة", "💰 توصيات التسعير", "📊 نظرة عامة"])

with tab1:
    st.subheader("المنتجات والكميات المطلوب شحنها في التوريدة القادمة")
    restock_df = make_restock_sheet(result)
    needs_restock = restock_df[restock_df["الكمية الموصى بشحنها"] > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("عدد المنتجات المطلوب توريدها", len(needs_restock))
    c2.metric("إجمالي الوحدات المطلوبة", int(needs_restock["الكمية الموصى بشحنها"].sum()))
    c3.metric("منتجات نفذت تمامًا", int((restock_df["المتاح حاليًا"] == 0).sum()))
    st.dataframe(restock_df, use_container_width=True, height=500)

with tab2:
    st.subheader("توصيات تعديل الأسعار حسب حركة المبيعات")
    price_df = make_price_sheet(result)
    st.dataframe(price_df, use_container_width=True, height=500)

with tab3:
    st.subheader("نظرة عامة على المخزون")
    st.dataframe(
        result[["sku", "product-name", "available", "units-shipped-t7", "units-shipped-t30",
                "current-weeks-of-cover", "restock-priority"]]
        .rename(columns={
            "sku": "SKU", "product-name": "اسم المنتج", "available": "المتاح",
            "units-shipped-t7": "مبيعات 7 أيام", "units-shipped-t30": "مبيعات 30 يوم",
            "current-weeks-of-cover": "أسابيع التغطية", "restock-priority": "الحالة",
        }),
        use_container_width=True, height=500
    )

st.markdown("---")
excel_bytes = to_excel_bytes({
    "خطة التوريد": make_restock_sheet(result),
    "توصيات التسعير": make_price_sheet(result),
})
st.download_button(
    "⬇️ تحميل التقرير الكامل (Excel)",
    data=excel_bytes,
    file_name="amazon_restock_pricing_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
