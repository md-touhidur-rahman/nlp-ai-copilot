# src/app/pages/data.py
import streamlit as st
from pathlib import Path

from src.data.loader import load_table
from src.data.schema import detect_schema
from src.data.filtering import filter_sales
from src.data.aggregate import (
    group_sum_by_date,
    group_sum,
    group_sum_by_month,
    group_sum_by_year,
)
from src.data.nl_parser import parse_data_query


st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("📊 Data Explorer")

DATA_DIR = Path("data/datasets/tabular")
csvs = sorted(DATA_DIR.glob("*.csv"))

if not csvs:
    st.warning("No CSV files found in data/datasets/tabular/")
    st.stop()

# ----------------------------
# Dataset selection
# ----------------------------
csv_file = st.selectbox("Select a dataset", csvs, format_func=lambda p: p.name)

df = load_table(csv_file)

st.subheader("Dataset Preview")
st.dataframe(df.head(100))

# ----------------------------
# Schema inspection
# ----------------------------
st.subheader("Detected Schema")
schema = detect_schema(df)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Numeric**")
    st.write(schema["numeric"])
with col2:
    st.markdown("**Categorical**")
    st.write(schema["categorical"])
with col3:
    st.markdown("**Datetime**")
    st.write(schema["datetime"])

# ----------------------------
# Simple filter + aggregation demo
# ----------------------------
st.divider()
st.subheader("Filter + Aggregate (Demo)")

store = st.number_input("Store number", min_value=1, value=1)
family = st.text_input("Family (exact match)", value="GROCERY I")
start = st.text_input("Start date", value="2013-01-01")
end = st.text_input("End date", value="2013-02-01")

compare_mode = st.selectbox(
    "Compare mode",
    [
        "Daily sales trend (group by date)",
        "Compare families (group by family)",
        "Monthly sales (group by month)",
        "Yearly sales (group by year)",
    ],
)

if st.button("Run filter + aggregate"):
    with st.spinner("Filtering data..."):
        # if comparing families/month/year, don't lock to a single family
        family_filter = family if compare_mode == "Daily sales trend (group by date)" else None

        df_filtered = filter_sales(
            df,
            store_nbr=store,
            family=family_filter,
            start=start,
            end=end,
        )

    st.write(f"Rows after filter: {len(df_filtered)}")

    if len(df_filtered) == 0:
        st.warning("No data for this filter.")
    else:
        if compare_mode == "Daily sales trend (group by date)":
            out = group_sum_by_date(df_filtered, date_col="date", value_col="sales")
            st.subheader("Daily Sales")
            st.dataframe(out)
            st.line_chart(out.set_index("date")["sales"])

        elif compare_mode == "Compare families (group by family)":
            out = group_sum(df_filtered, group_col="family", value_col="sales")
            st.subheader("Sales by Family (comparison)")
            st.dataframe(out.head(50))
            st.bar_chart(out.set_index("family")["sales"].head(20))

        elif compare_mode == "Monthly sales (group by month)":
            out = group_sum_by_month(df_filtered, date_col="date", value_col="sales")
            st.subheader("Monthly Sales")
            st.dataframe(out)
            st.bar_chart(out.set_index("month")["sales"])

        else:  # yearly
            out = group_sum_by_year(df_filtered, date_col="date", value_col="sales")
            st.subheader("Yearly Sales")
            st.dataframe(out)
            st.bar_chart(out.set_index("year")["sales"])

# ----------------------------
# NL → Computation
# ----------------------------
st.divider()
st.subheader("Ask a data question (NL → Computation)")

nl_query = st.text_input(
    "Examples:\n"
    "- 'Show daily sales for store 1 family Grocery I in jan 2013'\n"
    "- 'Top families by sales for store 1 in jan 2013'\n"
    "- 'which month sales maximum in 2016'\n"
    "- 'which year sales maximum'\n",
    key="data_nl_query",
)

run_nl = st.button("Run NL query", key="run_nl_query")

if run_nl:
    if not nl_query.strip():
        st.warning("Please type a question first.")
        st.stop()

    with st.spinner("Parsing your question..."):
        dq = parse_data_query(nl_query)

    st.markdown("### Parsed intent")
    st.write(dq)

    # Use parser mode ONLY (don’t re-detect in Streamlit)
    mode = getattr(dq, "mode", "daily")
    st.caption(f"Detected mode: {mode}")

    # Basic validation
    if dq.metric != "sales":
        st.warning("Right now this supports metric='sales' only.")
        st.stop()

    try:
        with st.spinner("Running computation..."):
            # Only lock to a family for DAILY mode
            family_filter = dq.family if mode == "daily" else None

            df_filtered = filter_sales(
                df,
                store_nbr=dq.store_nbr,
                family=family_filter,
                start=dq.start,
                end=dq.end,
            )

        rows = len(df_filtered)
        st.markdown("### Result")
        st.write(f"Rows after filter: {rows}")

        if rows == 0:
            st.warning("No rows matched your query. Try different store/family/date range.")
            st.stop()

        # ----------------------------
        # Branch by detected mode
        # ----------------------------
        if mode == "monthly":
            out = group_sum_by_month(df_filtered, date_col="date", value_col="sales")
            total_sales = float(out["sales"].sum())
            top_month = str(out.iloc[0]["month"])
            top_sales = float(out.iloc[0]["sales"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Months", int(out.shape[0]))
            c2.metric("Total sales", f"{total_sales:,.2f}")
            c3.metric("Top month", f"{top_month} ({top_sales:,.2f})")

            st.subheader("Monthly Sales (computed)")
            st.dataframe(out)
            st.bar_chart(out.set_index("month")["sales"])
            st.caption("Computed from the dataset (not from documents).")

        elif mode == "yearly":
            out = group_sum_by_year(df_filtered, date_col="date", value_col="sales")
            total_sales = float(out["sales"].sum())
            top_year = str(out.iloc[0]["year"])
            top_sales = float(out.iloc[0]["sales"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Years", int(out.shape[0]))
            c2.metric("Total sales", f"{total_sales:,.2f}")
            c3.metric("Top year", f"{top_year} ({top_sales:,.2f})")

            st.subheader("Yearly Sales (computed)")
            st.dataframe(out)
            st.bar_chart(out.set_index("year")["sales"])
            st.caption("Computed from the dataset (not from documents).")

        elif mode == "family":
            out = group_sum(df_filtered, group_col="family", value_col="sales")
            total_sales = float(out["sales"].sum())
            top_family = str(out.iloc[0]["family"])
            top_sales = float(out.iloc[0]["sales"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Families", int(out.shape[0]))
            c2.metric("Total sales", f"{total_sales:,.2f}")
            c3.metric("Top family", f"{top_family} ({top_sales:,.2f})")

            st.subheader("Sales by Family (computed)")
            st.dataframe(out.head(50))
            st.bar_chart(out.set_index("family")["sales"].head(20))
            st.caption("Computed from the dataset (not from documents).")

        else:
            # daily
            out = group_sum_by_date(df_filtered, date_col="date", value_col="sales")
            total_sales = float(out["sales"].sum())
            avg_sales = float(out["sales"].mean())
            days = int(len(out))

            c1, c2, c3 = st.columns(3)
            c1.metric("Days", days)
            c2.metric("Total sales", f"{total_sales:,.2f}")
            c3.metric("Avg sales/day", f"{avg_sales:,.2f}")

            st.subheader("Daily Sales (computed)")
            st.dataframe(out)
            st.line_chart(out.set_index("date")["sales"])
            st.caption("Computed from the dataset (not from documents).")

    except Exception as e:
        st.error("Something went wrong while computing the result.")
        st.code(str(e))
