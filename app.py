import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="JoSAA College Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOADERS ----------------
@st.cache_resource
def load_models():
    open_model = joblib.load("model/opening_rank_model.pkl")
    close_model = joblib.load("model/closing_rank_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    return open_model, close_model, encoders

@st.cache_data
def load_data():
    return pd.read_csv("data/final/jossa_features.csv")

@st.cache_data
def load_raw_data():
    return pd.read_csv("data/final/final_jossa_dataset.csv")

@st.cache_data
def load_feature_data():
    return pd.read_csv("data/final/jossa_features.csv")

open_model, close_model, encoders = load_models()
df = load_data()
raw_df = load_raw_data()       # for trends (2021–2025)
df = load_feature_data()       # for predictions


# ---------------- UTILS ----------------
def decode(df, encoders):
    for col in ["institute", "branch", "quota", "seat_type", "gender"]:
        if col in encoders:
            df[col] = encoders[col].inverse_transform(df[col])
    return df

decoded_raw_df = decode(raw_df.copy(), encoders)
decoded_df = decode(df.copy(), encoders)


def rank_to_chance(user_rank, open_rank, close_rank):
    if user_rank <= open_rank:
        return "Safe", 0.9
    if user_rank <= close_rank:
        return "Moderate", 0.6
    if user_rank <= close_rank * 1.1:
        return "Risky", 0.3
    return "Very Risky", 0.1

decoded_df = decode(df.copy(), encoders)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Home", "Data Overview", "Data Analysis", "Closing Rank Trends", "Predict Colleges", "Model Performance",  "Methodology"]
)

# ================= HOME =================
if page == "Home":
    st.title("JoSAA College Predictor")
    st.write(
        "A machine-learning based system that predicts **future JoSAA cutoffs** "
        "and estimates **college admission chances** using historical trends."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Institutes", decoded_df["institute"].nunique())
    col2.metric("Branches", decoded_df["branch"].nunique())
    col3.metric("Years of Data", 5)
    col4.metric("Seat Records", f"{len(decoded_df):,}")

    st.markdown("---")

    st.subheader("How to Use This Tool")
    st.markdown("""
    - Enter your **rank**
    - View **Safe / Moderate / Risky** colleges
    - Analyze **closing rank trends** before making choices  
    - Use it as **decision support**, not a guarantee
    """)

# ================= DATA OVERVIEW =================
elif page == "Data Overview":
    st.header("Dataset Overview")

    st.subheader("Sample Records")
    st.dataframe(
        decoded_df[
            ["year", "round", "institute", "branch", "close_rank"]
        ].head(25),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Closing Rank Trend Over Years (Competition Intensity)")

    trend_summary = (
        decoded_raw_df
        .groupby("year")["close_rank"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )

    fig = px.line(
        trend_summary,
        x="year",
        y="median",
        markers=True,
        title="Median Closing Rank by Year (Lower = More Competitive)",
        labels={"median": "Median Closing Rank", "year": "Year"}
    )

    # Add IQR band
    fig.add_traces([
        px.line(
            trend_summary, x="year", y="q25"
        ).data[0],
        px.line(
            trend_summary, x="year", y="q75"
        ).data[0]
    ])

    fig.data[1].line.dash = "dash"
    fig.data[2].line.dash = "dash"

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        hovermode="x unified",
        legend_title="Metric",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Median closing rank reflects overall competition. "
        "Dashed lines show the middle 50% of cutoffs (IQR)."
    )
elif page == "Data Analysis":
    st.header("Exploratory Data Analysis (JoSAA Cutoffs)")
    st.write(
        "This section explores historical JoSAA cutoff data to understand "
        "competition trends, round behaviour, and branch-level insights."
    )

    tabs = st.tabs([
        "Overview",
        "Year-wise Competition",
        "Round-wise Behaviour",
        "Institute & Branch Insights",
        "Key Conclusions"
    ])

    # ---------------- TAB 1: OVERVIEW ----------------
    with tabs[0]:
        st.subheader("Cutoff Stability & Volatility Analysis")

        st.write(
            "This analysis measures how **stable or volatile closing ranks** are "
            "across years. Stable cutoffs are more predictable, while volatile ones "
            "carry higher uncertainty."
        )

        # ---------------- Institute Stability ----------------
        inst_volatility = (
            decoded_raw_df
            .groupby(["institute", "year"])["close_rank"]
            .median()
            .reset_index()
            .groupby("institute")["close_rank"]
            .std()
            .dropna()
            .sort_values()
            .reset_index(name="std_dev")
            .head(10)
        )

        fig = px.bar(
            inst_volatility,
            x="std_dev",
            y="institute",
            orientation="h",
            title="Top 10 Most Stable Institutes (Lowest Yearly Variation)",
            labels={"std_dev": "Std. Dev of Closing Rank"}
        )

        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "Lower variation means cutoffs are consistent year-to-year, "
            "making predictions more reliable."
        )

        st.markdown("---")

        # ---------------- Branch Volatility ----------------
        branch_volatility = (
            decoded_raw_df
            .groupby(["branch", "year"])["close_rank"]
            .median()
            .reset_index()
            .groupby("branch")["close_rank"]
            .std()
            .dropna()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="std_dev")
        )

        fig = px.bar(
            branch_volatility,
            x="std_dev",
            y="branch",
            orientation="h",
            title="Top 10 Most Volatile Branches (High Yearly Variation)",
            labels={"std_dev": "Std. Dev of Closing Rank"}
        )

        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "High volatility branches show unpredictable demand shifts and "
            "are harder to forecast accurately."
        )

    # ---------------- TAB 2: YEAR-WISE COMPETITION ----------------
    with tabs[1]:
        st.subheader("Competition Trend Over Years")

        year_trend = (
            decoded_raw_df
            .groupby("year")["close_rank"]
            .median()
            .reset_index()
        )

        fig = px.line(
            year_trend,
            x="year",
            y="close_rank",
            markers=True,
            title="Median Closing Rank by Year (Lower = More Competition)",
            labels={"close_rank": "Median Closing Rank"}
        )

        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⬇ A downward trend indicates increasing competition. "
            "⬆ An upward trend suggests easing cutoffs."
        )

    # ---------------- TAB 3: ROUND-WISE BEHAVIOUR ----------------
    with tabs[2]:
        st.subheader("How Cutoffs Change Across Rounds")

        round_trend = (
            decoded_raw_df
            .groupby("round")["close_rank"]
            .median()
            .reset_index()
        )

        fig = px.line(
            round_trend,
            x="round",
            y="close_rank",
            markers=True,
            title="Median Closing Rank by Round",
            labels={"close_rank": "Median Closing Rank", "round": "JoSAA Round"}
        )

        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Later rounds usually show relaxed cutoffs, "
            "but the magnitude varies across years and branches."
        )

    # ---------------- TAB 4: INSTITUTE & BRANCH INSIGHTS ----------------
    with tabs[3]:
        st.subheader("Most Competitive Institutes & Branches")

        col1, col2 = st.columns(2)

        with col1:
            inst_comp = (
                decoded_raw_df
                .groupby("institute")["close_rank"]
                .median()
                .sort_values()
                .head(10)
                .reset_index()
            )

            fig = px.bar(
                inst_comp,
                x="close_rank",
                y="institute",
                orientation="h",
                title="Top 10 Most Competitive Institutes",
                labels={"close_rank": "Median Closing Rank"}
            )

            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            branch_comp = (
                decoded_raw_df
                .groupby("branch")["close_rank"]
                .median()
                .sort_values()
                .head(10)
                .reset_index()
            )

            fig = px.bar(
                branch_comp,
                x="close_rank",
                y="branch",
                orientation="h",
                title="Top 10 Most Competitive Branches",
                labels={"close_rank": "Median Closing Rank"}
            )

            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 5: KEY CONCLUSIONS ----------------
    with tabs[4]:
        st.subheader("Key Data-Driven Conclusions")

        st.markdown("""
        **1️ Competition is increasing over time**  
        Median closing ranks show a downward trend in recent years, indicating
        higher demand for top institutes and branches.

        **2️ Later rounds usually help, but not always**  
        While later JoSAA rounds generally relax cutoffs, highly competitive
        branches show minimal movement after early rounds.

        **3️ Branch matters more than institute in many cases**  
        Some mid-tier institutes with popular branches have tighter cutoffs
        than top institutes with less demanded branches.

        **4️ Cutoff behaviour is not uniform**  
        Year-to-year variation exists due to seat matrix changes, policy shifts,
        and applicant preferences.

        **5️ Predictive modelling is justified**  
        The presence of temporal patterns and consistent trends validates the
        use of machine learning models for cutoff prediction.
        """)

        st.success(
            "These insights form the foundation for the predictive models "
            "used in the College Predictor."
        )

# ================= CLOSING RANK TRENDS =================
elif page == "Closing Rank Trends":
    st.header("Closing Rank Trends (Year-wise by Round)")

    st.write(
        "This graph shows **how closing ranks change across JoSAA rounds**, "
        "with **separate lines for each year**. "
        "It helps answer whether waiting for later rounds helped in different years."
    )

    col1, col2 = st.columns(2)

    with col1:
        institute = st.selectbox(
            "Select Institute",
            sorted(decoded_df["institute"].unique())
        )

    with col2:
        branches = decoded_df[
            decoded_df["institute"] == institute
        ]["branch"].unique()

        branch = st.selectbox(
            "Select Branch",
            sorted(branches)
        )

    trend_df = decoded_raw_df[
        (decoded_raw_df["institute"] == institute) &
        (decoded_raw_df["branch"] == branch)
    ]


    if trend_df.empty:
        st.warning("No data available for this selection.")
    else:
        # Aggregate across categories → worst (max) closing rank per round-year
        trend_df = (
            trend_df
            .groupby(["year", "round"], as_index=False)
            .agg({"close_rank": "max"})
            .sort_values(["year", "round"])
        )

        fig = px.line(
            trend_df,
            x="round",
            y="close_rank",
            color="year",
            markers=True,
            title=f"Round-wise Closing Rank Trend<br>{institute} – {branch}",
            labels={
                "round": "JoSAA Round",
                "close_rank": "Closing Rank",
                "year": "Year"
            }
        )

        # Lower rank = better → invert axis
        fig.update_yaxes(autorange="reversed")

        fig.update_layout(
            hovermode="x unified",
            height=550,
            legend_title="Year"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each line represents a year. "
            "Steeper upward slopes indicate stronger relaxation in later rounds."
        )


# ================= PREDICTION =================
elif page == "Predict Colleges":
    st.header("Predict Your College Chances")

    user_rank = st.number_input(
        "Enter your JEE Rank",
        min_value=1,
        step=1
    )

    if user_rank > 0:
        latest = (
            df.sort_values("year")
              .groupby(["institute","branch","quota","seat_type","gender","round"])
              .tail(1)
              .copy()
        )

        latest["year"] = 2026

        feature_cols = [c for c in latest.columns if c not in ["open_rank","close_rank"]]

        latest["pred_open"] = open_model.predict(latest[feature_cols])
        latest["pred_close"] = close_model.predict(latest[feature_cols])

        latest["pred_open"] = latest["pred_open"].clip(lower=1)
        latest["pred_close"] = latest["pred_close"].clip(lower=1)

        chances = latest.apply(
            lambda r: rank_to_chance(user_rank, r["pred_open"], r["pred_close"]),
            axis=1
        )

        latest["chance"] = chances.apply(lambda x: x[0])
        latest["confidence"] = chances.apply(lambda x: x[1])

        latest = decode(latest, encoders)

        result = latest.sort_values("confidence", ascending=False)

        st.markdown("---")
        st.subheader("All Colleges Grouped by Chance")

        chance_order = ["Safe", "Moderate", "Risky", "Very Risky"]

        for chance in chance_order:
            subset = result[result["chance"] == chance]

            st.markdown(f"## {chance}")

            if subset.empty:
                st.info(f"No colleges fall under **{chance}** category for your rank.")
            else:
                st.dataframe(
                    subset[
                        ["institute", "branch", "pred_open", "pred_close", "confidence"]
                    ].sort_values("confidence", ascending=False),
                    use_container_width=True,
                    height=300
                )


# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":
    st.header("Model Performance Evaluation")

    st.write(
        "This section evaluates the predictive performance of the machine learning "
        "models used to estimate JoSAA opening and closing ranks."
    )

    # ---------------- PREPARE DATA ----------------
    feature_cols = [
        c for c in df.columns
        if c not in ["open_rank", "close_rank"]
    ]

    X = df[feature_cols]
    y_open = df["open_rank"]
    y_close = df["close_rank"]

    open_pred = open_model.predict(X)
    close_pred = close_model.predict(X)

    # ---------------- METRICS ----------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    def compute_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)
        }

    open_metrics = compute_metrics(y_open, open_pred)
    close_metrics = compute_metrics(y_close, close_pred)

    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Opening Rank Model**")
        st.metric("MAE", f"{open_metrics['MAE']:.0f}")
        st.metric("RMSE", f"{open_metrics['RMSE']:.0f}")
        st.metric("R²", f"{open_metrics['R²']:.3f}")

    with col2:
        st.markdown("**Closing Rank Model**")
        st.metric("MAE", f"{close_metrics['MAE']:.0f}")
        st.metric("RMSE", f"{close_metrics['RMSE']:.0f}")
        st.metric("R²", f"{close_metrics['R²']:.3f}")

    st.markdown("---")

    # ---------------- ACTUAL VS PREDICTED ----------------
    st.subheader("Actual vs Predicted Closing Rank")

    sample_df = pd.DataFrame({
        "Actual": y_close,
        "Predicted": close_pred
    }).sample(5000, random_state=42)

    fig = px.scatter(
        sample_df,
        x="Actual",
        y="Predicted",
        opacity=0.4,
        title="Predicted vs Actual Closing Rank"
    )

    fig.add_shape(
        type="line",
        x0=sample_df["Actual"].min(),
        y0=sample_df["Actual"].min(),
        x1=sample_df["Actual"].max(),
        y1=sample_df["Actual"].max(),
        line=dict(color="red", dash="dash")
    )

    fig.update_layout(
        xaxis_title="Actual Closing Rank",
        yaxis_title="Predicted Closing Rank",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Points closer to the diagonal indicate more accurate predictions."
    )

    # ---------------- ERROR DISTRIBUTION ----------------
    st.markdown("---")
    st.subheader("Prediction Error Distribution")

    errors = close_pred - y_close

    fig = px.histogram(
        errors,
        nbins=50,
        title="Distribution of Closing Rank Prediction Errors"
    )

    fig.update_layout(
        xaxis_title="Prediction Error (Predicted − Actual)",
        yaxis_title="Frequency",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "A centered error distribution suggests unbiased predictions."
    )

    # ---------------- INTERPRETATION ----------------
    st.markdown("---")
    st.subheader("Interpretation")

    st.markdown("""
    - **Mean Absolute Error (MAE)** indicates average deviation from actual cutoffs.
    - **R² score** measures how well historical trends explain cutoff variation.
    - Errors are approximately symmetric, suggesting no systematic bias.
    - The model is better suited for **relative comparison** than exact rank prediction.
    """)


# ================= METHODOLOGY =================
elif page == "Methodology":
    st.header("Methodology & Notes")

    st.markdown("""
    **Model:** XGBoost Regressors  
    **Targets:** Opening Rank & Closing Rank  

    **Approach**
    - Learns year-to-year cutoff movement
    - Uses lag & rolling features
    - Predicts numeric cutoffs instead of labels

    **Why trends matter**
    - Sudden rank jumps indicate changing demand
    - Stable trends give safer counselling signals

    **Disclaimer**
    This tool provides **probabilistic guidance**, not guarantees.
    JoSAA policies and seat matrices can change.
    """)

# ---------------- FOOTER ----------------
def about_the_coder():
    # We use a non-indented string to prevent Markdown from treating it as code
    html_code = """
    <style>
    .coder-card {
        background-color: transparent;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 20px;
        display: flex;
        align-items: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .coder-img {
        width: 100px; /* Slightly larger for better visibility */
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #FF4B4B; /* Streamlit Red */
        margin-right: 25px;
        flex-shrink: 0; /* Prevents image from shrinking */
    }
    .coder-info h3 {
        margin: 0;
        font-family: 'Source Sans Pro', sans-serif;
        color: inherit;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .coder-info p {
        margin: 10px 0;
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    .social-links {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .social-links a {
        text-decoration: none;
        color: #FF4B4B;
        font-weight: bold;
        font-size: 0.95rem;
        transition: color 0.3s;
    }
    .social-links a:hover {
        color: #ff2b2b;
        text-decoration: underline;
    }
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .coder-card {
            flex-direction: column;
            text-align: center;
            padding: 15px;
        }
        .coder-img {
            margin-right: 0;
            margin-bottom: 15px;
            width: 80px;
            height: 80px;
        }
        .social-links {
            justify-content: center;
        }
    }
    </style>  
    <div class="coder-card">
        <img src="https://ui-avatars.com/api/?name=Yash+Vasudeva&size=120&background=FF4B4B&color=fff&bold=true&rounded=true" class="coder-img" alt="Yash Vasudeva"/>
        <div class="coder-info">
            <h3>Developed by Yash Vasudeva</h3>
            <p>
                Results-driven Data & AI Professional skilled in <b>Data Analytics</b>, 
                <b>Machine Learning</b>, and <b>Deep Learning</b>. 
                Passionate about transforming raw data into business value and building intelligent solutions.
            </p>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/yash-vasudeva/" target="_blank">LinkedIn</a>
                <a href="https://github.com/yashvasudeva1" target="_blank">GitHub</a>
                <a href="mailto:vasudevyash@gmail.com">Contact</a>
                <a href="https://yashvasudeva.vercel.app/" target="_blank">Portfolio</a>
            </div>
        </div>
    </div>
    """
        
    st.markdown(html_code, unsafe_allow_html=True)

st.divider()

if __name__ == "__main__":
    about_the_coder()