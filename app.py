# ============================================================
# INDUSTRIAL MACHINE FAILURE PREDICTION
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_model.pkl"
    )

    return model


model = load_model()


# ============================================================
# 3. TITLE
# ============================================================

st.title(
    "⚙️ Industrial Machine Failure Prediction"
)

st.write(
    "Predict whether an industrial machine is likely to fail "
    "based on its operating conditions."
)


st.divider()


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.header(
    "Machine Information"
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


# ============================================================
# 5. MACHINE PARAMETERS
# ============================================================

st.subheader(
    "Enter Machine Parameters"
)

col1, col2 = st.columns(2)


with col1:

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )


with col2:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


# ============================================================
# 6. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
)


# ============================================================
# 7. PREDICTION
# ============================================================

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame(
        {
            "Type": [machine_type],

            "Air temperature [K]": [
                air_temperature
            ],

            "Process temperature [K]": [
                process_temperature
            ],

            "Rotational speed [rpm]": [
                rotational_speed
            ],

            "Torque [Nm]": [
                torque
            ],

            "Tool wear [min]": [
                tool_wear
            ]
        }
    )


    # Make prediction
    prediction = model.predict(
        input_data
    )[0]


    # Get probability
    probability = model.predict_proba(
        input_data
    )[0][1]


    failure_probability = probability * 100


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    st.subheader(
        "Prediction Result"
    )


    if prediction == 1:

        st.error(
            "🔴 Machine Failure Predicted"
        )

        st.warning(
            f"Estimated Failure Probability: "
            f"{failure_probability:.2f}%"
        )

        st.write(
            "The machine operating conditions indicate "
            "a possible failure."
        )


    else:

        st.success(
            "🟢 No Machine Failure Predicted"
        )

        st.info(
            f"Estimated Failure Probability: "
            f"{failure_probability:.2f}%"
        )

        st.write(
            "The machine operating conditions indicate "
            "a lower probability of failure."
        )


    # ========================================================
    # 9. INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "Machine Input Summary"
    )

    display_data = pd.DataFrame(
        {
            "Parameter": [
                "Machine Type",
                "Air Temperature [K]",
                "Process Temperature [K]",
                "Rotational Speed [rpm]",
                "Torque [Nm]",
                "Tool Wear [min]"
            ],

            "Value": [
                machine_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            ]
        }
    )


    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Machine Failure Prediction | "
    "Machine Learning Capstone Project"
)