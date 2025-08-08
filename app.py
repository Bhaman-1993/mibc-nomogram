import streamlit as st
import numpy as np, pandas as pd, joblib

st.set_page_config(page_title="MIBC Nomogram (Coxnet)", layout="centered")
st.title("MIBC Nomogram (Coxnet) — 12/24/36-month survival")

endpoint = st.selectbox("Endpoint", ["OS", "DFS"], index=0)
art_path = "os_nomogram_artifacts.joblib" if endpoint=="OS" else "dfs_nomogram_artifacts.joblib"

A = joblib.load(art_path)
pipe = A["pipeline"]
cont = A["continuous_vars"]
cat  = A["categorical_vars"]
pred_times = np.array(A["pred_times"])
S0_vals    = np.array(A["S0_vals"])
scale      = float(A["points_to_lp_scale"])
pt_unit_vec= np.array(A["points_per_unit"])

pre = pipe.named_steps["pre"]
cat_levels = {v: list(cats) for v, cats in zip(cat, pre.named_transformers_["cat"].categories_)}

st.subheader("Enter patient features")
with st.form("inputs"):
    vals = {}
    for v in cont:
        default = 60.0 if v=="age" else 0.0
        vals[v] = st.number_input(v, value=float(default))
    for v in cat:
        opts = cat_levels.get(v, [])
        vals[v] = st.selectbox(v, options=opts, index=0) if opts else st.text_input(v, "")
    submitted = st.form_submit_button("Predict")

def predict_row(d):
    X = pd.DataFrame([d])
    Xd = pre.transform(X)
    # total points = dot(design, points_per_unit)
    pts = float((Xd * pt_unit_vec).sum(axis=1)[0])
    lp = scale * pts
    surv = (S0_vals ** np.exp(lp))     # S(t|x) = S0(t) ** exp(lp)
    return pts, {int(t): float(s) for t, s in zip(pred_times, surv)}

if submitted:
    pts, surv_map = predict_row(vals)
    st.markdown(f"**Total points:** {pts:.1f}")
    st.write({f"{k} mo survival": f"{100*v:.1f}%" for k,v in surv_map.items()})

# Show quick metrics
st.markdown("### Cross-validated performance (from training)")
m = A.get("metrics", {})
if m:
    st.write("C-index:", m["c_index"])
    st.write("AUC @ times:", {int(t): {"cox": round(c,3), "gb": round(g,3)}
                              for t,c,g in zip(m["auc"]["times"], m["auc"]["cox"], m["auc"]["gb"])})
    st.write("Integrated Brier Score:", {k: round(v,3) for k,v in m["ibs"].items()})
