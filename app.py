import streamlit as st
import numpy as np

# --- الدالة الأصلية ---
def CalculateMomentWithML(input_list):
    bt, bb, hw, tt, tb, tw = input_list
    area_m2 = (hw*tw + bt*tt + bb*tb) / 1e6  
    Weight = area_m2 * 7850  

    X_max = [300,300,400,5,5,5]
    X_min = [150,150,250,3.5,3.5,3.5]
    Y_max = [1,1,1,1,1,1]
    Y_min = [-1,-1,-1,-1,-1,-1]

    normalised_input = []
    for i in range(len(input_list)):
        normalised_input.append(
            (Y_max[i]-Y_min[i])*(input_list[i]-X_min[i]) / (X_max[i] - X_min[i]) + Y_min[i]
        )

    input_array = np.array(normalised_input)

    IW1 = np.array([
        [0.00572662427139449,0.765475186245782,-0.376581821328129,-0.0356168354299637,0.510951913724441,-0.286732023674033],
        [-0.00038292543745906,0.0124990419626003,0.18438401708702,0.0201504467701285,0.0313314715678954,0.0894024706486257],
        [0.0826756516154421,0.0803498886584752,0.0656175771512403,0.143314856402687,0.0331890032227124,-0.138918337003687],
        [0.0205207850011245,-0.58246553628884,0.0703849806747943,0.01822776066042,0.0869954608843683,0.148390478120349],
        [0.0550633595384806,0.123739200472022,0.276852560627938,0.166703822936741,0.0695426865863749,-0.255826305317234],
        [-0.181183855770181,0.188212143760871,0.297329730647406,-0.241760820769866,0.0894002739678121,0.116218865206358],
        [-0.138327894250618,0.00790030708370294,0.105997705187058,-0.170387184230914,0.0173875070809942,0.131441139661752],
        [-0.535232566983595,-0.0308597947018132,-0.0969261634495595,0.239695220581758,-0.018406012081025,0.032431826631276]
    ])
    b1 = np.array([-1.36435212, -0.63093577, -0.65532611, -1.0326336, -0.98025896, 0.477132966, 0.485756574, -1.28729545])

    mid_layer = np.matmul(IW1, input_array) + b1 
    mid_layer_sigmoid = (2/(1+np.exp(-2*mid_layer))-1)

    LW2 = np.array([-0.17049309, 2.986669, 5.836611, -0.25705, -2.3145, -0.91166, 3.201017, -0.29726])
    b2 = np.array([1.510587286])
     
    Moment_normalised = np.matmul(LW2, mid_layer_sigmoid) + b2
    Moment = (Moment_normalised - (-1)) / (1 -(-1)) * (213634.9471958876 - 62055.0401490485) + 62055.0401490485
    return float(Moment), float(Weight)

# --- واجهة Streamlit ---
st.set_page_config(page_title="I-section Calculator", layout="wide")

st.markdown("<h1 style='text-align:center;'>Simply supported a solid monosymmetric I-section steel beam</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Geometric Parameters")
    bt = st.number_input("Top Flange Width (bt) mm", min_value=150.0, max_value=300.0, value=200.0, step=0.1)
    bb = st.number_input("Bottom Flange Width (bb) mm", min_value=150.0, max_value=300.0, value=200.0, step=0.1)
    hw = st.number_input("Web Height (hw) mm", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
    tt = st.number_input("Top Flange Thickness (tt) mm", min_value=3.5, max_value=5.0, value=4.0, step=0.1)
    tb = st.number_input("Bottom Flange Thickness (tb) mm", min_value=3.5, max_value=5.0, value=4.0, step=0.1)
    tw = st.number_input("Web Thickness (tw) mm", min_value=3.5, max_value=5.0, value=4.0, step=0.1)

    if st.button("Calculate"):
        moment, weight = CalculateMomentWithML([bt, bb, hw, tt, tb, tw])
        st.session_state["moment"] = moment
        st.session_state["weight"] = weight

with col2:
    st.subheader("After predicting")
    if "moment" in st.session_state:
        st.success(f"Predicted moment Capacity = {st.session_state['moment']:.2f} kN·mm")
        st.success(f"Weight = {st.session_state['weight']:.4f} kg/m")
    else:
        st.info("Enter values and click Calculate to see results")

    st.image("Form2.jpg", caption="I-section", use_column_width=True)
