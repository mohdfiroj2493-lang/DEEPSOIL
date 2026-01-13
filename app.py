import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.fft import rfft, irfft, rfftfreq

# -----------------------------
# Utilities
# -----------------------------
def _as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def validate_layers(df: pd.DataFrame):
    required = ["Thickness_m", "Vs_mps", "Rho_kgm3", "GammaRef", "Dmin", "Dmax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if (df["Thickness_m"] <= 0).any():
        return False, "All Thickness_m must be > 0"
    if (df["Vs_mps"] <= 0).any():
        return False, "All Vs_mps must be > 0"
    if (df["Rho_kgm3"] <= 0).any():
        return False, "All Rho_kgm3 must be > 0"
    if (df["GammaRef"] <= 0).any():
        return False, "All GammaRef must be > 0 (reference strain, e.g., 0.001 = 0.1%)"
    if ((df["Dmin"] < 0) | (df["Dmin"] > 0.3)).any():
        return False, "Dmin should be between 0 and 0.3 (fraction, e.g., 0.02)"
    if ((df["Dmax"] < 0) | (df["Dmax"] > 0.5)).any():
        return False, "Dmax should be between 0 and 0.5 (fraction)"
    if (df["Dmax"] < df["Dmin"]).any():
        return False, "Dmax must be >= Dmin for all layers"
    return True, ""

def gg_and_damping(gamma: np.ndarray, gamma_ref: np.ndarray, dmin: np.ndarray, dmax: np.ndarray):
    """
    Simple smooth curves (placeholder):
      G/Gmax = 1 / (1 + gamma/gamma_ref)
      D = Dmin + (Dmax - Dmin) * gamma / (gamma + gamma_ref)
    gamma is shear strain (unitless). gamma_ref (unitless).
    """
    x = np.maximum(gamma, 1e-12) / np.maximum(gamma_ref, 1e-12)
    gg = 1.0 / (1.0 + x)
    d = dmin + (dmax - dmin) * (x / (1.0 + x))
    return gg, d

def transfer_function_sh(layers_df: pd.DataFrame, freqs: np.ndarray):
    """
    1D SH-wave transfer matrix for vertical incidence.
    Top surface is traction-free.
    Input motion is at the base as an 'outcrop-like' motion is NOT handled here;
    treat base input as within-rock motion at base of layered stack.
    Returns complex transfer function from base acceleration to surface acceleration.
    """
    h = layers_df["Thickness_m"].to_numpy(dtype=float)
    vs = layers_df["Vs_mps"].to_numpy(dtype=float)
    rho = layers_df["Rho_kgm3"].to_numpy(dtype=float)
    damp = layers_df["Damping"].to_numpy(dtype=float)  # fraction

    # Complex shear modulus and wave speed (approx): use complex shear modulus G*(1 + 2iD)
    # Equivalent: complex shear wave velocity v* = v * sqrt(1 + 2iD)
    # For small D, v* ≈ v*(1+iD). We'll use sqrt form for better stability.
    vs_c = vs * np.sqrt(1.0 + 2j * damp)
    Z = rho * vs_c  # impedance (complex)

    w = 2 * np.pi * freqs
    tf = np.zeros_like(freqs, dtype=np.complex128)

    # For each frequency, build global matrix from base to surface
    # State vector: [u; tau] where tau = shear stress
    # Layer matrix:
    #   [ cos(kh)           (1/(Z*omega)) * sin(kh) ]
    #   [ -Z*omega*sin(kh)      cos(kh)            ]
    #
    # Relationship: S_top = M * S_base
    # Surface traction-free: tau_top = 0.
    # Solve for u_top / u_base and use accel = -w^2 u (same ratio for accel).
    for i, wi in enumerate(w):
        if wi == 0:
            tf[i] = 1.0 + 0j
            continue

        M = np.eye(2, dtype=np.complex128)
        for hj, Zj, vsj in zip(h, Z, vs_c):
            kj = wi / vsj
            c = np.cos(kj * hj)
            s = np.sin(kj * hj)
            Lj = np.array(
                [
                    [c, (s / (Zj * wi))],
                    [-(Zj * wi) * s, c],
                ],
                dtype=np.complex128,
            )
            M = Lj @ M

        # S_top = M * S_base
        # Let S_base = [u_b; tau_b]
        # tau_top = 0 = M21*u_b + M22*tau_b => tau_b = -(M21/M22)*u_b
        # u_top = M11*u_b + M12*tau_b
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]

        if abs(M22) < 1e-14:
            tf[i] = np.nan + 0j
            continue

        u_ratio = M11 - M12 * (M21 / M22)
        tf[i] = u_ratio

    return tf

def newmark_response_spectrum(acc_g: np.ndarray, dt: float, periods: np.ndarray, zeta: float = 0.05):
    """
    Compute pseudo-acceleration response spectrum (PSA) with Newmark-beta (average acceleration).
    acc_g in m/s^2.
    """
    beta = 1 / 4
    gamma = 1 / 2

    psa = np.zeros_like(periods, dtype=float)

    for i, T in enumerate(periods):
        if T <= 0:
            psa[i] = np.nan
            continue

        w = 2 * np.pi / T
        k = w * w
        c = 2 * zeta * w

        # normalized mass m=1
        a0 = 1 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1 / (beta * dt)
        a3 = 1 / (2 * beta) - 1
        a4 = gamma / beta - 1
        a5 = dt * (gamma / (2 * beta) - 1)

        keff = k + a0 + a1 * c

        u = 0.0
        ud = 0.0
        udd = 0.0

        umax = 0.0

        for ag in acc_g:
            # Effective load: p = -ag (since m=1)
            p = -ag
            peff = p + (a0 * u + a2 * ud + a3 * udd) + c * (a1 * u + a4 * ud + a5 * udd)

            un = peff / keff
            uddn = a0 * (un - u) - a2 * ud - a3 * udd
            udn = ud + dt * ((1 - gamma) * udd + gamma * uddn)

            u, ud, udd = un, udn, uddn
            umax = max(umax, abs(u))

        # PSA = w^2 * umax
        psa[i] = (w * w) * umax

    return psa

def compute_surface_motion_equiv_linear(layers_df: pd.DataFrame, acc_base: np.ndarray, dt: float,
                                        n_iter: int = 6, strain_scale: float = 0.65):
    """
    Iteratively update per-layer damping and modulus reduction based on an approximate strain measure.

    NOTE: This is a simplified 'equivalent-linear style' implementation:
    - uses transfer function to get surface motion
    - estimates a representative strain for each layer from peak surface velocity and Vs
      (crude; replace with depth-dependent strain computation for production)
    """
    df = layers_df.copy()

    # Initial values
    df["GoverGmax"] = 1.0
    df["Damping"] = df["Dmin"].astype(float).clip(lower=0.0)

    # Base FFT
    n = len(acc_base)
    freqs = rfftfreq(n, d=dt)
    A_base = rfft(acc_base)

    for it in range(n_iter):
        # Update Vs based on G/Gmax (Vs ∝ sqrt(G))
        vs0 = df["Vs_mps"].astype(float).to_numpy()
        gg = df["GoverGmax"].astype(float).to_numpy()
        df["Vs_eff"] = vs0 * np.sqrt(np.clip(gg, 1e-6, 1.0))
        df["Damping"] = df["Damping"].astype(float).clip(0.0, 0.5)

        tf = transfer_function_sh(
            df.rename(columns={"Vs_eff": "Vs_mps"}).assign(Vs_mps=df["Vs_eff"]),
            freqs
        )

        A_surf = A_base * tf
        acc_surf = irfft(A_surf, n=n)

        # Estimate a representative shear strain per layer (very simplified):
        # gamma_rep ≈ strain_scale * (V_peak / Vs_eff)
        # where V_peak from integrating surface acceleration -> velocity (baseline drift removed crudely).
        vel = np.cumsum(acc_surf) * dt
        vel = vel - np.mean(vel)
        vpk = np.max(np.abs(vel))

        gamma_rep = strain_scale * (vpk / np.maximum(df["Vs_eff"].to_numpy(), 1e-6))
        gamma_ref = df["GammaRef"].astype(float).to_numpy()
        dmin = df["Dmin"].astype(float).to_numpy()
        dmax = df["Dmax"].astype(float).to_numpy()

        gg_new, d_new = gg_and_damping(gamma_rep, gamma_ref, dmin, dmax)

        # Relaxation for stability
        alpha = 0.6
        df["GoverGmax"] = alpha * gg_new + (1 - alpha) * df["GoverGmax"].to_numpy()
        df["Damping"] = alpha * d_new + (1 - alpha) * df["Damping"].to_numpy()

    # Final surface motion
    df["Vs_eff"] = df["Vs_mps"].astype(float).to_numpy() * np.sqrt(np.clip(df["GoverGmax"].to_numpy(), 1e-6, 1.0))
    tf = transfer_function_sh(
        df.rename(columns={"Vs_eff": "Vs_mps"}).assign(Vs_mps=df["Vs_eff"]),
        freqs
    )
    acc_surf = irfft(A_base * tf, n=n)

    return df, freqs, tf, acc_surf

def load_motion_csv(file_bytes: bytes):
    """
    Accepts either:
      - single column: accel (m/s^2) with assumed dt entered by user
      - two columns: time(s), accel(m/s^2)
    """
    data = pd.read_csv(io.BytesIO(file_bytes))
    if data.shape[1] == 1:
        return None, data.iloc[:, 0].to_numpy(dtype=float)
    else:
        t = data.iloc[:, 0].to_numpy(dtype=float)
        a = data.iloc[:, 1].to_numpy(dtype=float)
        # infer dt
        dt = float(np.median(np.diff(t)))
        return dt, a

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="1D Site Response (Streamlit)", layout="wide")

st.title("1D Site Response (Streamlit) — DEEPSOIL-like workflow (fresh implementation)")

if "layers" not in st.session_state:
    st.session_state.layers = pd.DataFrame(
        {
            "Thickness_m": [5.0, 10.0, 0.0],  # last row can be 'halfspace' marker if you set thickness=0
            "Vs_mps": [200.0, 350.0, 800.0],
            "Rho_kgm3": [1800.0, 1900.0, 2100.0],
            "GammaRef": [0.001, 0.001, 0.001],  # 0.1%
            "Dmin": [0.02, 0.02, 0.01],
            "Dmax": [0.15, 0.12, 0.05],
        }
    )

if "motion" not in st.session_state:
    st.session_state.motion = None
    st.session_state.dt = 0.01

page = st.sidebar.radio("Workflow", ["1) Soil Profile", "2) Input Motion", "3) Run Analysis", "4) Results"])

# -----------------------------
# Page 1: Soil Profile
# -----------------------------
if page == "1) Soil Profile":
    st.subheader("Soil Profile")

    st.write(
        "Edit your layered profile below. This MVP uses a simple modulus-reduction + damping curve model "
        "driven by `GammaRef`, `Dmin`, `Dmax`."
    )

    df_edit = st.data_editor(
        st.session_state.layers,
        num_rows="dynamic",
        use_container_width=True
    )

    ok, msg = validate_layers(df_edit[df_edit["Thickness_m"] > 0].copy())
    if not ok:
        st.error(msg)
    else:
        st.success("Profile looks valid.")

    st.session_state.layers = df_edit

    # Quick plot of Vs profile
    dfp = df_edit.copy()
    dfp = dfp[dfp["Thickness_m"] > 0].reset_index(drop=True)
    if len(dfp) > 0:
        z = np.r_[0.0, np.cumsum(dfp["Thickness_m"].to_numpy())]
        vs = dfp["Vs_mps"].to_numpy()
        fig, ax = plt.subplots()
        for i in range(len(vs)):
            ax.plot([vs[i], vs[i]], [z[i], z[i+1]])
            if i < len(vs) - 1:
                ax.plot([vs[i], vs[i+1]], [z[i+1], z[i+1]])
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Vs Profile")
        st.pyplot(fig)

# -----------------------------
# Page 2: Input Motion
# -----------------------------
elif page == "2) Input Motion":
    st.subheader("Input Motion")

    st.write("Upload CSV with either:\n- **1 column**: acceleration (m/s²) and you set **dt**\n- **2 columns**: time(s), acceleration(m/s²) and dt is inferred")

    dt_user = st.number_input("Time step dt (s) (used only for 1-column files)", min_value=0.0001, value=float(st.session_state.dt), step=0.001, format="%.4f")
    st.session_state.dt = dt_user

    up = st.file_uploader("Upload acceleration CSV", type=["csv"])
    if up is not None:
        inferred_dt, acc = load_motion_csv(up.getvalue())
        if inferred_dt is not None:
            st.session_state.dt = inferred_dt
            st.info(f"Inferred dt = {inferred_dt:.6f} s from time column.")
        st.session_state.motion = acc
        st.success(f"Loaded motion with {len(acc)} points.")

    if st.session_state.motion is not None:
        acc = st.session_state.motion
        dt = float(st.session_state.dt)
        t = np.arange(len(acc)) * dt

        fig, ax = plt.subplots()
        ax.plot(t, acc)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Base Input Motion")
        st.pyplot(fig)

# -----------------------------
# Page 3: Run Analysis
# -----------------------------
elif page == "3) Run Analysis":
    st.subheader("Run Analysis")

    analysis_type = st.selectbox("Analysis type", ["Linear (fixed damping)", "Equivalent-linear style (iterative)"])

    col1, col2, col3 = st.columns(3)
    with col1:
        base_damping = st.number_input("Linear damping (fraction)", min_value=0.0, max_value=0.30, value=0.05, step=0.01)
    with col2:
        n_iter = st.number_input("Iterations (equiv-linear)", min_value=1, max_value=20, value=6, step=1)
    with col3:
        strain_scale = st.number_input("Strain scale factor", min_value=0.1, max_value=2.0, value=0.65, step=0.05)

    run = st.button("Run")

    if run:
        if st.session_state.motion is None:
            st.error("Please upload an input motion first.")
        else:
            # Prepare layers
            layers = st.session_state.layers.copy()
            layers = layers[layers["Thickness_m"] > 0].reset_index(drop=True)
            ok, msg = validate_layers(layers)
            if not ok:
                st.error(msg)
            else:
                acc_base = st.session_state.motion.astype(float)
                dt = float(st.session_state.dt)

                if analysis_type == "Linear (fixed damping)":
                    layers["GoverGmax"] = 1.0
                    layers["Damping"] = float(base_damping)
                    layers["Vs_eff"] = layers["Vs_mps"]

                    n = len(acc_base)
                    freqs = rfftfreq(n, d=dt)
                    tf = transfer_function_sh(
                        layers.rename(columns={"Vs_eff": "Vs_mps"}).assign(Vs_mps=layers["Vs_eff"]),
                        freqs
                    )
                    acc_surf = irfft(rfft(acc_base) * tf, n=n)

                    st.session_state.result = {
                        "layers_out": layers,
                        "freqs": freqs,
                        "tf": tf,
                        "acc_surf": acc_surf,
                        "analysis": "linear",
                    }
                    st.success("Run completed.")
                else:
                    layers_out, freqs, tf, acc_surf = compute_surface_motion_equiv_linear(
                        layers, acc_base, dt, n_iter=int(n_iter), strain_scale=float(strain_scale)
                    )

                    st.session_state.result = {
                        "layers_out": layers_out,
                        "freqs": freqs,
                        "tf": tf,
                        "acc_surf": acc_surf,
                        "analysis": "equiv_linear",
                    }
                    st.success("Run completed.")

# -----------------------------
# Page 4: Results
# -----------------------------
else:
    st.subheader("Results")

    if "result" not in st.session_state or st.session_state.result is None:
        st.info("Run an analysis first.")
    else:
        res = st.session_state.result
        dt = float(st.session_state.dt)
        acc_base = st.session_state.motion.astype(float)
        acc_surf = res["acc_surf"]
        freqs = res["freqs"]
        tf = res["tf"]
        layers_out = res["layers_out"]

        # Time histories
        t = np.arange(len(acc_base)) * dt

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots()
            ax.plot(t, acc_base)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("a (m/s²)")
            ax.set_title("Base Acceleration")
            st.pyplot(fig)

        with colB:
            fig, ax = plt.subplots()
            ax.plot(t, acc_surf)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("a (m/s²)")
            ax.set_title("Surface Acceleration")
            st.pyplot(fig)

        # Transfer function magnitude
        fig, ax = plt.subplots()
        ax.semilogx(freqs[1:], np.abs(tf[1:]))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|T(f)|")
        ax.set_title("Transfer Function Magnitude (Base → Surface)")
        st.pyplot(fig)

        # Response spectrum
        periods = np.logspace(np.log10(0.02), np.log10(5.0), 60)
        psa_base = newmark_response_spectrum(acc_base, dt, periods, zeta=0.05)
        psa_surf = newmark_response_spectrum(acc_surf, dt, periods, zeta=0.05)

        fig, ax = plt.subplots()
        ax.semilogx(periods, psa_base, label="Base")
        ax.semilogx(periods, psa_surf, label="Surface")
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("PSA (m/s²), 5% damped")
        ax.set_title("Response Spectra (PSA)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Final Layer Properties Used")
        show_cols = ["Thickness_m", "Vs_mps", "Vs_eff", "Rho_kgm3", "GoverGmax", "Damping", "GammaRef", "Dmin", "Dmax"]
        for c in show_cols:
            if c not in layers_out.columns:
                layers_out[c] = np.nan
        st.dataframe(layers_out[show_cols], use_container_width=True)

        # Export
        out_csv = layers_out[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download layer properties (CSV)", data=out_csv, file_name="layers_out.csv", mime="text/csv")

        # Export surface motion
        surf_df = pd.DataFrame({"t_s": t, "acc_surface_mps2": acc_surf})
        st.download_button(
            "Download surface acceleration (CSV)",
            data=surf_df.to_csv(index=False).encode("utf-8"),
            file_name="acc_surface.csv",
            mime="text/csv"
        )

st.caption(
    "Engineering note: This is an educational/starter implementation. For production use, add depth-dependent strain computation, "
    "proper base/outcrop boundary handling, better constitutive models, and verification against published benchmarks."
)
