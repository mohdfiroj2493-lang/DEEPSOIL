import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.fft import rfft, irfft, rfftfreq


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="1D Site Response (Streamlit)", layout="wide")


# -----------------------------
# Constants / helpers
# -----------------------------
LAYER_NUM_COLS = ["Thickness_m", "Vs_mps", "Rho_kgm3", "GammaRef", "Dmin", "Dmax"]
MOTION_MAX_POINTS_WARN = 1_000_000


def as_numeric_series(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().any():
        bad_idx = list(out[out.isna()].index[:10])
        raise ValueError(f"Column '{name}' has non-numeric/blank values at rows: {bad_idx} (showing up to 10).")
    return out


def clean_layers_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and are numeric.
    Drop rows that are completely empty. Keep only rows with Thickness > 0.
    """
    df = df.copy()

    # Ensure all required columns exist (if user deletes columns in editor somehow)
    for c in LAYER_NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Force numeric
    for c in LAYER_NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only positive thickness
    df = df[df["Thickness_m"] > 0].reset_index(drop=True)

    return df


def validate_layers(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No valid layers found. Add at least one layer with Thickness_m > 0.")

    # Check NaNs
    if df[LAYER_NUM_COLS].isna().any().any():
        bad = df[LAYER_NUM_COLS].isna().any()
        raise ValueError(f"Non-numeric/blank values in: {list(bad[bad].index)}")

    if (df["Thickness_m"] <= 0).any():
        raise ValueError("All Thickness_m must be > 0.")
    if (df["Vs_mps"] <= 0).any():
        raise ValueError("All Vs_mps must be > 0.")
    if (df["Rho_kgm3"] <= 0).any():
        raise ValueError("All Rho_kgm3 must be > 0.")
    if (df["GammaRef"] <= 0).any():
        raise ValueError("All GammaRef must be > 0 (e.g., 0.001 = 0.1%).")
    if ((df["Dmin"] < 0) | (df["Dmin"] > 0.3)).any():
        raise ValueError("Dmin should be between 0 and 0.3 (fraction, e.g., 0.02).")
    if ((df["Dmax"] < 0) | (df["Dmax"] > 0.5)).any():
        raise ValueError("Dmax should be between 0 and 0.5 (fraction).")
    if (df["Dmax"] < df["Dmin"]).any():
        raise ValueError("Dmax must be >= Dmin for all layers.")


def load_motion_csv(file_bytes: bytes):
    """
    Accepts:
      - 1 column: acceleration (m/s^2); dt must be provided by user
      - 2+ columns: first is time(s), second is accel(m/s^2); dt inferred by median diff(time)
    """
    data = pd.read_csv(io.BytesIO(file_bytes))
    if data.shape[1] < 1:
        raise ValueError("CSV appears empty.")

    if data.shape[1] == 1:
        acc = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        if np.isnan(acc).any():
            raise ValueError("Acceleration column contains non-numeric values.")
        return None, acc

    t = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    if np.isnan(t).any() or np.isnan(a).any():
        raise ValueError("Time or acceleration columns contain non-numeric values.")

    if len(t) < 3:
        raise ValueError("Time series too short.")

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer a valid dt from time column.")
    return dt, a


def gg_and_damping(gamma: np.ndarray, gamma_ref: np.ndarray, dmin: np.ndarray, dmax: np.ndarray):
    """
    Simple smooth placeholder curves:
      G/Gmax = 1 / (1 + gamma/gamma_ref)
      D = Dmin + (Dmax - Dmin) * gamma/(gamma + gamma_ref)
    """
    gamma = np.maximum(gamma, 1e-12)
    gamma_ref = np.maximum(gamma_ref, 1e-12)

    x = gamma / gamma_ref
    gg = 1.0 / (1.0 + x)
    d = dmin + (dmax - dmin) * (x / (1.0 + x))
    return gg, d


def build_solver_layers(layers_df: pd.DataFrame, base_vs: float, base_rho: float):
    """
    Return arrays for solver including half-space as last "layer" with large thickness.
    """
    h = as_numeric_series(layers_df["Thickness_m"], "Thickness_m").to_numpy(dtype=float)
    vs = as_numeric_series(layers_df["Vs_mps"], "Vs_mps").to_numpy(dtype=float)
    rho = as_numeric_series(layers_df["Rho_kgm3"], "Rho_kgm3").to_numpy(dtype=float)
    damp = as_numeric_series(layers_df["Damping"], "Damping").to_numpy(dtype=float)

    # Append half-space as a very thick layer (absorbing-ish approximation)
    h2 = np.concatenate([h, [1e6]])
    vs2 = np.concatenate([vs, [float(base_vs)]])
    rho2 = np.concatenate([rho, [float(base_rho)]])
    damp2 = np.concatenate([damp, [float(np.clip(damp[-1] if len(damp) else 0.02, 0.0, 0.5))]])

    return h2, vs2, rho2, np.clip(damp2, 0.0, 0.5)


def transfer_function_sh(layers_df: pd.DataFrame, freqs: np.ndarray, base_vs: float, base_rho: float):
    """
    1D SH-wave transfer matrix (vertical incidence), traction-free at surface.
    Returns complex transfer function from base within-motion to surface.
    """
    # ensure input freqs are 1D
    freqs = np.asarray(freqs).reshape(-1)
    if freqs.size == 0:
        raise ValueError("Frequency array is empty.")

    h, vs, rho, damp = build_solver_layers(layers_df, base_vs, base_rho)

    if not (len(h) == len(vs) == len(rho) == len(damp)):
        raise ValueError("Internal error: layer vectors have inconsistent lengths.")

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(vs)) or np.any(~np.isfinite(rho)) or np.any(~np.isfinite(damp)):
        raise ValueError("Non-finite values detected in layer properties.")

    if np.any(h <= 0) or np.any(vs <= 0) or np.any(rho <= 0):
        raise ValueError("Thickness, Vs, and Rho must be positive.")

    # Complex Vs via small-strain viscoelastic approximation
    # v* = v * sqrt(1 + 2 i D)
    vs_c = vs * np.sqrt(1.0 + 2j * damp)
    Z = rho * vs_c

    w = 2.0 * np.pi * freqs
    tf = np.zeros_like(freqs, dtype=np.complex128)

    for i, wi in enumerate(w):
        if wi == 0:
            tf[i] = 1.0 + 0j
            continue

        M = np.eye(2, dtype=np.complex128)

        for hj, Zj, vsj in zip(h, Z, vs_c):
            kj = wi / vsj
            kh = kj * hj

            c = np.cos(kh)
            s = np.sin(kh)

            # Layer matrix
            Lj = np.array(
                [
                    [c, (s / (Zj * wi))],
                    [-(Zj * wi) * s, c],
                ],
                dtype=np.complex128,
            )
            M = Lj @ M

        M22 = M[1, 1]
        if not np.isfinite(M22.real) or not np.isfinite(M22.imag) or np.abs(M22) < 1e-14:
            tf[i] = np.nan + 0j
            continue

        # traction-free at surface: tau_top = 0
        # u_top / u_base = M11 - M12*(M21/M22)
        M11, M12 = M[0, 0], M[0, 1]
        M21 = M[1, 0]
        tf[i] = M11 - M12 * (M21 / M22)

    # Replace any NaNs with 0 to avoid downstream crashes; still show warnings elsewhere
    tf = np.where(np.isfinite(tf), tf, 0.0 + 0j)
    return tf


def newmark_psa(acc_g: np.ndarray, dt: float, periods: np.ndarray, zeta: float = 0.05):
    """
    Pseudo-acceleration spectrum via Newmark-beta (average acceleration).
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
            p = -ag
            peff = p + (a0 * u + a2 * ud + a3 * udd) + c * (a1 * u + a4 * ud + a5 * udd)

            un = peff / keff
            uddn = a0 * (un - u) - a2 * ud - a3 * udd
            udn = ud + dt * ((1 - gamma) * udd + gamma * uddn)

            u, ud, udd = un, udn, uddn
            umax = max(umax, abs(u))

        psa[i] = (w * w) * umax

    return psa


def run_linear(layers_df: pd.DataFrame, acc_base: np.ndarray, dt: float, base_vs: float, base_rho: float, damping: float):
    df = layers_df.copy()
    df["Damping"] = float(np.clip(damping, 0.0, 0.5))

    n = len(acc_base)
    freqs = rfftfreq(n, d=dt)
    A_base = rfft(acc_base)

    tf = transfer_function_sh(df, freqs, base_vs, base_rho)
    acc_surf = irfft(A_base * tf, n=n)
    return df, freqs, tf, acc_surf


def run_equiv_linear(layers_df: pd.DataFrame, acc_base: np.ndarray, dt: float, base_vs: float, base_rho: float,
                     n_iter: int = 6, strain_scale: float = 0.65):
    """
    Robust "equivalent-linear style" iteration:
    - start from Dmin
    - update G/Gmax and D using a simple gamma proxy
    """
    df = layers_df.copy()
    df["GoverGmax"] = 1.0
    df["Damping"] = df["Dmin"].astype(float).clip(0.0, 0.5)

    n = len(acc_base)
    freqs = rfftfreq(n, d=dt)
    A_base = rfft(acc_base)

    for _ in range(int(n_iter)):
        vs0 = df["Vs_mps"].to_numpy(dtype=float)
        gg = np.clip(df["GoverGmax"].to_numpy(dtype=float), 1e-6, 1.0)
        df["Vs_eff"] = vs0 * np.sqrt(gg)

        # Use effective Vs for transfer; store in Vs_mps for solver
        tmp = df.copy()
        tmp["Vs_mps"] = tmp["Vs_eff"]

        tf = transfer_function_sh(tmp, freqs, base_vs, base_rho)
        acc_surf = irfft(A_base * tf, n=n)

        # Proxy strain: use peak surface velocity / Vs_eff per layer
        vel = np.cumsum(acc_surf) * dt
        vel = vel - np.mean(vel)
        vpk = float(np.max(np.abs(vel)))

        gamma_rep = strain_scale * (vpk / np.maximum(df["Vs_eff"].to_numpy(dtype=float), 1e-6))
        gamma_ref = df["GammaRef"].to_numpy(dtype=float)
        dmin = df["Dmin"].to_numpy(dtype=float)
        dmax = df["Dmax"].to_numpy(dtype=float)

        gg_new, d_new = gg_and_damping(gamma_rep, gamma_ref, dmin, dmax)

        # Relaxation
        alpha = 0.6
        df["GoverGmax"] = alpha * gg_new + (1 - alpha) * df["GoverGmax"].to_numpy(dtype=float)
        df["Damping"] = alpha * d_new + (1 - alpha) * df["Damping"].to_numpy(dtype=float)

        df["Damping"] = df["Damping"].astype(float).clip(0.0, 0.5)

    # Final surface with final properties
    vs0 = df["Vs_mps"].to_numpy(dtype=float)
    gg = np.clip(df["GoverGmax"].to_numpy(dtype=float), 1e-6, 1.0)
    df["Vs_eff"] = vs0 * np.sqrt(gg)

    tmp = df.copy()
    tmp["Vs_mps"] = tmp["Vs_eff"]

    tf = transfer_function_sh(tmp, freqs, base_vs, base_rho)
    acc_surf = irfft(A_base * tf, n=n)

    return df, freqs, tf, acc_surf


# -----------------------------
# Session state initialization
# -----------------------------
if "layers_raw" not in st.session_state:
    st.session_state.layers_raw = pd.DataFrame(
        {
            "Thickness_m": [5.0, 10.0],
            "Vs_mps": [200.0, 350.0],
            "Rho_kgm3": [1800.0, 1900.0],
            "GammaRef": [0.001, 0.001],  # 0.1%
            "Dmin": [0.02, 0.02],
            "Dmax": [0.15, 0.12],
        }
    )

if "motion" not in st.session_state:
    st.session_state.motion = None
    st.session_state.dt = 0.01

if "result" not in st.session_state:
    st.session_state.result = None


# -----------------------------
# UI
# -----------------------------
st.title("1D Site Response (Streamlit) — Robust MVP")

with st.sidebar:
    st.header("Workflow")
    page = st.radio("Go to", ["1) Soil Profile", "2) Input Motion", "3) Run", "4) Results"], index=0)

    st.header("Half-space (Base Rock)")
    base_vs = st.number_input("Base Vs (m/s)", min_value=50.0, value=800.0, step=10.0)
    base_rho = st.number_input("Base density ρ (kg/m³)", min_value=1000.0, value=2200.0, step=10.0)

    st.caption("Half-space is modeled as a very thick last layer to stabilize the transfer matrix.")


# -----------------------------
# Page 1: Soil Profile
# -----------------------------
if page == "1) Soil Profile":
    st.subheader("Soil Profile (Layers)")

    st.write(
        "Enter layers with **positive thickness**. All values must be numeric. "
        "GammaRef is reference strain (unitless), e.g., 0.001 = 0.1%."
    )

    edited = st.data_editor(
        st.session_state.layers_raw,
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.layers_raw = edited

    # Clean + validate and show issues here (no solver run)
    try:
        layers = clean_layers_table(edited)
        validate_layers(layers)
        st.success(f"Profile OK. {len(layers)} layer(s) will be used.")
    except Exception as e:
        st.error(str(e))
        layers = clean_layers_table(edited)

    # Plot Vs profile for cleaned layers
    if not layers.empty and layers["Vs_mps"].notna().all():
        z = np.r_[0.0, np.cumsum(layers["Thickness_m"].to_numpy(dtype=float))]
        vs = layers["Vs_mps"].to_numpy(dtype=float)

        fig, ax = plt.subplots()
        for i in range(len(vs)):
            ax.plot([vs[i], vs[i]], [z[i], z[i + 1]])
            if i < len(vs) - 1:
                ax.plot([vs[i], vs[i + 1]], [z[i + 1], z[i + 1]])
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Vs Profile")
        st.pyplot(fig)

    st.info("Tip: Don’t leave blanks. If you paste values, avoid commas (e.g., use 1800 not 1,800).")


# -----------------------------
# Page 2: Input Motion
# -----------------------------
elif page == "2) Input Motion":
    st.subheader("Input Motion")

    dt_user = st.number_input(
        "Time step dt (s) (used only for 1-column CSV)",
        min_value=0.0001,
        value=float(st.session_state.dt),
        step=0.001,
        format="%.4f",
    )
    st.session_state.dt = float(dt_user)

    up = st.file_uploader("Upload acceleration CSV", type=["csv"])
    if up is not None:
        try:
            inferred_dt, acc = load_motion_csv(up.getvalue())
            if inferred_dt is not None:
                st.session_state.dt = inferred_dt
                st.info(f"Inferred dt = {inferred_dt:.6f} s from time column.")
            st.session_state.motion = acc.astype(float)

            if len(acc) > MOTION_MAX_POINTS_WARN:
                st.warning(f"Large motion ({len(acc)} points). Runs may be slow on Streamlit Cloud.")

            st.success(f"Loaded motion with {len(acc)} points.")
        except Exception as e:
            st.error(str(e))
            st.session_state.motion = None

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
# Page 3: Run
# -----------------------------
elif page == "3) Run":
    st.subheader("Run Analysis")

    if st.session_state.motion is None:
        st.warning("Upload an input motion first (Input Motion page).")

    analysis_type = st.selectbox("Analysis type", ["Linear (fixed damping)", "Equivalent-linear style (iterative)"])

    col1, col2, col3 = st.columns(3)
    with col1:
        lin_damp = st.number_input("Linear damping (fraction)", min_value=0.0, max_value=0.30, value=0.05, step=0.01)
    with col2:
        n_iter = st.number_input("Equiv-linear iterations", min_value=1, max_value=25, value=6, step=1)
    with col3:
        strain_scale = st.number_input("Strain scale factor", min_value=0.1, max_value=2.0, value=0.65, step=0.05)

    if st.button("Run"):
        try:
            if st.session_state.motion is None:
                raise ValueError("No input motion loaded.")

            # Clean + validate layers
            layers = clean_layers_table(st.session_state.layers_raw)
            validate_layers(layers)

            # Ensure motion is numeric
            acc_base = np.asarray(st.session_state.motion, dtype=float).reshape(-1)
            if acc_base.size < 10:
                raise ValueError("Input motion too short.")
            if np.isnan(acc_base).any() or np.any(~np.isfinite(acc_base)):
                raise ValueError("Input motion contains NaN or non-finite values.")

            dt = float(st.session_state.dt)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid dt.")

            # Run chosen analysis
            if analysis_type == "Linear (fixed damping)":
                layers_out, freqs, tf, acc_surf = run_linear(
                    layers, acc_base, dt, base_vs, base_rho, float(lin_damp)
                )
                mode = "linear"
            else:
                layers_out, freqs, tf, acc_surf = run_equiv_linear(
                    layers, acc_base, dt, base_vs, base_rho, int(n_iter), float(strain_scale)
                )
                mode = "equiv_linear"

            st.session_state.result = {
                "mode": mode,
                "layers_out": layers_out,
                "freqs": freqs,
                "tf": tf,
                "acc_base": acc_base,
                "acc_surf": acc_surf,
                "dt": dt,
                "base_vs": base_vs,
                "base_rho": base_rho,
            }
            st.success("Run completed. Go to Results.")

        except Exception as e:
            # Show full error (not redacted) inside the app
            st.error(f"Run failed: {e}")
            st.session_state.result = None


# -----------------------------
# Page 4: Results
# -----------------------------
else:
    st.subheader("Results")

    res = st.session_state.result
    if res is None:
        st.info("No results yet. Run an analysis first.")
    else:
        acc_base = res["acc_base"]
        acc_surf = res["acc_surf"]
        dt = res["dt"]
        freqs = res["freqs"]
        tf = res["tf"]
        layers_out = res["layers_out"].copy()

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

        fig, ax = plt.subplots()
        ax.semilogx(freqs[1:], np.abs(tf[1:]))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|T(f)|")
        ax.set_title("Transfer Function Magnitude (Base → Surface)")
        st.pyplot(fig)

        periods = np.logspace(np.log10(0.02), np.log10(5.0), 60)
        psa_base = newmark_psa(acc_base, dt, periods, zeta=0.05)
        psa_surf = newmark_psa(acc_surf, dt, periods, zeta=0.05)

        fig, ax = plt.subplots()
        ax.semilogx(periods, psa_base, label="Base")
        ax.semilogx(periods, psa_surf, label="Surface")
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("PSA (m/s²), 5%")
        ax.set_title("Response Spectra (PSA)")
        ax.legend()
        st.pyplot(fig)

        # Ensure columns exist
        if "Vs_eff" not in layers_out.columns:
            layers_out["Vs_eff"] = layers_out["Vs_mps"]
        if "GoverGmax" not in layers_out.columns:
            layers_out["GoverGmax"] = 1.0
        if "Damping" not in layers_out.columns:
            layers_out["Damping"] = np.nan

        st.markdown("### Final Layer Properties Used")
        show_cols = ["Thickness_m", "Vs_mps", "Vs_eff", "Rho_kgm3", "GoverGmax", "Damping", "GammaRef", "Dmin", "Dmax"]
        st.dataframe(layers_out[show_cols], use_container_width=True)

        # Downloads
        out_csv = layers_out[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download layer properties (CSV)", data=out_csv, file_name="layers_out.csv", mime="text/csv")

        surf_df = pd.DataFrame({"t_s": t, "acc_surface_mps2": acc_surf})
        st.download_button(
            "Download surface acceleration (CSV)",
            data=surf_df.to_csv(index=False).encode("utf-8"),
            file_name="acc_surface.csv",
            mime="text/csv",
        )

        st.caption(
            "Engineering note: This is a robust starter implementation. Validate against benchmarks before design use. "
            "Equivalent-linear here uses a simplified strain proxy; production tools compute depth-dependent strains."
        )
