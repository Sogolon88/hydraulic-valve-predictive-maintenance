import numpy as np
import pandas as pd
import streamlit as st

from app_pages.shared import N_TRAIN, VALVE_LEVELS, mean_pressure_start, raw_data

st.title("Données du banc hydraulique")
st.caption(f"Un cycle = 60 s de mesures. Les {N_TRAIN} premiers cycles servent à construire "
           "le modèle, les suivants au test final (découpage de l'énoncé).")

ps2, fs1, profile = raw_data()
n_cycles = len(profile)

with st.container(horizontal=True):
    st.metric("Cycles", n_cycles, border=True)
    st.metric("Mesures PS2 par cycle", ps2.shape[1], help="Pression (bar), 100 Hz", border=True)
    st.metric("Mesures FS1 par cycle", fs1.shape[1], help="Débit (l/min), 10 Hz", border=True)
    st.metric("Cycles avec vanne optimale", f"{100 * (profile['valve'] == 100).mean():.1f} %",
              border=True)

left, right = st.columns(2)
with left:
    st.subheader("État de la vanne")
    counts = (profile["valve"].value_counts().reindex(VALVE_LEVELS)
              .rename_axis("vanne (%)").rename("cycles").reset_index())
    counts["vanne (%)"] = counts["vanne (%)"].astype(str) + " %"
    st.bar_chart(counts, x="vanne (%)", y="cycles", sort=False,
                 alt="Nombre de cycles par état de la vanne")
with right:
    st.subheader("Répartition train / test")
    partie = pd.Series(np.where(profile.index < N_TRAIN, "train", "test"), index=profile.index)
    st.dataframe(pd.crosstab(profile["valve"], partie, normalize="columns")[["train", "test"]]
                 .mul(100).round(1).sort_index(ascending=False).rename_axis("vanne (%)"),
                 column_config={c: st.column_config.NumberColumn(format="%.1f %%")
                                for c in ["train", "test"]},
                 alt="Pourcentage de cycles par état de la vanne dans le train et le test")

st.subheader("Début de cycle")
st.caption("Pression PS2 moyenne sur les cycles du train : elle monte plus tard quand la vanne est usée.")
st.line_chart(mean_pressure_start(), x_label="temps depuis le début du cycle (s)",
              y_label="pression PS2 moyenne (bar)",
              alt="Pression PS2 moyenne en début de cycle pour chaque état de la vanne")

st.subheader("Explorer un cycle")
cycle = st.number_input("Numéro du cycle", min_value=0, max_value=n_cycles - 1, value=0, step=1)
valve = profile["valve"].iloc[cycle]
with st.container(horizontal=True):
    st.badge(f"vanne à {valve} %", color="green" if valve == 100 else "orange",
             icon=":material/check_circle:" if valve == 100 else ":material/warning:")
    st.badge("train" if cycle < N_TRAIN else "test", color="gray")

ps2_col, fs1_col = st.columns(2)
ps2_col.line_chart(pd.DataFrame({"PS2 (bar)": ps2.iloc[cycle].values},
                                index=pd.Index(np.arange(ps2.shape[1]) / 100, name="temps (s)")),
                   x_label="temps (s)", y_label="pression PS2 (bar)",
                   alt=f"Pression PS2 du cycle {cycle}")
fs1_col.line_chart(pd.DataFrame({"FS1 (l/min)": fs1.iloc[cycle].values},
                                index=pd.Index(np.arange(fs1.shape[1]) / 10, name="temps (s)")),
                   x_label="temps (s)", y_label="débit FS1 (l/min)",
                   alt=f"Débit FS1 du cycle {cycle}")
