# %% load modules

import random
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ddm import Model
from ddm.functions import display_model, fit_adjust_model
from ddm.models import (
    BoundConstant,
    DriftConstant,
    NoiseConstant,
    OverlayNonDecision,
    ICPoint,
)
import ddm


pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)
pd.options.display.float_format = "{:,.3f}".format

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

st.sidebar.markdown("## Diffusion model parameters")


#%%

n_trials = 10
t_dur = 5

bound = st.sidebar.slider("boundary", 0.5, 2.0, step=0.1, value=0.7)
drift = st.sidebar.slider("drift", -0.8, 0.8, step=0.2, value=0.0)
ic = st.sidebar.slider("start point (bias)", -0.4, 0.4, step=0.1, value=0.0)
ndt = st.sidebar.slider("non-decision time", 0.3, 1.0, step=0.2, value=0.3)

xmin = 0
xmax = t_dur

model = Model(
    name="Simple model",
    drift=DriftConstant(drift=drift),
    noise=NoiseConstant(noise=1.0),
    bound=BoundConstant(B=bound),
    overlay=OverlayNonDecision(nondectime=ndt),
    IC=ICPoint(x0=ic),
    dx=0.001,
    dt=0.025,
    T_dur=t_dur,
)
sol = model.solve()
samp = sol.resample(500)
behav = samp.to_pandas_dataframe(drop_undecided=True)

rt = behav.groupby("correct").mean().round(2).reset_index(drop=True)
rt["proportion_trials"] = np.round(behav["correct"].value_counts() / behav.shape[0], 2)
rt = rt.sort_index(ascending=False)
rt.index = ["upper_bound", "lower_bound"]
rt = rt[["proportion_trials", "RT"]]

#%%

st.markdown("##### Simulated behavior")
st.dataframe(rt)

start_button = st.empty()
figcontainer = st.empty()

behav["bound"] = behav["correct"].replace({0: "lower", 1: "upper"})

dens = (
    alt.Chart(behav)
    .transform_density(
        density="RT", groupby=["bound"], as_=["RT", "density"], extent=[0, t_dur]
    )
    .mark_area()
    .encode(alt.X("RT:Q"), alt.Y("density:Q"), alt.Color("bound:N"))
)
# dens
st.sidebar.altair_chart(dens, use_container_width=True)

#%%

df = pd.DataFrame()
for n in range(n_trials):
    x = model.simulate_trial(cutoff=False, seed=np.random.randint(0, 2**32))
    try:
        idx = np.where(np.abs(x) >= bound)[0][0]
    except IndexError:
        continue
    x[(idx + 1) :] = np.nan
    x[x > bound] = bound
    x[x < -bound] = -bound
    df[n] = x
# df


#%%

df.index = np.linspace(0, t_dur, len(x))
df = df.unstack().reset_index(name="value")
df.columns = ["trial", "time", "evidence"]
df["trial"] = df["trial"].astype(str)

hline0 = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(size=1.0)
    .encode(y=alt.Y("y:Q", title="evidence"))
)

hlineL = (
    alt.Chart(pd.DataFrame({"y": [-bound]}))
    .mark_rule(size=2)
    .encode(y=alt.Y("y:Q", title="evidence"))
)

hlineU = (
    alt.Chart(pd.DataFrame({"y": [bound]}))
    .mark_rule(size=2)
    .encode(y=alt.Y("y:Q", title="evidence"))
)


def plot(t=100000):
    ylimit = bound
    # dat = df.copy().query("trial <= @t")
    t = str(t)
    dat = df.copy().query("trial == @t")

    chart = (
        alt.Chart(dat)
        .mark_line(size=2)
        .encode(
            x=alt.X(
                "time:Q",
                scale=alt.Scale(domain=(xmin, xmax)),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "evidence:Q",
                scale=alt.Scale(domain=(-ylimit, ylimit)),
                axis=alt.Axis(grid=False),
            ),
            color=alt.Color("trial", legend=None),
        )
    )
    return chart + hline0 + hlineL + hlineU


#%%

start_button.empty()
if start_button.button("Simulate trials", key="start"):
    for t in range(n_trials):
        chart = plot(t) if t == 0 else chart + plot(t)
        figcontainer.altair_chart(chart, use_container_width=True)
        time.sleep(0.5)


#%%