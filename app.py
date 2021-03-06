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
t_dur = 6

bound_max = 1.1
bound = st.sidebar.slider("boundary [0, ∞]", 0.2, bound_max, step=0.1, value=0.7)
drift = st.sidebar.slider("drift [-∞, ∞]", -3.0, 3.0, step=0.2, value=0.0)
ic = st.sidebar.slider(
    "start point or bias (0: no bias toward either bound)",
    -bound + 0.1,
    bound - 0.1,
    step=0.1,
    value=0.0,
)
ndt = st.sidebar.slider("non-decision time (s) [0, ∞]", 0.3, 1.5, step=0.2, value=0.3)

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
samp = sol.resample(5000)
behav = samp.to_pandas_dataframe(drop_undecided=True)

rt = behav.groupby("correct").mean().reset_index(drop=True)
rt["proportion_trials"] = behav["correct"].value_counts() / behav.shape[0]
rt = rt.sort_index(ascending=False)
rt.index = ["upper_bound", "lower_bound"]
rt = rt[["proportion_trials", "RT"]]

#%%

st.sidebar.markdown("#### DDM stochastic differential equation")
st.sidebar.latex(r"dX(t) = v \cdot dt + s \cdot dW(t)")
st.sidebar.markdown(
    """
- $dX(t)$: change in evidence X
- $v$: drift rate
- $dt$: change in time (precision)
- $s$: diffusion coefficient (usually 0.1)
- $dW(t)$: noise with mean 0 and variance $s^2 \cdot dt$
"""
)

#%%

code = """from numpy import sqrt
from numpy.random import normal

dt = 0.001  # time step (1 ms)
drift = 0.5
s = 0.1  # diffusion coef
evidence = drift * dt + normal(loc=0, scale=s * sqrt(dt))
"""
st.sidebar.code(code, language="python")

#%%

st.markdown("##### Simulated behavior")
st.table(rt)

dens_container = st.empty()
simulate_button = st.empty()
fig_containers = st.empty()

behav["bound"] = behav["correct"].replace({0: "lower", 1: "upper"})

dens = (
    alt.Chart(behav)
    .transform_density(
        density="RT", groupby=["bound"], as_=["RT", "density"], extent=[0, t_dur]
    )
    .mark_area()
    .encode(
        alt.X("RT:Q", title="reaction time (s)"),
        alt.Y("density:Q"),
        alt.Color("bound:N", legend=None),
        tooltip=["bound"],
    )
)
# dens
# st.sidebar.altair_chart(dens, use_container_width=True)
dens_container.empty()
dens_container.altair_chart(dens, use_container_width=True)

#%%

df = pd.DataFrame()
n = 0
while n < n_trials:
    # for n in range(n_trials + 10000):
    x = model.simulate_trial(cutoff=False, seed=np.random.randint(0, 2**32))
    try:
        idx = np.where(np.abs(x) >= bound)[0][0]
    except IndexError:
        continue
    x[(idx + 1) :] = np.nan
    x[x > bound] = bound
    x[x < -bound] = -bound
    df[n] = x
    n += 1
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
    # dat = df.copy().query("trial <= @t")
    t = str(t)
    dat = df.copy().query("trial == @t")
    if dat.shape[0] == 0:
        print("No data for trial", t)
        return None

    if np.sign(dat.query("evidence.notna()")["evidence"].to_list()[-1]) == 1:
        color = "#f58518"
    else:
        color = "#4c78a8"

    chart = (
        alt.Chart(dat)
        .mark_line(size=2, color=color)
        .encode(
            x=alt.X(
                "time:Q",
                scale=alt.Scale(domain=(xmin, xmax)),
                axis=alt.Axis(grid=False),
                title="time (s)",
            ),
            y=alt.Y(
                "evidence:Q",
                scale=alt.Scale(domain=(-bound_max, bound_max)),
                axis=alt.Axis(grid=False),
            ),
            # color=alt.Color("trial", legend=None),
            tooltip=["time", "evidence"],
        )
    )
    return chart + hline0 + hlineL + hlineU


#%%

simulate_button.empty()
if plot(0) is not None:
    chart = plot(0)
    for i in range(2):
        if plot(i) is not None:
            chart += plot(i + 1)
    fig_containers.altair_chart(chart, use_container_width=True)
if simulate_button.button("Simulate more decisions", key="start"):
    chart = None
    for t in range(n_trials):
        chart = plot(t) if t == 0 else chart + plot(t)
        fig_containers.altair_chart(chart, use_container_width=True)
        time.sleep(0.5)


#%%
