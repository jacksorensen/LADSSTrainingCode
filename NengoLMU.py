from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_dl
from nengo.utils.filter_design import cont2discrete
model = nengo.Network()
with model:
    theta = 1
    order = 8

    freq = 2
    rms = 0.3
    delay = 0.5

    dt = 0.001
    sim_t = 100
    seed = 0

    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))

    A, B, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method="zoh")


class IdealDelay(nengo.synapses.Synapse):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def m_step(self, shape_in, shape_out, dt, rng, state):
        buffer = deque([0] * int(self.delay / dt))

        def delay_func(t, x):
            buffer.apend(x.copy())
            return buffer.popleft()

        return delay_func


with nengo.Network(seed=seed) as net:
    stim = nengo.Node(
        output=nengo.processes.WhiteSignal(
            high=freq, period=sim_t, rms=rms, y0=0, seed=seed))

    p_stim = nengo.Probe(stim)
    p_ideal = nengo.Probe(stim, synapse=IdealDelay(delay))

with nengo_dl.Simulator(net) as sim:
    sim.run(10)

    plt.figure(figsize=(16, 6))
    plt.plot(sim.trange(), sim.data[p_stim], label='input')
    plt.plot(sim.trange(), sim.data[p_ideal], label='ideal')
    plt.legend()

with net:
    lmu = nengo.Node(size_in=order)
    nengo.Connection(stim, lmu, transform=B, synapse=None)
    nengo.Connection(lmu, lmu, transform=A, synapse=0)
with net:
    ens = nengo.Ensemble(1000, order, neuron_type=nengo.SpikingRectifiedLinear())
    nengo.Connection(lmu, ens, synapse=None)

    out = nengo.Node(size_in=1)

    err_node = nengo.Node(lambda t, x: x if t < sim_t * 0.8 else 0, size_in=1)

    nengo.Connection(stim, err_node, synapse=IdealDelay(delay), transform=-1)
    nengo.Connection(out, err_node, synapse=None)

    learn_conn = nengo.Connection(
        ens, out, function=lambda x: 0, learning_rule_type=nengo.PES(2e-4))

    nengo.Connection(err_node, learn_conn.learning_rule, synapse=None)
    p_out = nengo.Probe(out)

with nengo.Simulator(net) as sim:
    sim.run(sim_t)

t_per_plot = 10

for i in range(sim_t // t_per_plot):
    plot_slice = (sim.trange() >= t_per_plot * i) & (
            sim.trange() < t_per_plot * (i + 1))

    plt.figure(figsize=(16, 6))
    plt.plot(sim.trange()[plot_slice], sim.data[p_stim][plot_slice], label='input')
    plt.plot(sim.trange()[plot_slice], sim.data[p_ideal][plot_slice], label='ideal')
    plt.plot(sim.trange()[plot_slice], sim.data[p_out][plot_slice], label='output')

    if i * t_per_plot < sim_t * 0.8:
        plt.title('Learning On')
    else:
        plt.title('Learning Off')
    plt.ylim([-1, 1])
    plt.legend()


