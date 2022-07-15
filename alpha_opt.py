import nest
import time
from scipy.optimize import bisect

SIM_TIME = 1e+5
DELAY = 1.0
timers = []
st = time.time()
AX_DELAYS = [0.0, 0.5, 0.75, 1.0]
AX_DELAYS = [0.75]
ALPHA = 0.057


def alpha_optimizer(new_alpha):
    rate = 0
    for run in range(1, 2):
        out = 0
        for axd_ in AX_DELAYS:

            s = time.time()

            # For Different runs
            # tp = run if run < 3 else run - 2
            # _delay = DELAY * 10 ** (tp - 1)
            # axd = axd_ if run % 2 == 1 else axd_ * 10

            # Random seed
            nest.SetKernelStatus({
                # 'rng_seed': tp,
                'total_num_virtual_procs': 5})

            # Compiling Own Models
            nest.CopyModel('static_synapse', 'syn_exc', {"weight": 38.5})
            nest.CopyModel('static_synapse', 'syn_inh', {"weight": -192.5})
            nest.CopyModel('stdp_pl_synapse_hom_ax_delay', 'stdp_ax_delay', {"axonal_delay": axd,
                                                                             "delay": DELAY,
                                                                             "alpha": new_alpha,
                                                                             "lambda": 0.1,
                                                                             "mu": 0.4,
                                                                             "tau_plus": 15.0,
                                                                             "weight": 38.5
                                                                             })
            nest.CopyModel('stdp_pl_synapse_hom', 'stdp_hom', {"axonal_delay": axd,
                                                               "delay": _delay,
                                                               "alpha": new_alpha,
                                                               "lambda": 0.1,
                                                               "mu": 0.4,
                                                               "tau_plus": 15.0,
                                                               "weight": 38.5
                                                               })

            nest.CopyModel('poisson_generator', 'psn_exc', {"rate": 8.0})
            nest.CopyModel('poisson_generator', 'psn_inh', {"rate": 7680.0})
            nest.CopyModel('poisson_generator', 'psn_ext', {"rate": 16800.0})

            nest.CopyModel('iaf_psc_alpha_ax_delay', 'iaf_ax_del', {"C_m": 250.0,
                                                                    "E_L": 0.0,
                                                                    "I_e": 0.0,
                                                                    "tau_m": 10.0,
                                                                    "tau_minus": 30.0,
                                                                    "tau_syn_ex": 0.3258,
                                                                    "tau_syn_in": 0.3258,
                                                                    "t_ref": 0.5,
                                                                    "V_reset": 0.0,
                                                                    "V_th": 20.0,
                                                                    "V_m": 5.7})

            # Creating Nodes
            psn_exc = nest.Create("psn_exc")
            psn_inh = nest.Create("psn_inh")
            psn_ext = nest.Create("psn_ext")
            parrot_neurons = nest.Create("parrot_neuron", 3840)
            iaf_alpha = nest.Create("iaf_psc_alpha_ax_delay")
            spikerecorder = nest.Create("spike_recorder")

            # Connecting Nodes
            nest.Connect(psn_exc, parrot_neurons, {"rule": 'all_to_all'}, {'synapse_model': "syn_exc"})
            """if run < 3:
                nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                             {'synapse_model': "stdp_ax_delay"})
            else:
                nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                             {'synapse_model': "stdp_hom"})"""
                             
            nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                             {'synapse_model': "stdp_ax_delay"})
            nest.Connect(psn_inh, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_inh"})
            nest.Connect(psn_ext, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_exc"})
            nest.Connect(iaf_alpha, spikerecorder)

            # Simulation
            nest.Simulate(SIM_TIME)

            # Firing Rate
            print(spikerecorder.n_events, '\tspike recorder events')
            out += spikerecorder.n_events * 1000.0 / SIM_TIME

            # Checking timers
            timers.append(time.time() - s)

            # Getting Weights
            # synapses = nest.GetConnections(target=iaf_alpha)
            # weights = synapses.get('weight')

            # Resetting Kernel
            nest.ResetKernel()

            # Deleting objects to save Memory space
            del psn_exc, psn_inh, psn_ext, parrot_neurons, iaf_alpha, spikerecorder
        rate += out #/3
    # rate /= 4
    print(rate - 8.0,'\tfiring rate')
    return rate - 8.0


#Bisect only accepts range where f(min)* f(max) < 0
optimal_alpha = bisect(alpha_optimizer, ALPHA+0.30637500, ALPHA + 0.90637500, xtol=0.01, maxiter=10)
print(f"Optimal alpha for the STDP Synapse: {optimal_alpha:.8f}")
print(timers)
print(sum(timers))
print(time.time() - st)
