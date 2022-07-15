import matplotlib.pyplot as plt
import nest
import time
import pandas as pd

# AX_DELAYS = [i / 10 for i in range(11)]
# AX_DELAYS += [i for i in range(2, 11)]
SIM_TIME = 1e+5
DELAY = 1.0
FPATH = "results/ax_del_a_"
timers = []
st = time.time()
AX_DELAYS = [0.0, 0.5, 0.75, 1.0]
ALPHA = 0.619

for run in range(1, 5):

    for axd_ in AX_DELAYS:

        s = time.time()

        # For Different runs
        tp = run % 2 # 1,0,1,0
        _delay = DELAY * 10 ** (1 - tp)
        axd = axd_ * 10 ** (1 - tp)
        fpath = FPATH + str(axd) + '_delay_' + str(_delay)
        fpath = fpath if run < 3 else fpath + '_regular'

        # Random seed
        nest.SetKernelStatus({
            'rng_seed': 2705,
            'total_num_virtual_procs': 6})

        # Compiling Own Models
        nest.CopyModel('static_synapse', 'syn_exc', {"weight": 38.5})
        nest.CopyModel('static_synapse', 'syn_inh', {"weight": -192.5})
        nest.CopyModel('stdp_pl_synapse_hom_ax_delay', 'stdp_ax_delay', {"axonal_delay": axd,
                                                                         "delay": _delay,
                                                                         "alpha": ALPHA,
                                                                         "lambda": 0.1,
                                                                         "mu": 0.4,
                                                                         "tau_plus": 15.0,
                                                                         "weight": 38.5
                                                                         })
        nest.CopyModel('stdp_pl_synapse_hom', 'stdp_hom', {"axonal_delay": axd,
                                                           "delay": _delay,
                                                           "alpha": ALPHA,
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

        # Connecting Nodes
        nest.Connect(psn_exc, parrot_neurons, {"rule": 'all_to_all'}, {'synapse_model': "syn_exc"})
        if run < 3:
            nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                         {'synapse_model': "stdp_ax_delay"})
        else:
            nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                         {'synapse_model': "stdp_hom"})

        nest.Connect(psn_inh, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_inh"})
        nest.Connect(psn_ext, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_exc"})

        # Simulation
        nest.Simulate(SIM_TIME)

        # Checking timers
        timers.append(time.time() - s)

        # Getting Weights
        synapses = nest.GetConnections(target=iaf_alpha)
        weights = synapses.get('weight')

        # Storing Data
        data = pd.DataFrame(weights, columns=[str(axd) + 'seconds'])
        data.index.name = "id"
        data.to_csv(fpath + '.csv')

        # Plotting Data
        #bars, ranges, containers = plt.hist(weights, 20, range=(60, 80))
        #plt.bar_label(containers)
        #plt.title('Axanol Delay: ' + str(axd) + ' second(s)')
        #plt.savefig(fpath + '.png')
        #plt.clf()

        # Resetting Kernel
        nest.ResetKernel()

        # Deleting objects to save Memory space
        del psn_exc, psn_inh, psn_ext, parrot_neurons, iaf_alpha, synapses, weights, data
        # del bars, ranges, containers

print(timers)
print(sum(timers))
print(time.time() - st)
