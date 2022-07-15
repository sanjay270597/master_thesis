import matplotlib.pyplot as plt
import nest
import time
import pandas as pd
import numpy as np

# AX_DELAYS = [i / 10 for i in range(11)]
# AX_DELAYS += [i for i in range(2, 11)]
SIM_TIME = 5e+4
DELAY = 1.0
FPATH = "results/ax_del_"
AX_DELAYS = [0.5, 0.67, 0.83, 1.0]
ALPHA = 0.057


timers_data = []
timers = {}
st = time.time()
for checks in range(1, 6):

    for run in range(1, 5):
        timers_list = ['With'] if run < 3 else ['Without']
        timers_list.append('1ms Delay') if run % 2 == 1 else timers_list.append('10ms Delay')

        for axd_ in AX_DELAYS:

            # For Different runs
            tp = run if run < 3 else run - 2
            _delay = DELAY * 10 ** (tp - 1)
            axd = axd_ if run % 2 == 1 else axd_ * 10
            fpath = FPATH + str(axd) + '_delay_' + str(_delay)
            fpath = fpath if run < 3 else fpath + '_regular'
            fpath += '_run_' + str(checks)

            # Random seed
            nest.SetKernelStatus({
                'rng_seed': checks,
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
            nest.Connect(psn_exc, parrot_neurons, {"rule": 'all_to_all'},
                         {'synapse_model': "syn_exc"})
            if run < 3:
                nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                             {'synapse_model': "stdp_ax_delay"})
            else:
                nest.Connect(parrot_neurons, iaf_alpha, {"rule": 'all_to_all'},
                             {'synapse_model': "stdp_hom"})

            nest.Connect(psn_inh, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_inh"})
            nest.Connect(psn_ext, iaf_alpha, {"rule": 'one_to_one'}, {'synapse_model': "syn_exc"})

            # Registering Starting Time
            s = time.time()

            # Simulation
            nest.Simulate(SIM_TIME)

            # Checking end time
            timers[fpath] = time.time() - s
            timers_list.append(timers[fpath])

            # Getting Weights
            synapses = nest.GetConnections(target=iaf_alpha)
            weights = synapses.get('weight')
            # weight_check.append(weights)

            # Storing Data
            data = pd.DataFrame(weights, columns=[str(axd) + 'seconds'])
            data.index.name = "id"
            data.to_csv(fpath + '.csv')

            # Plotting Data
            # bars, ranges, containers = plt.hist(weights, 20, range=(0, 50))
            # plt.bar_label(containers)
            # plt.title('Axanol Delay: ' + str(axd) + ' second(s)')
            # plt.savefig(fpath + '.png')
            # plt.clf()

            # Resetting Kernel
            nest.ResetKernel()

            # Deleting objects to save Memory space
            del psn_exc, psn_inh, psn_ext, parrot_neurons, iaf_alpha, synapses, weights, data

        timers_data.append(timers_list)

    """weight_check = np.array(weight_check)
    checksum = sum((weight_check[::, :] - weight_check[4:, :]) > 1e-6)
    if (checksum != 0) :
        raise ValueError(checksum,' something\'s wrong')"""

time_df_cols = ['Weight Adjustments', 'Total Delay']
time_df_cols += [str(axd * 100) + '_pct_tot_delay' for axd in AX_DELAYS]
time_df = pd.DataFrame(timers_data, columns=time_df_cols)
time_df.to_csv('results/benchmark_timers.csv')

print(timers)
print(sum(timers.values()))
print(time.time() - st)
