import numpy as np 
import scipy
import matplotlib.pyplot as plt 

#from  8 up until 26 nodes
node_count = list(range(12,28,4))


d2_cqaoa_list=[0.397234287581685,
         0.2793861792123787,
         0.35967329810519005,
        0.43609545399838673
         ]

d3_cqaoa_list=[0.5176368034712318,
              0.15329508371236342,
               0.11523675940529636 ,
            0.15624255933754272]

d4_cqaoa_list=[0.3887135593672378,
             0.11124830488132825,
               0.13176713215425445,
               0.08446786897708472
         ]


# First Plot (Quantum Circuit Depth)
fig,ax1= plt.subplots(1,1, figsize = (12.5, 4))


ax1.set_xlabel('Number of qubits', fontsize=15, color="#444444", fontname="Segoe UI")
ax1.set_ylabel(r'$\mathrm{Var}_{\theta_k}[\partial_{\theta_k}\langle C(\theta)\rangle]$', fontsize=15, color="#444444", fontname="Segoe UI")
ax1.set_xticks(ticks = range(8, 27, 4))
ax1.plot(node_count[:len(d2_cqaoa_list)], d2_cqaoa_list, c='#20306f', marker="o", linestyle='solid', linewidth=3, label="Depth=2")
ax1.plot(node_count[:len(d3_cqaoa_list)], d3_cqaoa_list, c="#7d7d7d", marker="o", linestyle='solid', linewidth=3, label="Depth=3")
ax1.plot(node_count[:len(d4_cqaoa_list)], d4_cqaoa_list, c="#6929C4", marker="o", linestyle='solid', linewidth=3, label="Depth=4")

"""
# to plot the exponential fit
fit = np.polyfit(node_count, np.log(d4_cqaoa_list), deg=1)
x = np.linspace(node_count[0], node_count[-1], 200)


plt.semilogy(x,
             np.exp(fit[0] * x + fit[1]),
             '-',
             label=f'Depth=4 exp fit w/ {fit[0]:.2f}')
             
"""

ax1.legend(fontsize=14, labelcolor='linecolor')
ax1.grid()

plt.title("ConstrainedQAOA Variance of gradients")

# Show both plots side by side
plt.tight_layout()

#plt.savefig("constrainedQAOA_variance_plot_log_expfit.svg", format = "svg", dpi = 80, bbox_inches = "tight")

plt.show()
