"""
   Script for generating injected data with Grossman 1 component and isotropic model.
   Selected wavelenghts: 445 475 658 775 850 2190 [nm].
   With these set of parameters, we can exactly reproduce the injection made with bajes using GrossmanKBP-1-isotropic.
"""

from xkn import MKN, MKNConfig
from xkn.mkn import gen_inj_dict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Helvetica"],
        "xtick.major.size": 13.0,
        "ytick.major.size": 13.0,
        "xtick.minor.size": 7.0,
        "ytick.minor.size": 7.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


### Initialize MKNConfig object from the config file  
# select the path of the config.file
config_path = "example_injectionXkn_GrossmanModel/kn_config.ini"
mkn_config = MKNConfig(config_path)


### Initialize MKN object from the read parameters
mkn = MKN(*mkn_config.get_params(), log_level="WARNING")


### Function to create injection data files
def write_dict_to_files(data_dict):

    output_dir = 'example_injectionXkn_GrossmanModel/Injected_data' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for key, value in data_dict.items():
        if 'time' in value and 'mag' in value and 'sigma' in value and 'name' in value:
            filename = f"{output_dir}/{key}.txt"
            with open(filename, 'w') as file:
                file.write("# time\tmagnitude\tstd-mag\n") 
                for time, mag, sd in zip(value['time'], value['mag'], value['sigma']):
                    file.write(f"{time * 86400  + 1187008857}\t{mag}\t{sd}\n")


### Function to plot the injected lightcurves
def plot_inject_magnitudes(mags_a,                       
    ax=None,
    filename=None,
    title=None,
    titlesize=30,
    hsize=16,
    wsize=9,
    labelsize=30,
    ticksize=26,
    legendsize=12,
    legend_geom=[0, 0, 3, 4]):

    '''mags_a: dict of injected magnitudes with keys=bands.
       Each band is again a dictionary with keys: 'mag', 'time', 'sigma', 'name' '''

    colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(mags_a.keys())))[::-1]    
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(1, 1, figsize=(hsize, wsize), sharex=True, sharey=True)
    
    ymin = np.inf
    ymax = -np.inf
    for i, lam in enumerate(mags_a.keys()):
        if not i:
            xmin = mags_a[lam]["time"][0]
            xmax = mags_a[lam]["time"][-1]
        ymin = min(ymin, np.amin(mags_a[lam]["mag"]))  
        ymax = max(ymax, np.amax(mags_a[lam]["mag"]))
        ax.plot(mags_a[lam]["time"], mags_a[lam]["mag"], c=colors[i], label=f"{lam}")

    ymin *= 0.99
    ymax *= 1.0

    ax.grid(which="both", lw=1)
    ax.invert_yaxis()
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymax, ymin))
    ax.set_xscale("log")
    ax.set_xlabel("time [days]", fontsize=labelsize)
    ax.set_ylabel("magnitude", fontsize=labelsize)
    ax.tick_params(which="both", labelsize=ticksize)
    ax.legend(
        bbox_to_anchor=(legend_geom[0], legend_geom[1]),
        loc=legend_geom[2],
        ncol=legend_geom[3],
        fontsize=legendsize,
    )
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if ax_none:
        if filename is not None:
            fig.savefig(filename, facecolor="1", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":

    inputs = {
        "view_angle": 0.524,
        "distance": 40,
        "m_ej_dynamics": 0.03,
        "vel_dynamics": 0.13,
        "op_dynamics": 8,
        "high_lat_op_dynamics": 5,
        "low_lat_op_dynamics": 20,
        "m_ej_secular": 0.08,
        "vel_secular": 0.06,
        "op_secular": 5,
        "m_ej_wind": 0.02,
        "vel_wind": 0.1,
        "high_lat_op_wind": 1,
        "low_lat_op_wind": 5,
    }


    print("\nExample of injection: Computing injected data...")
    # create iinjection dictionary with all parameters
    inj_dict = gen_inj_dict(*mkn_config.get_params(), mkn_config.get_vars(inputs), sigma_min=0.0001, sigma_max=0.1) 
    # compute injected magnitudes
    mag_inj = mkn.gen_inj_data(inj_dict)
    # write and save injected magnitudes on data files
    write_dict_to_files(mag_inj)

    print("\nPlotting injected magnitudes...")
    # select a path for the plot in filename
    plot_inject_magnitudes(mag_inj, filename = "example_injectionXkn_GrossmanModel/plot.pdf")

