from icecube import dataio, icetray
from icecube.simclasses import I3CLSimMediumProperties
import numpy


# Open the file
filename = "/data/sim/IceCube/2020/filtered/level2/neutrino-generator/22859/0000000-0000999/Level2_NuTau_NuGenCCNC.022859.000000.i3.zst"  # or .i3.gz, etc.
infile = dataio.I3File(filename)

i = 0

M = icetray.I3Frame.Stream('M')  # same construct you used in I3Writer
Q = icetray.I3Frame.DAQ
Physics = icetray.I3Frame.Physics

SnowstormParameters = ["Absorption", "DOMEfficiency", "HoleIceForward_Unified_p0", "HoleIceForward_Unified_p1", "Scattering"]

# Prepare dictionary to store data
data = {param: [] for param in SnowstormParameters}
data["AbsorptionLength"] = []
data["ScatteringLength"] = []

iceDataDirectory = "/mnt/ceph1-npx/user/tvaneede/GlobalFit/SnowStorm_systematics/icetray/build/ice-models/resources/models/ICEMODEL/spice_ftp-v3"
icemodel_dat = numpy.loadtxt(iceDataDirectory+"/icemodel.dat", unpack=True)
CrystalDensityBenchmark = icemodel_dat[6][::-1]

# Loop only over M-frames
while infile.more():
    frame = infile.pop_frame()
    if frame.Stop == Q:  # <-- works, stops are stored as strings
        MediumProperties = frame["MediumProperties"]
        SnowstormParameterDict = frame["SnowstormParameterDict"]
        for var_name in SnowstormParameters:
            data[var_name].append( SnowstormParameterDict[var_name] )

        # Store per-layer absorption and scattering lengths for this event
        num_layers = len(MediumProperties.GetBirefringenceLayerScaling())
        absorption_lengths = [MediumProperties.GetAbsorptionLength(i).GetValue(400) for i in range(num_layers)]
        scattering_lengths = [MediumProperties.GetScatteringLength(i).GetValue(400) for i in range(num_layers)]
        data["AbsorptionLength"].append(absorption_lengths)
        data["ScatteringLength"].append(scattering_lengths)


        # break
    i+=1
    # if i == 2000: break

###
### plot the parameters
###
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

for ax, var_name in zip(axes, SnowstormParameters):
    values = data[var_name]
    ax.hist(values, bins=20, edgecolor="black")
    ax.set_title(var_name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

# Hide any unused subplots (if len(data) < 6)
for ax in axes[len(data):]:
    ax.set_visible(False)

plt.savefig("medium_properties_histograms.png", dpi=300)
# plt.show()

###
###
###

# Snowstorm parameters
scattering_param = numpy.array(data["Scattering"])
avg_scat_lengths = numpy.array([numpy.mean(event) for event in data["ScatteringLength"]])

# Plot
plt.figure(figsize=(6,4))
plt.scatter(scattering_param, avg_scat_lengths, alpha=0.7, color='green')
plt.xlabel("Scattering (Snowstorm Parameter)")
plt.ylabel("Average ScatteringLength [m]")
plt.title("Average ScatteringLength vs Scattering")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_scat_length_vs_scattering.png", dpi=300)
plt.close()