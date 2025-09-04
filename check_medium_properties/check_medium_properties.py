from icecube import dataio, icetray
from icecube.simclasses import I3CLSimMediumProperties
import numpy
from icecube.icetray import I3Units



# Open the file
# filename = "/data/sim/IceCube/2023/filtered/level2/neutrino-generator/22612/0000000-0000999/Level2_NuE_NuGenCCNC.022612.000000.i3.zst"  # or .i3.gz, etc.
# filename = "/mnt/ceph1-npx/user/tvaneede/GlobalFit/SnowStorm_systematics/iceprod/ScatFirst/hits_NuGenCCNC.i3.zst"
filename = "/mnt/ceph1-npx/user/tvaneede/GlobalFit/SnowStorm_systematics/iceprod/hits_NuGenCCNC.i3.zst"
infile = dataio.I3File(filename)

i = 0

M = icetray.I3Frame.Stream('M')  # same construct you used in I3Writer

SnowstormParameters = ["Absorption", "CrystalDensityParameterScaling", "DOMEfficiency", "HoleIceForward_Unified_p0", "HoleIceForward_Unified_p1", "Scattering"]

# Prepare dictionary to store data
data = {param: [] for param in SnowstormParameters}
data["AbsorptionLength"] = []
data["ScatteringLength"] = []

iceDataDirectory = "/mnt/ceph1-npx/user/tvaneede/GlobalFit/SnowStorm_systematics/icetray/build/ice-models/resources/models/ICEMODEL/spice_ftp-v3"
icemodel_dat = numpy.loadtxt(iceDataDirectory+"/icemodel.dat", unpack=True)
CrystalDensityBenchmark = icemodel_dat[6][::-1]

wavelength = 400*I3Units.nanometer

# Loop only over M-frames
while infile.more():
    frame = infile.pop_frame()
    if frame.Stop == M:  # <-- works, stops are stored as strings
        MediumProperties = frame["MediumProperties"]
        SnowstormParameterDict = frame["SnowstormParameterDict"]
        for var_name in SnowstormParameters:
            data[var_name].append( SnowstormParameterDict[var_name] )

        # Store per-layer absorption and scattering lengths for this event
        num_layers = len(MediumProperties.GetBirefringenceLayerScaling())
        absorption_lengths = [MediumProperties.GetAbsorptionLength(i).GetValue(wavelength) for i in range(num_layers)]
        scattering_lengths = [MediumProperties.GetScatteringLength(i).GetValue(wavelength) for i in range(num_layers)]
        data["AbsorptionLength"].append(absorption_lengths)
        data["ScatteringLength"].append(scattering_lengths)

        # check if i scaled the CrystalDensityParameter correctly
        CrystalDensity = MediumProperties.GetBirefringenceLayerScaling()
        nonmatches = []
        for i,x in enumerate(CrystalDensity):
            ScalingCheck = x/CrystalDensityBenchmark[i]
            if ScalingCheck != SnowstormParameterDict["CrystalDensityParameterScaling"]:
                # print( x,CrystalDensityBenchmark[i],"ScalingCheck", ScalingCheck )
                nonmatches.append(ScalingCheck)

        print( "CrystalDensityParameterScaling", SnowstormParameterDict["CrystalDensityParameterScaling"], "non matches / 171", len(nonmatches) )
        if len(nonmatches) > 0: print(nonmatches)



        # break
    i+=1
    # if i == 20: break

###
### plot the parameters
###
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

for ax, (var_name, values) in zip(axes, data.items()):
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

# Compute average scattering length per event
avg_scat_lengths = numpy.array([numpy.mean(event) for event in data["ScatteringLength"]])

# Snowstorm parameters
scattering_param = numpy.array(data["Scattering"])
crystal_density_param = numpy.array(data["CrystalDensityParameterScaling"])

# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(ncols=2, figsize=(12,5), constrained_layout=True)

# Left plot: average ScatteringLength vs Scattering
axes[0].scatter(scattering_param, avg_scat_lengths, alpha=0.7)
axes[0].set_xlabel("Scattering (Snowstorm Parameter)")
axes[0].set_ylabel("Average ScatteringLength [m]")
axes[0].set_title("Average ScatteringLength vs Scattering")
axes[0].grid(True)

# Right plot: average ScatteringLength vs CrystalDensityParameterScaling
axes[1].scatter(crystal_density_param, avg_scat_lengths, alpha=0.7, color='orange')
axes[1].set_xlabel("CrystalDensityParameterScaling (Snowstorm Parameter)")
axes[1].set_ylabel("Average ScatteringLength [m]")
axes[1].set_title("Average ScatteringLength vs CrystalDensityParameterScaling")
axes[1].grid(True)

# Save the combined figure
plt.savefig("avg_scat_length_vs_params.png", dpi=300)
plt.close()

###
###
###

# Snowstorm parameters
scattering_param = numpy.array(data["Scattering"])
crystal_density_param = numpy.array(data["CrystalDensityParameterScaling"])

# Apply filter for CrystalDensityParameterScaling between 0.95 and 1.05
mask = (crystal_density_param >= 0.95) & (crystal_density_param <= 1.05)

filtered_scattering = scattering_param[mask]
filtered_avg_scat_lengths = avg_scat_lengths[mask]

# Plot
plt.figure(figsize=(6,4))
plt.scatter(filtered_scattering, filtered_avg_scat_lengths, alpha=0.7, color='green')
plt.xlabel("Scattering (Snowstorm Parameter)")
plt.ylabel("Average ScatteringLength [m]")
plt.title("Average ScatteringLength vs Scattering\n(CrystalDensityParameterScaling 0.95-1.05)")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_scat_length_vs_scattering_filtered.png", dpi=300)
plt.close()

###
###
###

# Compute average scattering length per event
avg_scat_lengths = numpy.array([numpy.mean(event) for event in data["ScatteringLength"]])

# Snowstorm parameters
scattering_param = numpy.array(data["Scattering"])
crystal_density_param = numpy.array(data["CrystalDensityParameterScaling"])

# Apply filter for CrystalDensityParameterScaling between 0.95 and 1.05
mask = (scattering_param >= 0.95) & (scattering_param <= 1.05)

filtered_crystal_density_param = crystal_density_param[mask]
filtered_avg_scat_lengths = avg_scat_lengths[mask]

# Plot
plt.figure(figsize=(6,4))
plt.scatter(filtered_crystal_density_param, filtered_avg_scat_lengths, alpha=0.7, color='green')
plt.xlabel("CrystalDensityParameterScaling (Snowstorm Parameter)")
plt.ylabel("Average ScatteringLength [m]")
plt.title("Average ScatteringLength vs CrystalDensityParameterScaling\n(scattering_param 0.95-1.05)")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_scat_length_vs_params_filtered.png", dpi=300)
plt.close()