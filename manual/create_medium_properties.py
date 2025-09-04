from icecube import clsim
from os.path import expandvars
import numpy
import matplotlib.pyplot as plt
import copy
from icecube.simclasses import I3CLSimFunctionScatLenIceCube
from icecube.icetray import I3Units


# taken from MakeIceCubeMediumProperties.py
step = 0.01
x_step = numpy.arange(step/2, 1, step)
y_step = numpy.sqrt(1-x_step*x_step)

# Load medium properties
medium = clsim.MakeIceCubeMediumProperties(
    iceDataDirectory=expandvars("$I3_BUILD/ice-models/resources/models/ICEMODEL/spice_ftp-v3")
)

n_layers = medium.GetLayersNum()
z_start = medium.GetLayersZStart()
dz = medium.GetLayersHeight()

wavelength = 400*I3Units.nanometer

def calculate_depth_scattering(ScatteringScaling = 1,
                               CrystalDensityScaling = 1):

    depths = []
    scattering_lengths = []

    if CrystalDensityScaling != 1:
        new_medium = copy.deepcopy(medium)
        # obtain original parameters and layer scales
        bfrParas = medium.GetBirefringenceParameters()
        meanCosineTheta = medium.GetMeanCosineTheta()
        bfrLayerScale = medium.GetBirefringenceLayerScaling()

        # scale the layers
        bfrLayerScale_new = numpy.array([val * CrystalDensityScaling for val in bfrLayerScale])
        new_medium.SetBirefringenceLayerScaling( bfrLayerScale_new )

        # obtain new correction, see MakeIceCubeMediumProperties.py
        srf = bfrParas[-1]
        sx = (bfrParas[0]*numpy.exp(-bfrParas[1]*numpy.arctan(bfrParas[3]*y_step)**bfrParas[2])).clip(min=0)
        sy = (bfrParas[4]*numpy.exp(-bfrParas[5]*numpy.arctan(bfrParas[7]*y_step)**bfrParas[6])).clip(min=0)
        mx = (bfrParas[8]*numpy.arctan(bfrParas[11]*y_step*x_step)*numpy.exp(-bfrParas[9]*y_step+bfrParas[10]*x_step)).clip(min=0)
        bfrCorrection = srf * numpy.sum(sx*sx + sy*sy + mx*mx) * step/2 * bfrLayerScale_new/(1-meanCosineTheta)

    for i in range(n_layers):
        # center depth of layer i
        depth = z_start + (i + 0.5) * dz
        scat_len = medium.GetScatteringLength(i).GetValue(wavelength)

        depths.append(depth)

        if ScatteringScaling == 1 and CrystalDensityScaling == 1:
            scattering_lengths.append(scat_len)
        if ScatteringScaling != 1 and CrystalDensityScaling == 1:
            scattering_lengths.append( scat_len*(1.0/ScatteringScaling) )
        if CrystalDensityScaling != 1:

            # obtain scattering length object and parameters
            ScatteringLength = medium.GetScatteringLength(i)

            alpha = ScatteringLength.GetAlpha()
            b400 = ScatteringLength.GetB400()

            ScatteringLength_new = I3CLSimFunctionScatLenIceCube(alpha=alpha, b400=b400,bfrCorrection=bfrCorrection[i])

            new_medium.SetScatteringLength(i, ScatteringLength_new )

            scat_len = new_medium.GetScatteringLength(i).GetValue(wavelength)
            scattering_lengths.append(scat_len*(1.0/ScatteringScaling))

            # print("alpha",alpha,"b400",b400,"bfrCorrection[i]",bfrCorrection[i],"scat_len",scat_len,"scat_len_scaled",scat_len*(1.0/ScatteringScaling))

    depths = numpy.array(depths)
    scattering_lengths = numpy.array(scattering_lengths)

    return scattering_lengths, depths

###
### test
###
# scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

# scat_alt, depths = calculate_depth_scattering(ScatteringScaling=1.1, CrystalDensityScaling=1)
# print( "ratio", sum(scat_alt/scat_ref)/len(depths) )

# scat_alt, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1.1)
# print( "ratio", sum(scat_alt/scat_ref)/len(depths) )

# scat_alt, depths = calculate_depth_scattering(ScatteringScaling=1.1, CrystalDensityScaling=1.1)
# print( "ratio", sum(scat_alt/scat_ref)/len(depths) )

# Plot Scattering Length vs Depth
plt.figure(figsize=(4,8))
plt.plot(*calculate_depth_scattering(), marker='o', linestyle='-')
plt.gca().invert_yaxis()  # depth increases downward
plt.xlabel("Scattering Length [m]")
plt.ylabel("Depth [m]")
plt.title("Scattering Length vs Depth")
plt.grid(True)
plt.tight_layout()
plt.savefig("scattering_length_vs_depth.png", dpi=300)
plt.close()

# --- Plot 1: ScatteringScaling variations ---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

# Left: absolute scattering length
for scale in [0.9, 1, 1.1]:
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=scale, CrystalDensityScaling=1)
    axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"ScatteringScaling={scale}")


axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)

# Right: ratio to baseline (1.0)
for scale in [0.9, 1, 1.1]:
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=scale, CrystalDensityScaling=1)
    ratio = scat_len / scat_ref
    axes[1].plot(ratio, depths, marker='o', linestyle='-')

for ax in axes:
    ax.invert_yaxis()
axes[1].set_xlabel("Ratio to baseline")
axes[1].set_title("Ratio to ScatteringScaling=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("scattering_scaling_with_ratios.png", dpi=300)
plt.close()


# --- Plot 2: CrystalDensityScaling variations ---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

# Left: absolute scattering length
for scale in [0.9, 1, 1.1]:
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=scale)
    axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"CrystalDensityScaling={scale}")

axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)

# Right: ratio to baseline (1)
for scale in [0.9, 1, 1.1]:
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=scale)
    ratio = scat_len / scat_ref
    axes[1].plot(ratio, depths, marker='o', linestyle='-')

for ax in axes:
    ax.invert_yaxis()
axes[1].set_xlabel("Ratio to baseline")
axes[1].set_title("Ratio to CrystalDensityScaling=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("crystal_scaling_with_ratios.png", dpi=300)
plt.close()


# --- Plot 3: Both same direction variations ---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

# Left: absolute scattering length
for ScatteringScaling,CrystalDensityScaling in zip([0.9, 1, 1.1],[0.9,1,1.1]):
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
    axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling},Crys={CrystalDensityScaling}")

axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)

# Right: ratio to baseline (1)
for ScatteringScaling,CrystalDensityScaling in zip([0.9, 1, 1.1],[0.9,1,1.1]):
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
    ratio = scat_len / scat_ref
    axes[1].plot(ratio, depths, marker='o', linestyle='-')

for ax in axes:
    ax.invert_yaxis()
axes[1].set_xlabel("Ratio to baseline")
axes[1].set_title("Ratio to Both=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("scat_crystal_scaling_samedir_with_ratios.png", dpi=300)
plt.close()

# --- Plot 4: opposite direction variations ---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

# Left: absolute scattering length
for ScatteringScaling,CrystalDensityScaling in zip([0.9, 1, 1.1],[1.1,1,0.9]):
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
    axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling},Crys={CrystalDensityScaling}")

axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)

# Right: ratio to baseline (1)
for ScatteringScaling,CrystalDensityScaling in zip([0.9, 1, 1.1],[1.1,1,0.9]):
    scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
    ratio = scat_len / scat_ref
    axes[1].plot(ratio, depths, marker='o', linestyle='-')

for ax in axes:
    ax.invert_yaxis()
axes[1].set_xlabel("Ratio to baseline")
axes[1].set_title("Ratio to Both=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("scat_crystal_scaling_oppdir_with_ratios.png", dpi=300)
plt.close()


# --- Plot 5: When both same direction variations, we suddenly have a flat ratio again, but not like this ---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

axes[0].plot(scat_ref, depths, marker='o', linestyle='-', label=f"Scat=1,Crys=1")
ratio = scat_ref / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

ScatteringScaling = 1
CrystalDensityScaling = 0.9

scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling},Crys={CrystalDensityScaling}")
ratio = scat_len / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

scat_len = scat_len*(1.0/0.9)
axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling}/0.9,Crys={CrystalDensityScaling}")
ratio = scat_len / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

scat_len = scat_len*(1.1/0.9)
axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling}/1.1,Crys={CrystalDensityScaling}")
ratio = scat_len / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)



# --- Plot 6: wtf?? 
# problem solved, i applied the ScatteringScaling twice---
fig, axes = plt.subplots(ncols=2, figsize=(10,8), sharey=True)

# Baseline
scat_ref, depths = calculate_depth_scattering(ScatteringScaling=1, CrystalDensityScaling=1)

axes[0].plot(scat_ref, depths, marker='o', linestyle='-', label=f"Scat=1,Crys=1")
ratio = scat_ref / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

# option 1
ScatteringScaling = 0.9
CrystalDensityScaling = 0.9
scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling},Crys={CrystalDensityScaling}")
ratio = scat_len / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

# option 2, should be same as 1
ScatteringScaling = 1.0
scat_len, _ = calculate_depth_scattering(ScatteringScaling=ScatteringScaling, CrystalDensityScaling=CrystalDensityScaling)
scat_len = scat_len/0.9
axes[0].plot(scat_len, depths, marker='o', linestyle='-', label=f"Scat={ScatteringScaling}/0.9,Crys={CrystalDensityScaling}")
ratio = scat_len / scat_ref
axes[1].plot(ratio, depths, marker='o', linestyle='-')

axes[0].invert_yaxis()
axes[0].set_xlabel("Scattering Length [m]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Scattering Length vs Depth")
axes[0].legend()
axes[0].grid(True)

for ax in axes:
    ax.invert_yaxis()
axes[1].set_xlabel("Ratio to baseline")
axes[1].set_title("Ratio to Both=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("scat_crystal_scaling_wtf_with_ratios.png", dpi=300)
plt.close()
