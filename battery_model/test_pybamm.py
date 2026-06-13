import pybamm

print("PyBaMM Version:", pybamm.__version__)

model = pybamm.lithium_ion.SPM()

print("Battery Model Loaded Successfully")