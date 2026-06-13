import pybamm

# Create battery model
model = pybamm.lithium_ion.SPM()

# Battery parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Create simulation
sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values
)

# Simulate 1 hour
sim.solve([0, 3600])

# Show plots
sim.plot()