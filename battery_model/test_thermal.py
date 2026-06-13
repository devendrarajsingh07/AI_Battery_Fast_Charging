from thermal_model import ThermalModel

thermal = ThermalModel()

for step in range(20):

    temp = thermal.update(2.0)

    print(
        f"Step {step+1}: "
        f"Temperature = {temp:.2f} °C"
    )