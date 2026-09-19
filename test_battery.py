from battery import Battery

battery = Battery()

print("Initial SOC:", battery.get_soc_percent(), "%")

# Charge 20 MWh
stored = battery.charge(20)

print("Energy stored:", stored, "MWh")
print("SOC after charging:", battery.get_soc_percent(), "%")

# Discharge 10 MWh
delivered = battery.discharge(10)

print("Energy delivered:", delivered, "MWh")
print("SOC after discharging:", battery.get_soc_percent(), "%")