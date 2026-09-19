class Battery:
    def __init__(
        self,
        capacity_mwh=100,
        power_mw=20,
        efficiency=0.90,
        soc_min=0.20,
        soc_max=0.90,
        initial_soc=0.50,
        degradation_cost=2.0
    ):
        self.capacity_mwh = capacity_mwh
        self.power_mw = power_mw
        self.efficiency = efficiency

        self.soc_min = soc_min * capacity_mwh
        self.soc_max = soc_max * capacity_mwh

        self.soc_mwh = initial_soc * capacity_mwh

        self.degradation_cost = degradation_cost

    def charge(self, energy_mwh):
        """
        Charge the battery.
        energy_mwh = electricity purchased from the grid.
        """

        energy_mwh = min(energy_mwh, self.power_mw)

        available_space = self.soc_max - self.soc_mwh

        # Account for charging efficiency
        actual_stored = energy_mwh * self.efficiency

        actual_stored = min(actual_stored, available_space)

        self.soc_mwh += actual_stored

        return actual_stored

    def discharge(self, energy_mwh):
        """
        Discharge the battery.
        energy_mwh = energy delivered to the grid.
        """

        energy_mwh = min(energy_mwh, self.power_mw)

        available_energy = self.soc_mwh - self.soc_min

        # Account for discharge efficiency
        required_from_battery = energy_mwh / self.efficiency

        required_from_battery = min(
            required_from_battery,
            available_energy
        )

        delivered = required_from_battery * self.efficiency

        self.soc_mwh -= required_from_battery

        return delivered

    def get_soc_percent(self):
        return (self.soc_mwh / self.capacity_mwh) * 100