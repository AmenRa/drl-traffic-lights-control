def get_lanes_state(self, l1_id, l2_id, l3_id, l4_id):
    # Vehicle count per lane
    l1_vehicle_count = traci.lane.getLastStepVehicleNumber(self, l1_id)
    l2_vehicle_count = traci.lane.getLastStepVehicleNumber(self, l2_id)
    l3_vehicle_count = traci.lane.getLastStepVehicleNumber(self, l3_id)
    l4_vehicle_count = traci.lane.getLastStepVehicleNumber(self, l4_id)

    # Halting vehicle count per lane
    l1_halting_vehicle_count = traci.lane.getLastStepHaltingNumber(self, l1_id)
    l2_halting_vehicle_count = traci.lane.getLastStepHaltingNumber(self, l2_id)
    l3_halting_vehicle_count = traci.lane.getLastStepHaltingNumber(self, l3_id)
    l4_halting_vehicle_count = traci.lane.getLastStepHaltingNumber(self, l4_id)

    # Average speed per lane
    l1_average_vehicle_speed = traci.lane.getLastStepMeanSpeed(self, l1_id)
    l2_average_vehicle_speed = traci.lane.getLastStepMeanSpeed(self, l2_id)
    l3_average_vehicle_speed = traci.lane.getLastStepMeanSpeed(self, l3_id)
    l4_average_vehicle_speed = traci.lane.getLastStepMeanSpeed(self, l4_id)

    # Cumulated waiting time per
    l1_cumulated_waiting_time = traci.lane.getWaitingTime(self, l1_id)
    l2_cumulated_waiting_time = traci.lane.getWaitingTime(self, l2_id)
    l3_cumulated_waiting_time = traci.lane.getWaitingTime(self, l3_id)
    l4_cumulated_waiting_time = traci.lane.getWaitingTime(self, l4_id)

    # tls state or previous action
