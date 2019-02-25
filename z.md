```bash
#!/anaconda3/bin/python
#!/usr/bin/python3
```








Environment Variables:
1. Green Phase Duration: 31 sec (same as SUMO)
2. Yellow Phase Duration: 6 sec (same as SUMO)


Reward:
1. 1 / cumulated waiting time
2. 1 otherwise (do not count first 10 steps to avoid falling reward)

Experiments:

1. State (incoming lanes):
  1. number of vehicles per lane | traci.lane.getLastStepVehicleNumber(self, laneID)
  3. number of halting vehicles | getLastStepHaltingNumber(self, laneID)
  2. avarage speed per lane|  traci.lane.getLastStepMeanSpeed(self, laneID)
  3. cumulated waiting time per | lane traci.lane.getWaitingTime(self, laneID)
  4. tls state or previous action

2. State (blocks):
  1. number of cars per block going to the tls
  2. avarage speed per block
  3. cumulated waiting time per block
  4. tls state or previous action

3. Network:
  1. ReLU vs Swish
  2. RNN vs LTSM vs GRU
  3. Optimizer: Adam
  4. Loss: mean_squared_error
