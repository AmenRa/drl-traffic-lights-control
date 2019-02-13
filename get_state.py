def getState(self):
    positionMatrix = []
    velocityMatrix = []

    cellLength = 7
    offset = 11
    speedLimit = 14

    junctionPosition = traci.junction.getPosition('0')[0]
    vehicles_road1 = traci.edge.getLastStepVehicleIDs('1si')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('2si')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('3si')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('4si')
    for i in range(12):
        positionMatrix.append([])
        velocityMatrix.append([])
        for j in range(12):
            positionMatrix[i].append(0)
            velocityMatrix[i].append(0)

    for v in vehicles_road1:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if(ind < 12):
            positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
            velocityMatrix[2 - traci.vehicle.getLaneIndex(
                v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

    for v in vehicles_road2:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
        if(ind < 12):
            positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
            velocityMatrix[3 + traci.vehicle.getLaneIndex(
                v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

    junctionPosition = traci.junction.getPosition('0')[1]
    for v in vehicles_road3:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if(ind < 12):
            positionMatrix[6 + 2 -
                           traci.vehicle.getLaneIndex(v)][11 - ind] = 1
            velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

    for v in vehicles_road4:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
        if(ind < 12):
            positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
            velocityMatrix[9 + traci.vehicle.getLaneIndex(
                v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

    light = []
    if(traci.trafficlight.getPhase('0') == 4):
        light = [1, 0]
    else:
        light = [0, 1]

    position = np.array(positionMatrix)
    position = position.reshape(1, 12, 12, 1)

    velocity = np.array(velocityMatrix)
    velocity = velocity.reshape(1, 12, 12, 1)

    lgts = np.array(light)
    lgts = lgts.reshape(1, 2, 1)

    return [position, velocity, lgts]
