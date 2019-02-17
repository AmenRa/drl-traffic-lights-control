import sys
import os
import optparse
import random
import numpy as np

def generate_routefile(max_time_steps = 3600, pH = 1. / 15, pV = 1. / 25, pAR = 1. / 20, pAL = 1. / 30):

    random.seed(42)  # make tests reproducible

    # Declare predefined routes
    with open("environment/model.rou.xml", "w") as routes:
        print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

        <!-- <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/> -->

        <vType accel="1.0" decel="4.5" id="SUMO_DEFAULT_TYPE" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

        <route id="west_to_north" edges="1_1_1-1_1_0 1_1_0-1_1_3"/>
        <route id="west_to_east" edges="1_1_1-1_1_0 1_1_0-1_1_5"/>
        <route id="west_to_south" edges="1_1_1-1_1_0 1_1_0-1_1_7"/>
        <route id="north_to_west" edges="1_1_3-1_1_0 1_1_0-1_1_1"/>
        <route id="north_to_east" edges="1_1_3-1_1_0 1_1_0-1_1_5"/>
        <route id="north_to_south" edges="1_1_3-1_1_0 1_1_0-1_1_7"/>
        <route id="east_to_west" edges="1_1_5-1_1_0 1_1_0-1_1_1"/>
        <route id="east_to_north" edges="1_1_5-1_1_0 1_1_0-1_1_3"/>
        <route id="east_to_south" edges="1_1_5-1_1_0 1_1_0-1_1_7"/>
        <route id="south_to_west" edges="1_1_7-1_1_0 1_1_0-1_1_1"/>
        <route id="south_to_north" edges="1_1_7-1_1_0 1_1_0-1_1_3"/>
        <route id="south_to_east" edges="1_1_7-1_1_0 1_1_0-1_1_5"/>

        ''', file=routes)
        lastVeh = 0
        vehNr = 0
        # For each step do...
        for i in range(max_time_steps):
            # Generate a west_to_east vehicle if...
            if random.uniform(0, 1) < pH:
                print('    <vehicle id="W2E_%i" type="SUMO_DEFAULT_TYPE" route="west_to_east" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a east_to_west vehicle if...
            if random.uniform(0, 1) < pH:
                print('    <vehicle id="E2W_%i" type="SUMO_DEFAULT_TYPE" route="east_to_west" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a north_to_south vehicle if...
            if random.uniform(0, 1) < pV:
                print('    <vehicle id="N2S_%i" type="SUMO_DEFAULT_TYPE" route="north_to_south" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a south_to_north vehicle if...
            if random.uniform(0, 1) < pV:
                print('    <vehicle id="S2N_%i" type="SUMO_DEFAULT_TYPE" route="south_to_north" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate west_to_south a vehicle if...
            if random.uniform(0, 1) < pAR:
                print('    <vehicle id="W2S_%i" type="SUMO_DEFAULT_TYPE" route="west_to_south" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate north_to_west a vehicle if...
            if random.uniform(0, 1) < pAR:
                print('    <vehicle id="N2W_%i" type="SUMO_DEFAULT_TYPE" route="north_to_west" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate east_to_north a vehicle if...
            if random.uniform(0, 1) < pAR:
                print('    <vehicle id="E2N_%i" type="SUMO_DEFAULT_TYPE" route="east_to_north" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate south_to_east a vehicle if...
            if random.uniform(0, 1) < pAR:
                print('    <vehicle id="S2E_%i" type="SUMO_DEFAULT_TYPE" route="south_to_east" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a west_to_north vehicle if...
            if random.uniform(0, 1) < pAL:
                print('    <vehicle id="W2N_%i" type="SUMO_DEFAULT_TYPE" route="west_to_north" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a north_to_east vehicle if...
            if random.uniform(0, 1) < pAL:
                print('    <vehicle id="N2E_%i" type="SUMO_DEFAULT_TYPE" route="north_to_east" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a east_to_south vehicle if...
            if random.uniform(0, 1) < pAL:
                print('    <vehicle id="E2S_%i" type="SUMO_DEFAULT_TYPE" route="east_to_south" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

            # Generate a south_to_west vehicle if...
            if random.uniform(0, 1) < pAL:
                print('    <vehicle id="S2W_%i" type="SUMO_DEFAULT_TYPE" route="south_to_west" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

        print("</routes>", file=routes)
