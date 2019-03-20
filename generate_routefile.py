import numpy as np
import math


# generation of routes of cars
def generate_routefile(max_steps, seed):
    # make tests reproducible
    np.random.seed(seed)

    # initializations
    low_mode = False
    standard_mode = False
    ns_mode = False
    new_mode = False

    # low density
    if seed % 4 == 0:
        n_cars_generated = 600
        low_mode = True
        print("Mode: low")
        # high density
    elif seed % 4 == 1:
        n_cars_generated = 6000
        standard_mode = True
        print("Mode: high")
        # main source is north/south
    elif seed % 4 == 2:
        n_cars_generated = 3000
        ns_mode = True
        print("Mode: north-south main")
        # main source is east/west
    elif seed % 4 == 3:
        n_cars_generated = 3000
        new_mode = True
        print("Mode: east-west main")

    # the generation of cars is distributed according to a weibull distribution
    timings = np.random.weibull(2, n_cars_generated)
    timings = np.sort(timings)

    # reshape the distribution to fit the interval 0:max_steps
    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = max_steps
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

    # round every value to int -> effective steps when a car will be generated
    car_gen_steps = np.rint(car_gen_steps)

    # produce the file for cars generation, one car per line
    route_file_path = 'environment/tlcs_train.rou.xml'
    routes_file = open(route_file_path, 'w')
    routes_file.write('''<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>''')

    if standard_mode or low_mode:
        for car_counter, step in enumerate(car_gen_steps):
            straight_or_turn = np.random.uniform()
            # choose direction: straight or turn - 75% of times the car goes straight
            if straight_or_turn < 0.75:
                # choose a random source & destination
                route_straight = np.random.randint(1, 5)
                if route_straight == 1:
                    routes_file.write('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" />' % (car_counter, step))
                elif route_straight == 2:
                    routes_file.write('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" />' % (car_counter, step))
                elif route_straight == 3:
                    routes_file.write('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" />' % (car_counter, step))
                else:
                    routes_file.write('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" />' % (car_counter, step))
            # car that turn -25% of the time the car turns
            else:
                # choose random source source & destination
                route_turn = np.random.randint(1, 9)
                if route_turn == 1:
                    routes_file.write('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" />' % (car_counter, step))
                elif route_turn == 2:
                    routes_file.write('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" />' % (car_counter, step))
                elif route_turn == 3:
                    routes_file.write('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" />' % (car_counter, step))
                elif route_turn == 4:
                    routes_file.write('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))
                elif route_turn == 5:
                    routes_file.write('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" />' % (car_counter, step))
                elif route_turn == 6:
                    routes_file.write('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" />' % (car_counter, step))
                elif route_turn == 7:
                    routes_file.write('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" />' % (car_counter, step))
                elif route_turn == 8:
                    routes_file.write('    <vehicle id="S_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))

    if ns_mode:
        for car_counter, step in enumerate(car_gen_steps):
            # car goes straight or turns
            straight_or_turn = np.random.uniform()
            # choose the source
            source = np.random.uniform()
            # destination if the car goes straight
            destination_straight = np.random.uniform()
            # destination if the car turns
            destination_turn = np.random.randint(1, 5)
            # choose source: N S or E W
            if straight_or_turn < 0.75:
                # choose destination
                if source < 0.90:
                    if destination_straight < 0.5:
                        routes_file.write('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" />' % (car_counter, step))
                    else:
                        routes_file.write('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" />' % (car_counter, step))
                # source: E W
                else:
                    # choose destination
                    if destination_straight < 0.5:
                        routes_file.write('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" />' % (car_counter, step))
                    else:
                        routes_file.write('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" />' % (car_counter, step))
            # behavior: turn
            else:
                # choose source: N S or E W
                if source < 0.90:
                    # choose destination
                    if destination_turn == 1:
                        routes_file.write('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 2:
                        routes_file.write('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 3:
                        routes_file.write('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 4:
                        routes_file.write('    <vehicle id="S_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))
                # source: E W
                else:
                    # choose destination
                    if destination_turn == 1:
                        routes_file.write('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 2:
                        routes_file.write('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 3:
                        routes_file.write('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 4:
                        routes_file.write('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" />' % (car_counter, step))

    if new_mode:
        for car_counter, step in enumerate(car_gen_steps):
            straight_or_turn = np.random.uniform()
            source = np.random.uniform()
            destination_straight = np.random.uniform()
            destination_turn = np.random.randint(1, 5)
            # choose behavior: straight or turn
            if straight_or_turn < 0.75:
                # choose source: N S or E W
                if source < 0.90:
                    # choose destination
                    if destination_straight < 0.5:
                        routes_file.write('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" />' % (car_counter, step))
                    else:
                        routes_file.write('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" />' % (car_counter, step))
                # source: N S
                else:
                    # choose destination
                    if destination_straight < 0.5:
                        routes_file.write('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" />' % (car_counter, step))
                    else:
                        routes_file.write('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" />' % (car_counter, step))
            # behavior: turn
            else:
                # choose source: N S or E W
                if source < 0.90:
                    # choose destination
                    if destination_turn == 1:
                        routes_file.write('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 2:
                        routes_file.write('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 3:
                        routes_file.write('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 4:
                        routes_file.write('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" />' % (car_counter, step))
                # source: N S
                else:
                    # choose destination
                    if destination_turn == 1:
                        routes_file.write('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 2:
                        routes_file.write('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 3:
                        routes_file.write('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" />' % (car_counter, step))
                    elif destination_turn == 4:
                        routes_file.write('    <vehicle id="S_E_%i" type="standard_car" route="N_E" depart="%s" />' % (car_counter, step))

    routes_file.write("</routes>")
    routes_file.close()
