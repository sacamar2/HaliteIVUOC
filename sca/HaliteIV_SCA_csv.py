#!/usr/bin/env python
# coding: utf-8

from kaggle_environments import evaluate, make
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

num_games=3
all_data_csv=pd.DataFrame()
for j in range(num_games):
	# Creación e inicialización el entorno
	env = make("halite", debug=True)
	env.render()
	
	# Simulación de la partida con los dos robots
	raw_data_game=env.run(["submission.py","submission.py"])
	
	
	# Extraemos y guardamos la información de la partida para analizarla
	
	# Todos los datos de la partida en formato JSON:
	game_data=pd.DataFrame(raw_data_game)
	
	# Construimos las variables donde se guardarán los datos
	num_players=len(game_data.columns)
	rewards,statuses,shipyards_players,ships_players,actions,halites=[],[],[],[],[],[]
	halite_on_cargo,halite_on_env,halite_on_starships,halite_total=[],[],[],[]
	halite_on_cargo_enemy,halite_on_starships_enemy,halite_total_enemy=[],[],[]
	
	# Separación de la información sobre el mapa en el transcurso de la partida
	history_board=pd.DataFrame(list(pd.DataFrame(list(game_data.iloc[:,0]))['observation']))
	halite_map_evo=history_board['halite']
	
	# Here it contains data about cargo and from ships and shipyards but not the position or the actions
	all_status_num_data_players=pd.DataFrame(list(history_board['players'])) 
	
	# En esta sección se extrae la información de cada jugador por separado
	# Get actual position from the enemy ships, we will use the actions following the paths
	init_frame,end_frame=0,len(game_data)
	for i in np.arange(0,num_players):
		game_data_player=pd.DataFrame(list((game_data.iloc[:,i])))
		reward,status=game_data_player['reward'][init_frame:end_frame],game_data_player['status'][init_frame:end_frame]
		action=game_data_player['action'][init_frame:end_frame]
		halite=pd.DataFrame(list(all_status_num_data_players[i]))[0][init_frame:end_frame]
		shipyards=0 # No terminado
		ships=0 # No terminado
		rewards.append(reward)
		statuses.append(status)
		shipyards_players.append(shipyards) # No terminado
		ships_players.append(ships) # No terminado
		actions.append(action)
		halites.append(halite)
	rewards,statuses=np.array(rewards),np.array(statuses)
	
	# Datos del Jugador
	init_frame,end_frame=0,400
	for i in range(init_frame,end_frame):
		ships_key=game_data[0][i]['observation']['players'][0][2].keys()
		all_my_ships=game_data[0][i]['observation']['players'][0][2]
		hoc=sum([all_my_ships[i][1] for i in ships_key])
		hos=halites[0][i]
		ht=hoc+hos
		halite_on_cargo.append(hoc)
		halite_on_starships.append(hos)
		halite_total.append(ht)
		hoe=sum(halite_map_evo[i])
		halite_on_env.append(hoe)
	
	# Datos del Enemigo
	for i in range(init_frame,end_frame):
		ships_key=game_data[0][i]['observation']['players'][1][2].keys()
		all_my_ships=game_data[0][i]['observation']['players'][1][2]
		hoc=sum([all_my_ships[i][1] for i in ships_key])
		hos=halites[1][i]
		ht=hoc+hos
		halite_on_cargo_enemy.append(hoc)
		halite_on_starships_enemy.append(hos)
		halite_total_enemy.append(ht)

	all_data_csv["halite_total_enemy_"+j.__str__()]=np.int_(halite_total_enemy)
	all_data_csv["halite_total_"+j.__str__()]=np.int_(halite_total)
	all_data_csv["halite_on_env_"+j.__str__()]=np.int_(halite_on_env)
	
all_data_csv.to_csv('all_data_csv.csv')




