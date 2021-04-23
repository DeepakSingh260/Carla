import os
import sys
# sys.path.append(".")
import glob
import math 
from enum import Enum 
from collections import deque
import random
import networkx as nx 
import numpy as np

from agents.navigation.controller import VehiclePIDController
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
def positive(num):
	"""
	Return the given number if positive, else 0

			:param num: value to check
	"""
	return num if num > 0.0 else 0.0

def get_speed(vehicle):

	vel = vehicle.get_velocity()

	return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def distance_vehicle(waypoint , vehicle_transfrom):

	loc = vehicle_transfrom.location

	x = waypoint.transform.location.x - loc.x

	y = waypoint.transform.location.y - loc.y 

	return math.sqrt(x*x + y*y)

def vector(location_1 , location_2):

	x = location_2.x - location_1.x 
	y = location_2.y - location_1.y 
	z = location_2.z - location_1.z 
	norm = np.linalg.norm([x , y , z ]) + np.finfo(float).eps 

	return [x/norm , y/norm , z/norm ]
 


def is_within_distance(target_location , current_location , orientation , max_distance , d_angle_th_up , d_angle_th_low = 0):

	target_vector = np.array([target_location.x - current_location.x , target_location.y- current_location.y])
	norm_target = np.linalg.norm(target_vector)

	if norm_target < 0.001:

		return True

	if norm_target > max_distance:
		return False

	forward_vector = np.array([math.cos(math.radians(orientation)) , math.sin(math.radians(orientation))])
	d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector , target_vector)/norm_target,-1.,1.)))

	return d_angle_th_low<d_angle <d_angle_th_up


def compute_distance(location_1 ,location_2):

	x = location_2.x - location_1.x
	y = location_2.y - location_1.y
	z = location_2.z - location_1.z
	norm = np.linalg.norm([x,y,z])+ np.finfo(float).eps

	return norm 


class Normal(object):

	max_speed = 50
	speed_lim_dist = 3
	speed_decrease = 10
	safety_time = 3
	min_proximity_threshold = 10
	braking_distance = 6
	overtake_counter = -1
	tailgate_counter = 0


class RoadOption(Enum):

	VOID = -1
	LEFT = 1
	RIGHT = 2
	STRAIGHT = 3
	LANEFOLLOW = 4
	CHANGELANELEFT = 5
	CHANGELANERIGHT = 6


class LocalPlanner(object):

	FPS = 20

	def __init__(self,agent):

		self._vehicle = agent.vehicle

		self._map = agent.vehicle.get_world().get_map()
		self._target_speed = None
		self.sampling_radius = None
		self._min_distance = 3
		self._current_distance = None
		self.target_road_option = None
		self._vehicle_controller = None
		self._global_plan = None
		self._pid_controller = None
		self.waypoints_queue = deque(maxlen = 20000)
		self._buffer_size = 5
		self._waypoint_buffer = deque(maxlen=self._buffer_size)

		self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
		self.args_lat_city_dict = {
            'K_P': 0.58,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1.0 / self.FPS}
		self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
		self.args_long_city_dict = {
            'K_P': 0.15,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

	def get_incoming_waypoint_and_direction(self,steps=3):

		if len(self.waypoints_queue)>steps:
			return self.waypoints_queue[steps]

		else:

			try:
				wpt , direction = self.waypoints_queue[-1]
				return wpt,direction
			except IndexError as i:

				return None ,RoadOption.VOID

		return None , RoadOption.VOID


	def set_speed(self,speed):

		self._target_speed = speed 


	def set_global_plan(self , current_plan , clean = True):

		print('set_global_plan called')

		for elem in current_plan : 

			self.waypoints_queue.append(elem)

		if clean :

			self._waypoint_buffer.clear()

			for _ in range(self._buffer_size):

				if self.waypoints_queue:

					self._waypoint_buffer.append(self.waypoints_queue.popleft())

				else :
					break


		self._global_plan = True 

	def run_step(self, target_speed = None , debug= False):

		print('run_step  _local_planner called')

		if target_speed is not None :

			self._target_speed = target_speed

		else :

			self.target_speed = self.vehicle.get_speed_limit()


		if (len(self.waypoints_queue) == 0):

			control = carla.VehicleControl()

			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			control.hand_brake = False
			control.manual_gear_shift = False 

			return control 

		if not self._waypoint_buffer:

			for i in range(self._buffer_size):

				if self.waypoints_queue:

					self._waypoint_buffer.append(self.waypoints_queue.popleft())


				else :

					break 

		self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

		self.target_waypoint , self.target_road_option = self._waypoint_buffer[0]

		if target_speed > 50:

			args_lat = self.args_lat_hw_dict
			args_long = self.args_long_hw_dict

		else :

			args_lat = self.args_lat_city_dict
			args_long = self.args_long_city_dict

		self._pid_controller = VehiclePIDController(self._vehicle , args_lateral = args_lat , args_longitudinal = args_long)

		control = self._pid_controller.run_step(self._target_speed , self.target_waypoint)

		vehicle_transfrom = self._vehicle.get_transform()

		max_index = -1

		for i , (waypoint , _) in enumerate(self._waypoint_buffer):

			if distance_vehicle(waypoint , vehicle_transfrom) < self._min_distance:

				max_index = i 

		if max_index >=0:

			for i in range(max_index+1):

				self._waypoint_buffer.popleft()


		return control 


class GlobalRoutePlannerDAO(object):

	def __init__(self , wmap , sampling_resolution):

		self._sampling_resolution = sampling_resolution
		self._wmap = wmap

	def get_topology(self):
		"""
		Accessor for topology.
		This function retrieves topology from the server as a list of
		road segments as pairs of waypoint objects, and processes the
		topology into a list of dictionary objects.

		:return topology: list of dictionary objects with the following attributes
		    entry   -   waypoint of entry point of road segment
		    entryxyz-   (x,y,z) of entry point of road segment
		    exit    -   waypoint of exit point of road segment
		    exitxyz -   (x,y,z) of exit point of road segment
		    path    -   list of waypoints separated by 1m from entry
		                to exit
		"""
		topology = []
		# Retrieving waypoints to construct a detailed topology
		for segment in self._wmap.get_topology():
			wp1, wp2 = segment[0], segment[1]
			l1, l2 = wp1.transform.location, wp2.transform.location
			# Rounding off to avoid floating point imprecision
			x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
			wp1.transform.location, wp2.transform.location = l1, l2
			seg_dict = dict()
			seg_dict['entry'], seg_dict['exit'] = wp1, wp2
			seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
			seg_dict['path'] = []
			endloc = wp2.transform.location
			if wp1.transform.location.distance(endloc) > self._sampling_resolution:
				w = wp1.next(self._sampling_resolution)[0]
				while w.transform.location.distance(endloc) > self._sampling_resolution:
					seg_dict['path'].append(w)
					w = w.next(self._sampling_resolution)[0]
			else:
				seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
			topology.append(seg_dict)
		return topology

	def get_resolution(self):

		return self._sampling_resolution

	def get_waypoint(self,location):

		waypoint = self._wmap.get_waypoint(location)
		return waypoint

class GlobalRoutePlanner(object):

	def __init__(self,dao):

		self._dao = dao
		self._topology = None
		self._graph = None
		self._id_map = None

		self._road_id_to_edge = None

		self._intersection_end_node = -1
		self._previous_decision = RoadOption.VOID


	def setup(self):

		print('setup  called ')
		self._topology = self._dao.get_topology()
		self._graph , self._id_map  , self._road_id_to_edge = self._build_graph()
		self._find_loose_ends()
		self._lane_change_link() 

	def _build_graph(self):

		print('_build_graph called')

		graph = nx.DiGraph()

		id_map = dict()

		road_id_to_edge = dict()

		for segment in self._topology:

			entry_xyz , exit_xyz = segment['entryxyz'] , segment['exitxyz']

			path = segment['path']
			entry_wp , exit_wp = segment['entry'] , segment['exit']

			intersection = entry_wp.is_junction

			road_id , section_id , lane_id = entry_wp.road_id , entry_wp.section_id , entry_wp.lane_id

			for vertex in entry_xyz , exit_xyz:

				if vertex not in id_map:

					new_id = len(id_map)
					id_map[vertex] = new_id
					graph.add_node(new_id , vertex = vertex)

			n1 = id_map[entry_xyz]
			n2 = id_map[exit_xyz]

			if road_id not in road_id_to_edge:

				road_id_to_edge[road_id] = dict()
			if section_id not in road_id_to_edge[road_id]:

				road_id_to_edge[road_id][section_id] = dict()

			road_id_to_edge[road_id][section_id][lane_id] = (n1 , n2)

			entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
			exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

			graph.add_edge(n1 , n2 , length = len(path)+1 , path = path , entry_waypoint = entry_wp , exit_waypoint = exit_wp , entry_vector = np.array([entry_carla_vector.x, entry_carla_vector.y , entry_carla_vector.z ]), exit_vector = np.array([exit_carla_vector.x , exit_carla_vector.y , exit_carla_vector.z ]), net_vector = vector(entry_wp.transform.location , exit_wp.transform.location),intersection= intersection , type = RoadOption.LANEFOLLOW)

		return graph , id_map , road_id_to_edge


	def _find_loose_ends(self):

		print('_find_loose_ends called')

		count_loose_ends = 0

		hop_resolution = self._dao.get_resolution()

		for segment in self._topology:

			end_wp = segment['exit']
			exit_xyz = segment['exitxyz']
			road_id , section_id , lane_id = end_wp.road_id , end_wp.section_id , end_wp.lane_id

			if road_id in self._road_id_to_edge and section_id in self._road_id_to_edge[road_id] and lane_id in self._road_id_to_edge[road_id][section_id]:
				pass

			else :
				count_loose_ends+=1
				if road_id not in self._road_id_to_edge:
					self._road_id_to_edge[road_id] = dict()

				if section_id not in self._road_id_to_edge[road_id]:

					self._road_id_to_edge[road_id][section_id] = dict()

				n1 = self._id_map[exit_xyz]
				n2 = -1*count_loose_ends
				self._road_id_to_edge[road_id][section_id][lane_id] = (n1 , n2)

				next_wp = end_wp.next(hop_resolution)
				path=[]
				while next_wp is not None and next_wp[0].road_id == road_id and next_wpt[0].section_id==section_id and end_wp[0].lane_id == lane_id:

					path.append(next_wp[0])
					next_wp = next_wp[0].next(hop_resolution)

				if path :
					n2_xyz = (path[-1].transform.location.x,path[-1].transform.location.y , path[-1].location.z)
					self._graph.add_node(n2 , vertex = n2_xyz)

					self._graph.add_edge(n1 , n2 , length= len(path)+1 , path = path , entry_waypoint = end_wp , exit_waypoint = path[-1], entry_vector=None , exit_vector = None , net_vector=None , intersection = end_wp.is_junction , type = RoadOption.LANEFOLLOW )


	def _localize(self , location):
	

		print('_localize called')
		waypoint = self._dao.get_waypoint(location)
		edge = None
		try:

			edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]

		except KeyError:

			print('failed to localize')


		return edge 				

	def _lane_change_link(self):

		print('_lane_change_link called ')
	
		for segment in self._topology:

			left_found , right_found = False , False 

			for waypoint in segment['path']:

				if not segment['entry'].is_junction:

					next_waypoint , next_road_option , next_segment = None , None , None 	


					if waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:

						next_waypoint = waypoint.get_right_lane()

						if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:

							next_road_option = RoadOption.CHANGELANERIGHT
							next_segment = self._localize(next_waypoint.transform.location)

							if next_segment is not None:

								self._graph.add_edge(self._id_map[segment['entryxyz']] , next_segment[0] , entry_waypoint = waypoint , exit_waypoint = next_waypoint ,intersection = False , exit_vector = None , path = [] , length=0 ,type = next_road_option , change_waypoint = next_waypoint )

								right_found = True

					if waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:

						next_waypoint = waypoint.get_left_lane()

						if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:

							next_road_option = RoadOption.CHANGELANELEFT
							next_segment  = self._localize(next_waypoint.transform.location)

							if next_segment is not None:

								self._graph.add_edge(self._id_map[segment['entryxyz']] , next_segment[0] , entry_waypoint = waypoint , exit_waypoint = next_waypoint , intersection = False , exit_vector = None, path =[] , length = 0 , type = next_road_option , change_waypoint = next_waypoint)

								left_found = True

					if left_found and right_found :

						break


	def _distance_heuristic(self , n1 ,n2 ):

		l1 = np.array(self._graph.nodes[n1]['vertex'])

		l2 = np.array(self._graph.nodes[n2]['vertex'])

		return np.linalg.norm(l1 - l2)

	def _path_search(self,  origin , destination):

		print('_path_search called')
	
		start , end = self._localize(origin) , self._localize(destination)	

		route = nx.astar_path(self._graph , source = start[0] , target = end[0] , heuristic = self._distance_heuristic , weight = 'length')
		route.append(end[1])
		return route 				

	def _successive_last_intersection_edge(self,index , route):

		print('_successive_last_intersection_edge called')

		last_intersection_edge = None

		last_node = None 

		for node1 , node2 in [(route[i] , route[i+1]) for i in range(index , len(route)-1)]:

			candidate_edge = self._graph.edges[node1 , node2]

			if node1 == route[index]:

				last_intersection_edge = candidate_edge

			if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:

				last_intersection_edge = candidate_edge
				last_node = node2

			else :
				break 

		return last_node , last_intersection_edge



	def _turn_decision(self , index , route , threshold = math.radians(35)):

		print('_turn_decision called')

		decision = None
		previous_node = route[index-1]
		current_node = route[index]
		next_node = route[index+1]
		next_edge = self._graph.edges[current_node , next_node]

		if index > 0:

			if self._previous_decision != RoadOption.VOID and self._intersection_end_node > 0 and self._intersection_end_node != previous_node and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']:

				decision = self._previous_decision

			else :
				self._intersection_end_node = -1
				current_edge = self._graph.edges[previous_node , current_node]

				calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge['intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']

				if calculate_turn :

					last_node , tail_edge   = self._successive_last_intersection_edge(index , route)

					self._intersection_end_node = last_node

					if tail_edge is not None :

						next_edge = tail_edge

					cv , nv = current_edge['exit_vector'] , next_edge['exit_vector']

					if cv is None or nv is None :

						return next_edge['type']

					cross_list = []

					for neighbor in self._graph.successors(current_node):

						select_edge = self._graph.edges[current_node , neighbor]

						if select_edge['type'] == RoadOption.LANEFOLLOW:

							if neighbor != route[index+1]:

								sv = select_edge['net_vector']
								cross_list.append(np.cross(cv ,sv)[2])

					next_cross = np.cross(cv , nv)[2]

					deviation = math.acos(np.clip(np.dot(cv , nv )/(np.linalg.norm(nv)) , -1.0 , 1.0))

					if not cross_list :

						cross_list.append(0)

					if deviation < threshold:

						decision = RoadOption.STRAIGHT

					elif cross_list and next_cross < min(cross_list):

						decision = RoadOption.LEFT

					elif cross_list and next_cross > max(cross_list):

						decision = RoadOption.RIGHT

					elif next_cross < 0:

						decision = RoadOption.LEFT

					elif next_cross > 0 :

						decision = RoadOption.RIGHT


				else :

					decision = next_edge['type']


				self._previous_decision = decision

				return decision 


	def _find_closest_in_list(self , current_waypoint , waypoint_list):

		print('_find_closest_in_list called')

		min_distance = float('inf')

		closest_index = -1
		for i , waypoint in enumerate(waypoint_list):
			distance = waypoint.transform.location.distance(current_waypoint.transform.location)
			if distance < min_distance :

				min_distance = distance 
				closest_index = i 


		return closest_index



	def trace_route(self , origin , destination):

		print('trace_route of GlobalRoutePlanner called')

		route_trace = []

		route = self._path_search(origin , destination)
		current_waypoint = self._dao.get_waypoint(origin)
		destination_waypoint = self._dao.get_waypoint(destination)
		resolution = self._dao.get_resolution()

		for i in range(len(route)-1):

			road_option = self._turn_decision(i , route)
			edge = self._graph.edges[route[i] , route[i+1]]
			path = []

			if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:

				route_trace.append((current_waypoint , road_option))

				exit_wp = edge['exit_waypoint']

				n1  , n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]

				next_edge = self._graph.edges[n1,n2]

				if next_edge['path']:

					closest_index = self._find_closest_in_list(current_waypoint , next_edge['path'])
					closest_index = min(len(next_edge['path'])-1 , closest_index+5)

					current_waypoint = next_edge['path'][closest_index]

				else :

					current_waypoint = next_edge['exit_waypoint']

				route_trace.append((current_waypoint , road_option))

			else :

				path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]

				closest_index = self._find_closest_in_list(current_waypoint , path)

				for waypoint in path[closest_index:]:

					current_waypoint = waypoint 
					route_trace.append((current_waypoint , road_option ))

					if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*resolution:

						break 

					elif len(route) - i <=2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:

						destination_index = self._find_closest_in_list(destination_waypoint , path)

						if closest_index > destination_index:

							break 

		return route_trace 



class Agent(object):

	def __init__(self , vehicle):

		self._vehicle = vehicle
		self._proximity_tlight_threshold = 5.0
		self._proximity_vehicle_threshold = 10.0
		self._local_planner = None
		self._world = self._vehicle.get_world()
		try :
			self._map = self._world.get_map()

		except RuntimeError as error:

			print('RuntimeError : {}' .format(error))

		self._last_traffic_light = None 


	def _bh_is_vehicle_hazard(self , ego_wpt , ego_loc , vehicle_list , proximity_th , up_angle_th , low_angle_th = 0 , lane_offset =0):

		print('_bh_is_vehicle_hazard called')

		if ego_wpt.lane_id < 0 and lane_offset !=0:

			lane_offset*=-1

		for target_vehicle in vehicle_list:

			target_vehicle_loc = target_vehicle.get_location()

			target_wpt = self._map.get_waypoint(target_vehicle_loc)

			if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id !=ego_wpt.lane_id +lane_offset:

				next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]

				if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:

					continue


			if is_within_distance(target_vehicle_loc , ego_loc , self._vehicle.get_transform().rotation.yaw,proximity_th , up_angle_th , low_angle_th):

				return (True , target_vehicle , compute_distance(target_vehicle_loc , ego_loc))

		return (False , None , -1)


	@staticmethod	
	def emergency_stop():

		print('emergency_stop called ')

		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0 
		control.brake = 1.0
		control.hand_brake = False 

		return control 

class BehavoirAgent(Agent):

	def __init__(self,vehicle , ignore_traffic_light = False , behavior = 'normal'):


		super(BehavoirAgent,self).__init__(vehicle)
		self.vehicle = vehicle
		self.ignore_traffic_light = ignore_traffic_light
		self._local_planner = LocalPlanner(self)
		self._grp = None
		self.look_ahead_steps = 0

		#vehicle information 

		self.speed = 0
		self.speed_limit = 0
		self.direction = None
		self.incoming_direction = None
		self.incoming_waypoint = None
		self.start_waypoint = None
		self.end_waypoint = None
		self.is_at_traffic_light = 0
		self.light_state = "Green"
		self.light_id_to_ignore = -1
		self.min_speed = 5
		self.behavior = None
		self._sampling_resolution = 4.5

		if behavior == 'normal':

			self.behavior = Normal()

	def update_information(self):

		print('update_information called')

		self.speed = get_speed(self.vehicle)
		self.speed_limit =  self.vehicle.get_speed_limit()

		self._local_planner.set_speed(self.speed_limit)
		self.direction = self._local_planner.target_road_option

		if self.direction is None:

			self.direction = RoadOption.LANEFOLLOW

		self.look_ahead_steps = int((self.speed_limit)/10)

		self.incoming_waypoint , self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps =self.look_ahead_steps)

		if self.incoming_direction is None:

			self.direction = RoadOption.LANEFOLLOW


		self.is_at_traffic_light = self.vehicle.is_at_traffic_light()

		if self.ignore_traffic_light:

			self.light_state = "Green"

		else:

			self.light_state = str(self.vehicle.get_traffic_light_state())

	def set_destination(self,start_location ,end_location , clean = False):

		print('set_destination called')

		if clean :

			self._local_planner.waypoints_queue.clear()

		self.start_waypoint = self._map.get_waypoint(start_location)

		self.end_waypoint = self._map.get_waypoint(end_location)

		route_trace = self._trace_route(self.start_waypoint , self.end_waypoint)

		self._local_planner.set_global_plan(route_trace , clean)


	def _trace_route(self,start_waypoint , end_waypoint):

		print('_trace_route BehavoirAgent called')

		if self._grp is None:
			 wld = self.vehicle.get_world()

			 dao = GlobalRoutePlannerDAO(wld.get_map() , sampling_resolution = self._sampling_resolution)

			 grp = GlobalRoutePlanner(dao)
			 grp.setup()
			 self._grp = grp

		route = self._grp.trace_route(start_waypoint.transform.location , end_waypoint.transform.location)

		return route 


	def  reroute(self, spawn_points):

		print('reroute called ')

		random.shuffle(spawn_points)
		new_start = self._local_planner.waypoints_queue[-1][0].transform.location 
		destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location 

		self.set_destination(new_start , destination)


	def traffic_light_manager(self, waypoint ):

		print('traffic_light_manager called')

		light_id =  self.vehicle.get_traffic_light().id if self.vehicle.get_traffic_light() is not None else -1 

		if self.light_state == "Red":

			if not waypoint.is_junction and (self.light_id_to_ignore!= light_id or light_id == -1 ):

				return 1

			elif waypoint.is_junction and light_id != -1:

				self.light_id_to_ignore = light_id

		if self.light_id_to_ignore != light_id:

			light_id_to_ignore = -1


		return 0 


	def _overtake(self, location , waypoint , vehicle_list):

		print('overtake called 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

		left_turn = None 
		right_turn = None 

		left_wpt = waypoint.get_left_lane()

		right_wpt = waypoint.get_right_lane()

		if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both ) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:

			new_vehicle_state , _ ,_  = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit / 3 ) , up_angle_th = 180 , lane_offset =-1)


			if not new_vehicle_state : 

				self.behavior.overtake_counter = 200
				self.set_destination(left_wpt.transform.location , self.end_waypoint , clean = True)

		elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving :


			new_vehicle_state , _ , _ = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit / 3) , up_angle_th = 180 , lane_offset = 1)

			if not new_vehicle_state : 

				self.behavior.overtake_counter = 200
				self.set_destination(right_wpt.transform.location , self.end_waypoint.transform.location , clean = True)


	def _tailgating(self , location , waypoint , vehicle_list):

		print('tailgating called ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt')

		left_turn = waypoint.left_lane_marking.lane_change

		right_turn = waypoint.right_lane_marking.lane_change

		left_wpt = waypoint.get_left_lane()
		right_wpt = waypoint.get_right_lane()

		behind_vehicle_state , behind_vehicle , _ = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , low_angle_th = 160)


		if behind_vehicle_state and self.speed < get_speed(behind_vehicle) :

			if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving : 

				new_vehicle , _ ,_ = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset = 1)

				if not new_vehicle : 

					print('tailgating moving towards right')
					self.behavior.tailgate_counter = 200
					self.set_destination(right_wpt.transform.location , end_waypoint.transform.location , clean = True)

			elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving : 

				new_vehicle , _ , _ = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset = -1)
 

				if not new_vehicle_state : 

					print('tailgating , moving to the left !')
					self.behavior.tailgate_counter = 200
					self.set_destination(left_wpt.transform.location , self.end_waypoint.transform.location , clean = True)

	def collision_and_car_avoid_manager(self , location , waypoint):

		print('collision_and_car_avoid_manager called')

		vehicle_list = self._world.get_actors().filter("*vehicle*")

		def dist(v) :

			 return v.get_location().distance(waypoint.transform.location)
		

		vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self.vehicle.id]

		if self.direction == RoadOption.CHANGELANELEFT :

			vehicle_state , vehicle , distance = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset =-1)

		elif  self.direction == RoadOption.CHANGELANERIGHT:

			vehicle_state , vehicle , distance = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/2) , up_angle_th = 180 , lane_offset = 1)

		else:

			vehicle_state , vehicle , distance = self._bh_is_vehicle_hazard(waypoint , location , vehicle_list , max(self.behavior.min_proximity_threshold , self.speed_limit/3 ) , up_angle_th = 30 )

			if vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10  and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):


				self._overtake(location , waypoint , vehicle_list )


			elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and self.speed > 10 and self.behavior.tailgate_counter == 0:

				self._tailgating(location , waypoint , vehicle_list )


		return vehicle_state , vehicle , distance


	def car_following_manager(self , vehicle , distance , debug = False) :

		print('car car_following_manager called')

		vehicle_speed = get_speed(vehicle)

		delta_v = max(1 , (self.speed - vehicle_speed)/3.6) 

		ttc = distance/delta_v if delta_v != 0 else distance / np.nextafter(0. , 1.)

		if self.behavior.safety_time > ttc > 0.0 :

			control = self._local_planner.run_step(target_speed = min(positive(vehicle_speed - self.behavior.speed_decrease) , min (self.behavior.max_speed , self.speed_limit - self.behavior.speed_lim_dist)) , debug = debug)


		elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time :

			control = self._local_planner.run_step(target_speed = min(max(self.min_speed , vehicle_speed) , min(self.behavior.max_speed , self.speed_limit - self.behavior.speed_lim_dist)) , debug = debug)



		else :

			control = self._local_planner.run_step(target_speed = min(self.behavior.max_speed , self.speed_limit - self.behavior.speed_lim_dist ) , debug = debug)



		return control 



	def run_step(self , debug = False ) :

		print('run_step function called')

		self.update_information()

		control = None 
		if self.behavior.tailgate_counter > 0:

			self.behavior.tailgate_counter -=1

		if self.behavior.overtake_counter > 0 :

			self.behavior.overtake_counter -= 1

		ego_vehicle_loc = self.vehicle.get_location()
		ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

		if self.traffic_light_manager(ego_vehicle_wp) !=0:

			return self.emergency_stop()

		vehicle_state , vehicle , distance = self.collision_and_car_avoid_manager(ego_vehicle_loc , ego_vehicle_wp)

		if vehicle_state :

			distance = distance - max(vehicle.bounding_box.extent.y , vehicle.bounding_box.extent.x) - max(self.vehicle.bounding_box.extent.y , self.vehicle.bounding_box.extent.x )


			if distance < self.behavior.braking_distance:

				return self.emergency_stop()

			else:

				control = self.car_following_manager(vehicle , distance)


		elif self.incoming_waypoint  and  self.incoming_waypoint.is_junction and (self.incoming_direction == RoadOption.LEFT or self.incoming_direction == RoadOption.RIGHT) :

			control = self._local_planner.run_step(target_speed = min(self.behavior.max_speed , self.speed_limit - 5) , debug = debug)


		else :

			control = self._local_planner.run_step(target_speed = min(self.behavior.max_speed , self.speed_limit - self.behavior.speed_lim_dist) , debug = debug)

		return control

actor_list = []
try :

	print('started')
	client = carla.Client('localhost',2000)
	client.set_timeout(10.0)
	world = client.get_world()
	map  =  world.get_map()
	print('townmap')
	blueprint_library = world.get_blueprint_library()

	vehicle = world.spawn_actor(  blueprint_library.filter('cybertruck')[0], carla.Transform(carla.Location(x = 130 , y = 195 , z = 40) , carla.Rotation(yaw = 180)) )
	actor_list.append(vehicle)
	agent = BehavoirAgent(vehicle)
	print('agent created')
	spawn_point = world.get_map().get_spawn_points()
	random.shuffle(spawn_point)
	if spawn_point[0].location != vehicle.get_location():

		destination = spawn_point[0].location

	else :
		destination = spawn_point[1].location
	agent.set_destination(vehicle.get_location() , destination  , clean = True)

	while True :

		if len(agent._local_planner.waypoints_queue) < 21:
			agent.reroute(spawn_point)

		elif len(agent._local_planner.waypoints_queue) == 0:
			print('target reached')
			break

		speed_limit = vehicle.get_speed_limit()
		agent._local_planner.set_speed(speed_limit)
			
		control = agent.run_step()
		vehicle.apply_control(control)

finally :

	client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
