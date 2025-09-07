class_name Core
extends Node

var player: CharacterBody2D
var subviewport: SubViewport

var done: bool
var action: int
var reward: int = -1

var screenshot: Image
var timeout_sec: float = 60
var episode_count: int = 0
var shml: SharedMemoryLink

func _ready():
	shml = SharedMemoryLink.new()
	
	if not shml.connect():
		print("Failed to connect to shared memory. Another instance might be running.")
		get_tree().quit()
		return
		
	if not shml.is_first_instance():
		print("Another instance is primary. Quitting.")
		get_tree().quit()
		return
		
	print("Successfully connected to shared memory.")
	
	await get_tree().process_frame
	if not shml.send_meta(subviewport.get_texture().get_image()):
		print("Failed to send initial metadata to Python.")
		get_tree().quit()
	
	episode_count += 1
		


func _physics_process(_delta):
	await get_tree().process_frame
	await get_tree().process_frame
	
	if not shml or not shml.is_connected() or not shml.is_first_instance():
		return
		
	screenshot = subviewport.get_texture().get_image()

	var success = shml.send_step(
		player.position.x,
		player.position.y,
		player.velocity.x,
		player.velocity.y,
		screenshot,
		reward,
		done
	)
	
	if done:
		done = false
		print("Episode ", episode_count, " completed")
		get_tree().reload_current_scene()
		episode_count += 1
		
	reward = -1

	if not success:
		print("Failed to send step data to Python. Disconnecting.")
		shml.disconnect()
		get_tree().quit()
		return
	
	action = shml.read_action()

	#if action == -1:
		#print("Did not receive a valid action from Python.")
		#return
	

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if shml and shml.is_connected():
			shml.disconnect()
			print("Godot cleanly disconnected from shared memory.")
