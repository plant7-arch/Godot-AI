extends Node

#var s = {"hi":[3, 3, 3]}
#var t = 1
#
#@onready var timer: Timer = $Timer
#@onready var player: CharacterBody2D = %Player
#
#const interpreter_relative_path = "PythonFiles//venv//Scripts//python.exe"
#const script_relative_path = "PythonFiles//main.py"
#
#var DIR = OS.get_executable_path().get_base_dir()
#var interpreter_path = DIR + interpreter_relative_path
#var script_path = DIR + script_relative_path
#
#func _ready() -> void:
	#if !OS.has_feature("standalone"):
		#interpreter_path = ProjectSettings.globalize_path("res://"+interpreter_relative_path)
		#script_path =  ProjectSettings.globalize_path("res://"+script_relative_path)
	#print(OS.get_executable_path())
	#OS.execute(interpreter_path, [script_path])
	##save_to_file(JSON.stringify(s))
	##print(load_from_file())
	##print(ProjectSettings.globalize_path("res://memory.json"))
	##timer.start()
	#
	#pass
#
#
## Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta: float) -> void:
	##save_to_file(
		##JSON.stringify(
			##[player.global_position, player.velocity]
		##)
	##)
	###timer.
	##t+=1
	#pass
#
#func save_to_file(content):
	#var file = FileAccess.open("res://data//memory.json", FileAccess.WRITE)
	#file.store_string(content)
#
#func load_from_file():
	#var file = FileAccess.open("res://data//memory.json", FileAccess.READ)
	#var content = file.get_as_text()
	#return content
#
#
##func _on_timer_timeout() -> void:
	##save_to_file(JSON.stringify(t))
	##t+=1
