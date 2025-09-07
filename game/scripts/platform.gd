extends AnimatableBody2D

var SPEED = 96
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

var direction = 1
#var time = 0
#1.5 s 96 pixel
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _physics_process(delta: float) -> void:
	position.x+= direction * SPEED * delta
	if position.x >= 704:
		direction = -1
	elif position.x <= 608:
		direction = 1
	#time += delta
	#print("Time: " + str(time))
