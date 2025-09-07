extends CharacterBody2D

const SPEED = 180.0
const JUMP_VELOCITY = -300.0

@onready var animated_sprite: AnimatedSprite2D = $AnimatedSprite2D		

# Action:
# 0 == "idle"
# 1 == "move_left"
# 2 == "move_right"
# 3 == "jump"

var direction: int = 1

func _ready() -> void:
	Connection.done = false
	Connection.player = self

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta
	
	if Connection.action == 3 and is_on_floor():
		velocity.y = JUMP_VELOCITY
	
	if Connection.action != 3:
		direction = Connection.action
	
	if direction == 2:
		animated_sprite.flip_h = false
	elif direction == 1:
		animated_sprite.flip_h = true
	
	if is_on_floor():
		if direction == 0:
			animated_sprite.play('idle')
		else:
			animated_sprite.play('run')
	else:
		animated_sprite.play('jump')

	if direction == 0:
		velocity.x = move_toward(velocity.x, 0, SPEED)
	elif direction == 1:
		velocity.x = -SPEED
	elif direction == 2:
		velocity.x =  SPEED
		
		
	move_and_slide()


func _on_timer_timeout() -> void:
	Connection.done = true
	
