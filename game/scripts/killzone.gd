extends Area2D

@onready var timer: Timer = $Timer

func _on_body_entered(body: Node2D) -> void:
	print("Player died")
	body.get_node('CollisionShape2D').queue_free()
	Engine.time_scale = 0.1
	timer.start()


func _on_timer_timeout() -> void:
	Engine.time_scale = 1
	Connection.done = true
