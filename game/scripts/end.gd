extends Node

signal frame_processed
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	Connection.end = self


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	frame_processed.emit()
