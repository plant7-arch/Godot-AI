extends Node

#@onready var sub_viewport: SubViewport = $".."
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.



# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	print(get_viewport().get_camera_2d())
	if Input.is_action_just_pressed("jump"):
		take_background_screenshot()


# This function can be called at any time.
func take_background_screenshot():
	# Get the SubViewport node.
	

	## It's good practice to wait for the next frame to ensure it's rendered.
	await RenderingServer.frame_post_draw

	# Get the image data from the viewport's texture.
	var image = get_viewport().get_texture().get_image()

	# Save the image to a file.
	var error = image.save_png("res://background_screenshot.png")

	if error == OK:
		print("Screenshot saved successfully from SubViewport!")
	else:
		print("Error saving screenshot.")
