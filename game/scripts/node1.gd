extends Node


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	print("Node1")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:pass
	
	#print("Viewport: " + str(get_parent().get_parent().get_parent().get_parent().get_viewport().size))
	#print(get_viewport())
	#if Input.is_action_just_pressed("jump"):
		#take_background_screenshot()

# This function can be called at any time.
func take_background_screenshot():
	# It's good practice to wait for the next frame to ensure it's rendered.
	await get_tree().process_frame

	# Get the image data from the viewport's texture.
	var image = get_viewport().get_texture().get_image()

	# Save the image to a file.
	var error = image.save_png("res://background_screenshot2.png")

	if error == OK:
		print("Screenshot saved successfully from SubViewport!")
	else:
		print("Error saving screenshot.")
