from ultralytics import YOLO, checks, hub
checks()

hub.login('ultralytics_api_key_here')

model = YOLO('https://hub.ultralytics.com/models/drsVaD3IRc5Cmx2HQQeW')
results = model.train(device='mps')