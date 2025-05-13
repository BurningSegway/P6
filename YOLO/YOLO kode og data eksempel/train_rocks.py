#from ultralytics import YOLO, checks, hub
#checks()

#hub.login('ultralytics_api_key_here')

#model = YOLO('https://hub.ultralytics.com/models/drsVaD3IRc5Cmx2HQQeW')
#results = model.train(device='mps')

from ultralytics import YOLO, checks, hub
checks()

hub.login('bb45ab1ce9f6990415bb1868e761b3bea39b8c0b01')

model = YOLO('https://hub.ultralytics.com/models/ogWRIrcFlewDC3HOCDf4')
results = model.train(device='mps')
