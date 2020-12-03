# load_pt_file

>>> weights = '/home/kirito/yolov3/runs/train/exp/weights/best.pt'
>>> 
>>> device = select_device('0')
>>> 
>>> model = attempt_load(weights, map_location=device)  # load FP32 model


Fusing layers... 
>>> 
>>> 
>>> from torchsummaryX import summary
>>> 
>>> arch = summary(model, torch.rand((1, 3, 416, 416)).cuda())
