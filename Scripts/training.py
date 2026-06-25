import torch
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('yolov8x.pt')

# Check if CUDA is available and specify device manually
if torch.cuda.is_available():
    device_ids = [0, 1, 2]  # Specify the GPUs you want to use
    model = model.to(f'cuda:{device_ids[0]}')  # Move model to the first GPU

    # Train the model with a specified batch size
    model.train(data='data.yaml', 
                epochs=200, 
                imgsz=1100, 
                batch=30, 
                device=device_ids, 
                save_period=10, 
                optimizer='SGD',
                augment=True,
                save_json=True,
                save_conf=True, 
                nms=True, 
                save_txt=True, 
                conf=0.4
                )
else:
    print("CUDA not available. Using CPU.")