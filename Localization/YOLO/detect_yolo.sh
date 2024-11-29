# Navigate to the YOLOv5 directory
cd yolov5

# Run the YOLO training command
python detect.py --weights /yolo-model/best.pt --source /dataset/images/val --img 416 --conf 0.5 --save-txt --save-conf


