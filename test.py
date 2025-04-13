import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
import shutil

# Import your modules
from dataset import CocoDataset, Resizer, Normalizer
from config import COCO_CLASSES, colors


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation in TensorFlow")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="data/COCO")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models/tensorflow_efficientdet_coco")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()
    return args


def test(opt):
    # Load the trained model
    model = tf.keras.models.load_model(opt.pretrained_model)
    
    # Initialize dataset with transforms
    dataset = CocoDataset(
        root_dir=opt.data_path, 
        set='val2017', 
        transform=lambda sample: Normalizer()(Resizer()(sample))
    )
    
    # Create output directory
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    
    # Process each image in the dataset
    for index in range(len(dataset)):
        data = dataset[index]
        scale = data['scale']
        
        # Prepare image for model input
        # TensorFlow expects (batch_size, height, width, channels)
        image = data['img']
        if isinstance(image, np.ndarray):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # Get predictions from the model
        scores, labels, boxes = model(image, training=False)
        
        # Convert to numpy arrays
        scores = scores.numpy()[0]
        labels = labels.numpy()[0]
        boxes = boxes.numpy()[0]
        
        # Scale boxes back to original image size
        boxes /= scale
        
        if boxes.shape[0] > 0:
            # Load original image for drawing predictions
            image_info = dataset.coco.loadImgs(dataset.image_ids[index])[0]
            path = os.path.join(dataset.root_dir, 'images', dataset.set_name, image_info['file_name'])
            output_image = cv2.imread(path)
            
            # Draw predictions on the image
            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt.cls_threshold:
                    break
                    
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                color = colors[pred_label]
                
                # Convert coordinates to integers
                xmin = int(round(float(xmin), 0))
                ymin = int(round(float(ymin), 0))
                xmax = int(round(float(xmax), 0))
                ymax = int(round(float(ymax), 0))
                
                # Draw bounding box
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Draw label background
                text = f"{COCO_CLASSES[pred_label]} : {pred_prob:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(
                    output_image, 
                    (xmin, ymin), 
                    (xmin + text_size[0] + 3, ymin + text_size[1] + 4), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output_image, 
                    text,
                    (xmin, ymin + text_size[1] + 4), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1,
                    (255, 255, 255), 
                    1
                )
            
            # Save the output image
            output_path = f"{opt.output}/{image_info['file_name'][:-4]}_prediction.jpg"
            cv2.imwrite(output_path, output_image)
            print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    opt = get_args()
    test(opt)