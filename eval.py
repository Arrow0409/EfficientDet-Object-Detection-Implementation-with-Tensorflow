from dataset import CocoDataset, Resizer, Normalizer
from torchvision import transforms
from pycocotools.cocoeval import COCOeval
import json
import tensorflow as tf


def evaluate_coco(dataset, model, threshold=0.05):
    results = []
    image_ids = []

    for index in range(len(dataset)):
        data = dataset[index]
        scale = data['scale']
        
        # Convert PyTorch tensor to TensorFlow tensor and adjust dimensions
        img = tf.convert_to_tensor(data['img'].numpy(), dtype=tf.float32)
        img = tf.transpose(img, [2, 0, 1])  # Change from HWC to CHW format
        img = tf.expand_dims(img, axis=0)   # Add batch dimension
        
        # Run inference
        scores, labels, boxes = model(img)
        
        # Convert output tensors to numpy arrays for processing
        scores = scores.numpy()
        labels = labels.numpy()
        boxes = boxes.numpy()
        
        # Rescale boxes to original image size
        boxes /= scale

        if boxes.shape[0] > 0:
            # Convert from [x1, y1, x2, y2] to [x, y, width, height] format
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                if score < threshold:
                    break

                image_result = {
                    'image_id': dataset.image_ids[index],
                    'category_id': dataset.label_to_coco_label(label),
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

        # append image to list of processed images
        image_ids.append(dataset.image_ids[index])

        # print progress
        print('{}/{}'.format(index, len(dataset)), end='\r')

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    # Load the model (converted from PyTorch to TensorFlow)
    # You'll need to implement the model loading for TensorFlow separately
    efficientdet = tf.keras.models.load_model("trained_models/signatrix_efficientdet_coco_tf")
    
    dataset_val = CocoDataset("data/COCO", set='val2017',
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    evaluate_coco(dataset_val, efficientdet)