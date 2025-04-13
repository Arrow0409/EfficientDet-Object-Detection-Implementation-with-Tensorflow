import os
import tensorflow as tf
import numpy as np

from pycocotools.coco import COCO
import cv2

from utils import BBoxTransform, ClipBoxes, Anchors
from loss_function import FocalLoss


class CocoDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, set='train2017', transform=None, batch_size=1):
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.batch_size = batch_size

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_images = []
        batch_annotations = []
        batch_scales = []

        for i in range(self.batch_size):
            index = idx * self.batch_size + i
            if index >= len(self.image_ids):
                break

            img = self.load_image(index)
            annot = self.load_annotations(index)
            sample = {'img': img, 'annot': annot}
            
            if self.transform:
                sample = self.transform(sample)
                
            batch_images.append(sample['img'])
            batch_annotations.append(sample['annot'])
            if 'scale' in sample:
                batch_scales.append(sample['scale'])
        
        # Preparing the batch
        if 'scale' in sample:
            return {'img': tf.stack(batch_images, axis=0), 
                    'annot': tf.ragged.constant(batch_annotations).to_tensor(default_value=-1), 
                    'scale': tf.convert_to_tensor(batch_scales)}
        else:
            return {'img': tf.stack(batch_images, axis=0), 
                    'annot': tf.ragged.constant(batch_annotations).to_tensor(default_value=-1)}

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        return 80


def create_tf_dataset(dataset, batch_size=2):
    """Create a TensorFlow dataset from a CocoDataset"""
    
    def generator():
        for i in range(len(dataset)):
            batch = dataset[i]
            yield batch['img'], batch['annot']
    
    output_signature = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32)
    )
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )


class Resizer:
    """Convert numpy arrays in sample to TensorFlow tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': tf.convert_to_tensor(new_image, dtype=tf.float32), 
                'annot': tf.convert_to_tensor(annots, dtype=tf.float32), 
                'scale': scale}


class Augmenter:
    """Data augmentation for training."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            if isinstance(image, tf.Tensor):
                image = image.numpy()
            if isinstance(annots, tf.Tensor):
                annots = annots.numpy()
                
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            if 'scale' in sample:
                scale = sample['scale']
                sample = {'img': tf.convert_to_tensor(image, dtype=tf.float32), 
                          'annot': tf.convert_to_tensor(annots, dtype=tf.float32),
                          'scale': scale}
            else:
                sample = {'img': tf.convert_to_tensor(image, dtype=tf.float32), 
                          'annot': tf.convert_to_tensor(annots, dtype=tf.float32)}

        return sample


class Normalizer:
    """Normalize images with mean and standard deviation."""

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        normalized_image = (image.astype(np.float32) - self.mean) / self.std
        
        if 'scale' in sample:
            scale = sample['scale']
            return {'img': tf.convert_to_tensor(normalized_image, dtype=tf.float32), 
                    'annot': annots,
                    'scale': scale}
        else:
            return {'img': tf.convert_to_tensor(normalized_image, dtype=tf.float32), 
                    'annot': annots}


def prepare_data_pipeline(root_dir, set_name='train2017', batch_size=2, common_size=512, augmentation=True):
    """Create a complete data pipeline with transformations"""
    # Initialize utility classes
    bbox_transform = BBoxTransform()
    clip_boxes = ClipBoxes()
    anchors = Anchors()
    
    # Initialize loss function
    focal_loss = FocalLoss()
    
    transforms = []
    
    # Add normalizer
    transforms.append(Normalizer())
    
    # Add resizer
    transforms.append(Resizer())
    
    # Add augmentation for training
    if augmentation and set_name == 'train2017':
        transforms.append(Augmenter())
    
    def apply_transforms(sample):
        for transform in transforms:
            sample = transform(sample)
        return sample
    
    # Create dataset
    dataset = CocoDataset(
        root_dir=root_dir,
        set=set_name,
        transform=apply_transforms,
        batch_size=batch_size
    )
    
    # Create TensorFlow dataset
    tf_dataset = create_tf_dataset(dataset, batch_size)
    
    # Set prefetch for better performance
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset, dataset, {
        'bbox_transform': bbox_transform,
        'clip_boxes': clip_boxes,
        'anchors': anchors,
        'focal_loss': focal_loss
    }