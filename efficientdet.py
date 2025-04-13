import tensorflow as tf
import math
import numpy as np
# We'll need to import or implement an equivalent to efficientnet_pytorch
# For this conversion, I'll assume we've imported EfficientNet from a TF package
from tensorflow.keras.applications import EfficientNetB0
from utils import BBoxTransform, ClipBoxes, Anchors
from loss_function import FocalLoss
# TensorFlow equivalent for NMS
from tensorflow.image import non_max_suppression


def nms(dets, thresh):
    boxes = dets[:, :4]
    scores = dets[:, 4]
    indices = non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=thresh)
    return indices


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding='same')
        self.pointwise_conv = tf.keras.layers.Conv2D(
            num_channels, kernel_size=1, strides=1, padding='valid')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=4e-5)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiFPN(tf.keras.layers.Layer):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.p5_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.p4_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.p3_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')

        self.p4_downsample = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.p5_downsample = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.p6_downsample = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.p7_downsample = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        # Weight variables
        # In TensorFlow, we'll use regular variables instead of nn.Parameter
        self.p6_w1 = self.add_weight(name='p6_w1', shape=(2,), initializer='ones', trainable=True)
        self.p5_w1 = self.add_weight(name='p5_w1', shape=(2,), initializer='ones', trainable=True)
        self.p4_w1 = self.add_weight(name='p4_w1', shape=(2,), initializer='ones', trainable=True)
        self.p3_w1 = self.add_weight(name='p3_w1', shape=(2,), initializer='ones', trainable=True)
        
        self.p4_w2 = self.add_weight(name='p4_w2', shape=(3,), initializer='ones', trainable=True)
        self.p5_w2 = self.add_weight(name='p5_w2', shape=(3,), initializer='ones', trainable=True)
        self.p6_w2 = self.add_weight(name='p6_w2', shape=(3,), initializer='ones', trainable=True)
        self.p7_w2 = self.add_weight(name='p7_w2', shape=(2,), initializer='ones', trainable=True)

        # ReLU layers
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.relu(self.p6_w1)
        weight = p6_w1 / (tf.reduce_sum(p6_w1) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.relu(self.p5_w1)
        weight = p5_w1 / (tf.reduce_sum(p5_w1) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.relu(self.p4_w1)
        weight = p4_w1 / (tf.reduce_sum(p4_w1) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.relu(self.p3_w1)
        weight = p3_w1 / (tf.reduce_sum(p3_w1) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.relu(self.p4_w2)
        weight = p4_w2 / (tf.reduce_sum(p4_w2) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.relu(self.p5_w2)
        weight = p5_w2 / (tf.reduce_sum(p5_w2) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.relu(self.p6_w2)
        weight = p6_w2 / (tf.reduce_sum(p6_w2) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.relu(self.p7_w2)
        weight = p7_w2 / (tf.reduce_sum(p7_w2) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        self.conv_layers = []
        self.relu_layers = []
        
        for _ in range(num_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(in_channels, kernel_size=3, strides=1, padding='same'))
            self.relu_layers.append(tf.keras.layers.ReLU())
        
        self.header = tf.keras.layers.Conv2D(
            num_anchors * 4, kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.relu_layers[i](x)
            
        x = self.header(x)
        # In TensorFlow, we use 'channels_last' format by default
        # so our output tensor is [batch, height, width, channels]
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        
        # Reshape to get [batch, -1, 4]
        x = tf.reshape(x, [batch_size, height * width * channels // 4, 4])
        return x


class Classifier(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.conv_layers = []
        self.relu_layers = []
        
        for _ in range(num_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(in_channels, kernel_size=3, strides=1, padding='same'))
            self.relu_layers.append(tf.keras.layers.ReLU())
        
        self.header = tf.keras.layers.Conv2D(
            num_anchors * num_classes, kernel_size=3, strides=1, padding='same')
        self.act = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.relu_layers[i](x)
            
        x = self.header(x)
        x = self.act(x)
        
        # Reshape to get appropriate dimensions
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        
        # Reshape to [batch, -1, num_classes]
        x = tf.reshape(x, [batch_size, height * width * self.num_anchors, self.num_classes])
        return x


class EfficientNetBackbone(tf.keras.layers.Layer):
    def __init__(self):
        super(EfficientNetBackbone, self).__init__()
        # Get the base model with weights
        base_model = EfficientNetB0(include_top=False, weights='imagenet')
        
        # Extract specific layers that correspond to feature maps
        self.model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer('block3a_expand_activation').output,  # C3
                base_model.get_layer('block5c_add').output,               # C4
                base_model.get_layer('top_activation').output              # C5
            ]
        )
        
    def call(self, x):
        return self.model(x)


class EfficientDet(tf.keras.Model):
    def __init__(self, num_anchors=9, num_classes=20, compound_coef=0):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef

        self.num_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]

        self.conv3 = tf.keras.layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='valid')
        self.conv4 = tf.keras.layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='valid')
        self.conv5 = tf.keras.layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='valid')
        self.conv6 = tf.keras.layers.Conv2D(self.num_channels, kernel_size=3, strides=2, padding='same')
        
        self.conv7_relu = tf.keras.layers.ReLU()
        self.conv7_conv = tf.keras.layers.Conv2D(self.num_channels, kernel_size=3, strides=2, padding='same')

        # Create BiFPN layers
        self.bifpn_layers = []
        for _ in range(min(2 + self.compound_coef, 8)):
            self.bifpn_layers.append(BiFPN(self.num_channels))

        self.num_classes = num_classes
        self.regressor = Regressor(
            in_channels=self.num_channels, 
            num_anchors=num_anchors,
            num_layers=3 + self.compound_coef // 3
        )
        
        self.classifier = Classifier(
            in_channels=self.num_channels, 
            num_anchors=num_anchors, 
            num_classes=num_classes,
            num_layers=3 + self.compound_coef // 3
        )

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        # Initialize weights 
        self._init_weights()
        
        self.backbone_net = EfficientNetBackbone()

    def _init_weights(self):
        # In TensorFlow Keras, weight initialization happens in the layers themselves
        # But we can manually override these initializations to match PyTorch's
        
        # Set classifier header bias to achieve prior probability
        prior = 0.01
        bias_initializer = tf.keras.initializers.Constant(-math.log((1.0 - prior) / prior))
        self.classifier.header.bias_initializer = bias_initializer
        
        # Set regressor header weights and bias to zeros
        self.regressor.header.kernel_initializer = 'zeros'
        self.regressor.header.bias_initializer = 'zeros'

    def freeze_bn(self):
        # Find all BatchNormalization layers and set them to inference mode
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    def call(self, inputs, training=False):
        if isinstance(inputs, list) and len(inputs) == 2:
            is_training = True
            img_batch, annotations = inputs
        else:
            is_training = False
            img_batch = inputs

        # Get features from backbone
        features = self.backbone_net(img_batch)
        c3, c4, c5 = features
        
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7_conv(self.conv7_relu(p6))

        features = [p3, p4, p5, p6, p7]
        
        # Apply BiFPN layers sequentially
        for bifpn in self.bifpn_layers:
            features = bifpn(features)

        # Apply regression and classification to each feature
        regression_outputs = [self.regressor(feature) for feature in features]
        regression = tf.concat(regression_outputs, axis=1)
        
        classification_outputs = [self.classifier(feature) for feature in features]
        classification = tf.concat(classification_outputs, axis=1)
        
        anchors = self.anchors(img_batch)

        if is_training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            # Get max scores for each anchor
            scores = tf.reduce_max(classification, axis=2, keepdims=True)
            
            # Filter anchors based on score threshold
            scores_over_thresh = tf.greater(scores[0, :, 0], 0.05)
            
            if tf.reduce_sum(tf.cast(scores_over_thresh, tf.int32)) == 0:
                # Return empty tensors if no detections
                return [tf.zeros(0), tf.zeros(0), tf.zeros((0, 4))]

            # Filter classifications, anchors, and scores
            filtered_classification = tf.boolean_mask(classification[0], scores_over_thresh)
            filtered_anchors = tf.boolean_mask(transformed_anchors[0], scores_over_thresh)
            filtered_scores = tf.boolean_mask(scores[0], scores_over_thresh)
            
            # Combine anchors and scores for NMS
            detections = tf.concat([filtered_anchors, filtered_scores], axis=1)
            
            # Apply NMS
            anchors_nms_idx = nms(detections, 0.5)
            
            # Get final detections
            nms_classifications = tf.gather(filtered_classification, anchors_nms_idx)
            nms_scores = tf.reduce_max(nms_classifications, axis=1)
            nms_classes = tf.argmax(nms_classifications, axis=1)
            nms_anchors = tf.gather(filtered_anchors, anchors_nms_idx)
            
            return [nms_scores, nms_classes, nms_anchors]


if __name__ == '__main__':
    def count_parameters(model):
        return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

    model = EfficientDet(num_classes=80)
    print(count_parameters(model))