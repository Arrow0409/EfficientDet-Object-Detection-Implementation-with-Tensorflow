import tensorflow as tf


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    
    # Expand dimensions to broadcast properly
    a_expanded = tf.expand_dims(a, axis=1)
    
    iw = tf.minimum(a_expanded[:, :, 2], b[:, 2]) - tf.maximum(a_expanded[:, :, 0], b[:, 0])
    ih = tf.minimum(a_expanded[:, :, 3], b[:, 3]) - tf.maximum(a_expanded[:, :, 1], b[:, 1])
    
    iw = tf.maximum(iw, 0)
    ih = tf.maximum(ih, 0)
    
    ua = tf.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = tf.maximum(ua, 1e-8)
    
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def call(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = tf.shape(classifications)[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            # Filter valid annotations (where the class is not -1)
            valid_indices = tf.where(tf.not_equal(bbox_annotation[:, 4], -1))
            bbox_annotation = tf.gather_nd(bbox_annotation, valid_indices)

            if tf.shape(bbox_annotation)[0] == 0:
                regression_losses.append(tf.constant(0.0))
                classification_losses.append(tf.constant(0.0))
                continue

            classification = tf.clip_by_value(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            
            IoU_max = tf.reduce_max(IoU, axis=1)
            IoU_argmax = tf.argmax(IoU, axis=1)

            # compute the loss for classification
            targets = tf.ones_like(classification) * -1

            # Background targets for IoU < 0.4
            negative_indices = tf.less(IoU_max, 0.4)
            targets = tf.where(
                tf.expand_dims(negative_indices, -1),
                tf.zeros_like(targets),
                targets
            )

            # Positive indices for IoU >= 0.5
            positive_indices = tf.greater_equal(IoU_max, 0.5)
            num_positive_anchors = tf.reduce_sum(tf.cast(positive_indices, tf.float32))

            # Get corresponding annotations for each anchor
            assigned_annotations = tf.gather(bbox_annotation, IoU_argmax)

            # Set positive targets
            targets = tf.where(
                tf.expand_dims(positive_indices, -1),
                tf.zeros_like(targets),
                targets
            )
            
            # Create indices for scatter update
            pos_idx = tf.where(positive_indices)
            class_idx = tf.cast(tf.gather_nd(assigned_annotations[:, 4], pos_idx), tf.int32)
            scatter_indices = tf.stack([pos_idx[:, 0], class_idx], axis=1)
            
            # Create sparse tensor for scatter update
            sparse_targets = tf.sparse.SparseTensor(
                indices=scatter_indices,
                values=tf.ones_like(class_idx, dtype=tf.float32),
                dense_shape=tf.shape(targets)
            )
            # Convert to dense
            target_updates = tf.sparse.to_dense(sparse_targets)
            targets = tf.where(tf.greater(target_updates, 0), target_updates, targets)

            # Alpha factor
            alpha_factor = tf.ones_like(targets) * alpha
            alpha_factor = tf.where(tf.equal(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            
            # Focal weight
            focal_weight = tf.where(tf.equal(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * tf.pow(focal_weight, gamma)

            # BCE loss
            bce = -(targets * tf.math.log(classification) + (1.0 - targets) * tf.math.log(1.0 - classification))
            cls_loss = focal_weight * bce

            # Zero out ignored regions (targets == -1)
            cls_loss = tf.where(tf.not_equal(targets, -1.0), cls_loss, tf.zeros_like(cls_loss))
            
            classification_losses.append(tf.reduce_sum(cls_loss) / tf.maximum(num_positive_anchors, 1.0))

            # Regression loss calculation
            if tf.reduce_sum(tf.cast(positive_indices, tf.int32)) > 0:
                # Get positive annotations
                pos_assigned_annotations = tf.boolean_mask(assigned_annotations, positive_indices)
                
                anchor_widths_pi = tf.boolean_mask(anchor_widths, positive_indices)
                anchor_heights_pi = tf.boolean_mask(anchor_heights, positive_indices)
                anchor_ctr_x_pi = tf.boolean_mask(anchor_ctr_x, positive_indices)
                anchor_ctr_y_pi = tf.boolean_mask(anchor_ctr_y, positive_indices)

                gt_widths = pos_assigned_annotations[:, 2] - pos_assigned_annotations[:, 0]
                gt_heights = pos_assigned_annotations[:, 3] - pos_assigned_annotations[:, 1]
                gt_ctr_x = pos_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = pos_assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = tf.maximum(gt_widths, 1.0)
                gt_heights = tf.maximum(gt_heights, 1.0)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = tf.math.log(gt_widths / anchor_widths_pi)
                targets_dh = tf.math.log(gt_heights / anchor_heights_pi)

                targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
                
                norm = tf.constant([[0.1, 0.1, 0.2, 0.2]], dtype=tf.float32)
                targets = targets / norm

                # Get regression predictions for positive indices
                regression_pos = tf.boolean_mask(regression, positive_indices)
                
                # Smooth L1 loss
                regression_diff = tf.abs(targets - regression_pos)
                
                regression_loss = tf.where(
                    tf.less_equal(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * tf.square(regression_diff),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(tf.reduce_mean(regression_loss))
            else:
                regression_losses.append(tf.constant(0.0))

        # Stack and mean the losses
        classification_loss = tf.reduce_mean(tf.stack(classification_losses), keepdims=True)
        regression_loss = tf.reduce_mean(tf.stack(regression_losses), keepdims=True)
        
        return classification_loss, regression_loss