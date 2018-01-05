import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy
import scipy.misc
from keras import backend as K
from keras.models import load_model

from YAD2K.yad2k.models.keras_yolo import yolo_head, yolo_eval
from network.yolo_utils import read_classes, read_anchors, preprocess_image, generate_colors, draw_boxes

sess = K.get_session()
class_names = read_classes("YAD2K/model_data/coco_classes.txt")
anchors = read_anchors("YAD2K/model_data/yolo_anchors.txt")
# image_shape = (720., 1280.)
image_shape = (600., 800.)
yolo_model = load_model("YAD2K/model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape, score_threshold=0.4)

def predict():
    global sess
    imgs = os.listdir('imgs')[:1]

    for img in imgs:
        out_scores, out_boxes, out_classes = yolo_predict(sess, img)
        print('out_scores', out_scores)
        print('out_boxes', out_boxes)
        print('out_classes', out_classes)



def yolo_predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_image("imgs/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    #out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    out_scores, out_boxes, out_classes = sess.run([boxes, scores, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    #out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict={yolo_model.input: image_data,input_image_shape: [image.size[1], image.size[0]],K.learning_phase(): 0})

    ### END CODE HERE ###

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #print('image', image)
    #draw_boxes(np.array(image), out_boxes, out_classes, class_names, scores=out_scores)
    #draw_boxes(image, boxes, box_classes, class_names, scores=None) #coursera
    #draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors) native


    # Save the predicted bounding box on the image
    #image.save(os.path.join("data", image_file), quality=90)
    # Display the results in the notebook
    #output_image = scipy.misc.imread(os.path.join("data", image_file))
    # output_image = scipy.misc.imread(image)
    #imshow(output_image)
    imshow(image)
    plt.show()

    return out_scores, out_boxes, out_classes


# out_scores, out_boxes, out_classes = predict(sess, "0114.jpg")
# out_scores, out_boxes, out_classes = predict(sess, "IMAG0595.jpg")