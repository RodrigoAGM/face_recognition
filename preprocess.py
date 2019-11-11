from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import facenet
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

class preprocesses:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dataset = facenet.get_dataset(self.input_datadir)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0

            for cls in dataset:

                output_class_dir = os.path.join(output_dir, cls.name)

                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)

                    if not os.path.exists(output_filename):
                        try:
                            img = cv2.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue

                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)

                            img = img[:, :, 0:3]

                            bb_temp = [0,0,0,0]
                            flag = False
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                            for (x,y,w,h) in faces:
                                bb_temp = [x,y,w,h]

                                if bb_temp[0] == 0 & bb_temp[1]==0 & bb_temp[2]==0 & bb_temp[3] ==0:
                                    print("Face not detected")
                                else: 
                                    roi_color = img[y:y+h, x:x+w]
                                    roi_color = cv2.resize(roi_color,(160,160), interpolation=cv2.INTER_LINEAR)
                                    img_item = output_filename
                                    cv2.imwrite(img_item, roi_color)

                            nrof_successfully_aligned += 1
                            text_file.write('%s %d %d %d %d\n' % (
                            output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))                                


        return (nrof_images_total,nrof_successfully_aligned)