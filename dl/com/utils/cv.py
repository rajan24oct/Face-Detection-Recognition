import cv2, sys, numpy, os


class cvDlTrain:

    size = 2
    fn_haar = 'opencv_cascade/haarcascade_frontalface_default.xml'
    fn_dir = 'database'

    model = None


    def __init__(self):
        self.haar_cascade = cv2.CascadeClassifier(self.fn_haar)



    def webcamInit(self, fn_name):
        self.fn_name = fn_name
        self.path = os.path.join(self.fn_dir, self.fn_name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        # Generate name for image file
        self.pin = sorted([int(n[:n.find('.')]) for n in os.listdir(self.path)
                           if n[0] != '.'] + [0])[-1] + 1

        self.webcam = cv2.VideoCapture(0)



    def train(self, fn_name):
        self.webcamInit(fn_name)
        (im_width, im_height) = (112, 92)
        count = 0
        pause = 0
        count_max = 20
        while count < count_max:

            # Loop until the camera is working
            rval = False
            while (not rval):
                # Put the image from the webcam into 'frame'
                (rval, frame) = self.webcam.read()
                if (not rval):
                    print("Failed to open webcam. Trying again...")

            # Get image size
            height, width, channels = frame.shape

            # Flip frame
            frame = cv2.flip(frame, 1, 0)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scale down for speed
            mini = cv2.resize(gray, (int(gray.shape[1] / self.size), int(gray.shape[0] / self.size)))

            # Detect faces
            faces = self.haar_cascade.detectMultiScale(mini)

            # We only consider largest face
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * self.size for v in face_i]

                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))

                # Draw rectangle and write name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, self.fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0))

                # Remove false positives
                if (w * 6 < width or h * 6 < height):
                    print("Face too small")
                else:

                    # To create diversity, only save every fith detected image
                    if (pause == 0):
                        print("Saving training sample " + str(count + 1) + "/" + str(count_max))

                        # Save image file
                        cv2.imwrite('%s/%s.png' % (self.path, self.pin), face_resize)

                        self.pin += 1
                        count += 1

                        pause = 1

            if (pause > 0):
                pause = (pause + 1) % 5

            cv2.namedWindow("DigitalLab", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("DigitalL"
                                  "ab", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('DigitalLab', frame)
            key = cv2.waitKey(10)
            if key == 27:
                break


    def trainModel(self):


        # Create a list of images and a list of corresponding names
        (images, lables, names, id) = ([], [], {}, 0)

        # Get the folders containing the training data
        for (subdirs, dirs, files) in os.walk(self.fn_dir):

            # Loop through each folder named after the subject in the photos
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(self.fn_dir, subdir)

                # Loop through each photo in the folder
                for filename in os.listdir(subjectpath):

                    # Skip non-image formates
                    f_name, f_extension = os.path.splitext(filename)
                    if (f_extension.lower() not in
                            ['.png', '.jpg', '.jpeg', '.gif', '.pgm']):
                        print("Skipping " + filename + ", wrong file type")
                        continue
                    path = subjectpath + '/' + filename
                    lable = id

                    # Add to training data
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1
        (im_width, im_height) = (112, 92)

        # Create a Numpy array from the two lists above
        (images, lables) = [numpy.array(lis) for lis in [images, lables]]

        # OpenCV trains a model from the images
        # NOTE FOR OpenCV2: remove '.face'
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, lables)
        self.model = model
        self.names = names


    def recognisePeople(self):

        (im_width, im_height) = (112, 92)
        self.webcam = cv2.VideoCapture(0)


        while True:

            # Loop until the camera is working
            rval = False
            while (not rval):
                # Put the image from the webcam into 'frame'
                (rval, frame) = self.webcam.read()
                if (not rval):
                    print("Failed to open webcam. Trying again...")

            # Flip the image (optional)
            frame = cv2.flip(frame, 1, 0)

            # Convert to grayscalel
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to speed up detection (optinal, change size above)
            mini = cv2.resize(gray, (int(gray.shape[1] / self.size), int(gray.shape[0] / self.size)))

            # Detect faces and loop through each one
            faces = self.haar_cascade.detectMultiScale(mini)
            for i in range(len(faces)):
                face_i = faces[i]

                # Coordinates of face after scaling back by `size`
                (x, y, w, h) = [v * self.size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))

                # Try to recognize the face
                prediction = self.model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # [1]
                # Write the name of recognized face
                cv2.putText(frame,
                            ' %s , got you' % (self.names[prediction[0]]),
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                # cv2.putText(frame,
                #             'Welcome %s We got you - %.0f' % (names[prediction[0]], prediction[1]),
                #             (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # Show the image and check for ESC being pressed

            cv2.namedWindow("DigitalLab", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("DigitalLab", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('DigitalLab', frame)

            key = cv2.waitKey(10)
            if key == 27:
                break