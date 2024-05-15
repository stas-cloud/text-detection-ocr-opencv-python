import cv2
import easyocr
import matplotlib.pyplot as plt


THRESHOLD = 0.25 # 

# read image
image_path = './data/test.jpeg'

img = cv2.imread(image_path)

# instantiate text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_from_img = reader.readtext(img)

# draw bbox and text
for t in text_from_img:
    print(t)

    bbox, text, score = t

    if score > THRESHOLD:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.85, (255, 0, 0), 3)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
