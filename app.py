from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            f = request.files['file']
            # Read the image
            img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (900, 600))
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection to find edges
            edges = cv2.Canny(blur, 50, 150)

            # Find contours in the image
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate through all contours and find the one that represents the number plate
            location = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    # Approximate the contour to a polygon
                    approx = cv2.approxPolyDP(
                        contour, 0.02*cv2.arcLength(contour, True), True)

                    # Check if the polygon has four sides (as number plates usually do)
                    if len(approx) == 4:
                        # Draw a green rectangle around the number plate
                        location = approx
                        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]

            kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])*10
            image_sharp = cv2.filter2D(src=Cropped, ddepth=-1, kernel=kernel)
            out = cv2.GaussianBlur(image_sharp,(5,5),0)

            config = ('--dpi 300 --oem 1 --psm 6')
            text = pytesseract.image_to_string(Cropped, config=config)

            return render_template('result.html', text=text, photo=f)
    except:
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
