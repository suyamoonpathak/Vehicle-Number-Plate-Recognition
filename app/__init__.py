from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pytesseract
import re
import json
from nanonets import NANONETSOCR
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
            img = cv2.imdecode(np.fromstring(
                f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (900, 600))
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection to find edges
            edges = cv2.Canny(blur, 50, 150)

            # Find contours in the image
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                        cv2.drawContours(img, [approx], 0, (0, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(approx)

            try:
                if (len(location) != 0):
                    mask = np.zeros(gray.shape, np.uint8)
                    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                    new_image = cv2.bitwise_and(img, img, mask=mask)
            # plate_roi = new_image[y:y+h, x:x+w]
            except:
                new_image = img

            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Cropped.jpg', new_image)

            # OCR

            # Replace YOUR_API_KEY and YOUR_MODEL_ID with your actual values
            api_key = '99e17aa2-dd30-11ed-9a00-8e45386f1393'
            model_id = '45c3ea0d-dc25-4d41-99f4-5aed304f92ed'
            # Initialise
            model = NANONETSOCR()

            # Authenticate
            # This software is perpetually free :)
            # You can get your free API key (with unlimited requests) by creating a free account on https://app.nanonets.com/#/keys?utm_source=wrapper.
            model.set_token(api_key)

            # PDF / Image to Raw OCR Engine Output
            pred_json = model.convert_to_prediction('Cropped.jpg')
            print(json.dumps(pred_json, indent=2))

            # PDF / Image to String
            string = model.convert_to_string('Cropped.jpg')

            # print(string)

            def filter_text(text, regex):
                # Replace all occurrences of O with 0 and I with 1
                text = re.sub('O', '0', text.upper())
                text = re.sub('I', '1', text)

                # Extract only uppercase letters and digits
                filtered_text = ''.join(filter(str.isalnum, text.upper()))

                # Apply regex to filtered text
                match = re.search(regex, filtered_text)
                if match:
                    return match.group()
                else:
                    # Remove non-matching characters from start of the string
                    while len(filtered_text) > 0 and not re.match(regex, filtered_text):
                        filtered_text = filtered_text[1:]
                    # Remove non-matching characters from end of the string
                    while len(filtered_text) > 0 and not re.match(regex, filtered_text):
                        filtered_text = filtered_text[:-1]
                    # Check if the resulting text matches the regex
                    match = re.search(regex, filtered_text)
                    if match:
                        return match.group()
                    else:
                        # If there are multiple sections that could potentially match the regex pattern,
                        # try each section and return the first matching one
                        sections = re.findall('[A-Z0-9]+', text.upper())
                        for section in sections:
                            section = re.sub('O', '0', section)
                            section = re.sub('I', '1', section)
                            match = re.search(regex, section)
                            if match:
                                return match.group()
                        # If no matching sections are found, return None
                        return None

            # for indian number plates
            regex = '^[A-Z]{2}[0-9]{2}[A-HJ-NP-Z]{1,2}[0-9]{4}$|^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$'
            text = filter_text(string, regex)
            if (text == None):
                text = string + " (apologies in case of noisy output)"
            return render_template('result.html', text=text, photo=f)
    except:
        return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
