import mysql.connector

from flask import Flask, render_template, request, send_from_directory, redirect
from PIL import Image
import numpy as np
import joblib

import os
from skimage.feature import hog


app = Flask(__name__)


db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="your_password",
    database="medicinal_plant_db"
)
model = joblib.load("model/plant_classifier.pkl")
encoder = joblib.load("model/label_encoder.pkl")

IMG_SIZE = (128, 128)

os.makedirs("uploads", exist_ok=True)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


@app.route("/clear-history", methods=["POST"])
def clear_history():

    cursor = db.cursor()

    cursor.execute("DELETE FROM prediction_history")

    db.commit()

    cursor.close()

    return redirect("/")


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    info = {}
    uploaded_image = None
    confidence = 0
    history = []

    total_predictions = 0
    most_predicted = "N/A"
    average_confidence = 0


    if request.method == "POST":

        file = request.files["image"]

        allowed_extensions = ["jpg", "jpeg", "png"]

        if "." not in file.filename or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions:
            return "Invalid file! Please upload JPG, JPEG or PNG image."

        if file:

            uploaded_image = True

            file.save("uploads/uploaded_image.jpg")

            try:
                image = Image.open(
                    "uploads/uploaded_image.jpg"
                ).convert("RGB")

            except Exception:

                return "Invalid or corrupted image. Please upload a valid JPG, JPEG or PNG image."
                            

            image = image.resize(IMG_SIZE)

            image_array = np.array(image)

            gray_image = np.mean(image_array, axis=2)

            features = hog(
                gray_image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )


            prediction_number = model.predict([features])[0]

            decision_scores = model.decision_function([features])

            if len(decision_scores.shape) > 1:
                confidence = np.max(decision_scores[0])
            else:
                confidence = np.max(decision_scores)

            confidence = min(
                max((confidence + 1) * 50, 0),
                100
            )


            prediction = encoder.inverse_transform(
                [prediction_number]
            )[0]


            cursor = db.cursor()

            cursor.execute(
                """
                INSERT INTO prediction_history
                (plant_name, confidence)
                VALUES (%s, %s)
                """,
                (prediction, confidence)
            )

            db.commit()

            cursor.close()


            cursor = db.cursor(dictionary=True)

            plant_name = prediction.replace(
                " bg aug",
                ""
            )

            cursor.execute(
                """
                SELECT *
                FROM plants
                WHERE name = %s
                """,
                (plant_name,)
            )

            info = cursor.fetchone()

            cursor.close()


    cursor = db.cursor(dictionary=True)


    cursor.execute(
        """
        SELECT
            plant_name,
            confidence,
            prediction_time
        FROM prediction_history
        WHERE plant_name IS NOT NULL
        ORDER BY id DESC
        LIMIT 10
        """
    )

    history = cursor.fetchall()


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM prediction_history
        WHERE plant_name IS NOT NULL
        """
    )

    total_predictions = cursor.fetchone()["total"]


    cursor.execute(
        """
        SELECT plant_name, COUNT(*) AS count
        FROM prediction_history
        WHERE plant_name IS NOT NULL
        GROUP BY plant_name
        ORDER BY count DESC
        LIMIT 1
        """
    )

    result = cursor.fetchone()

    if result:
        most_predicted = result["plant_name"]


    cursor.execute(
        """
        SELECT AVG(confidence) AS avg_confidence
        FROM prediction_history
        WHERE plant_name IS NOT NULL
        """
    )

    result = cursor.fetchone()

    if result["avg_confidence"] is not None:
        average_confidence = result["avg_confidence"]


    cursor.close()


    return render_template(
        "index.html",
        prediction=prediction,
        info=info,
        uploaded_image=uploaded_image,
        confidence=confidence,
        history=history,
        total_predictions=total_predictions,
        most_predicted=most_predicted,
        average_confidence=average_confidence
    )


if __name__ == "__main__":
    app.run(debug=True)