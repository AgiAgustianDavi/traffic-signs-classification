from flask import Flask, request, render_template

app = Flask(__name__)
model = keras.models.load_model("traffic_sign_model.h5")
class_labels = list(train_generator.class_indices.keys())

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_path = "static/uploaded_image.jpg"
        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        class_name = class_labels[class_idx]
        return render_template('index.html', class_name=class_name, image_path=file_path)
    return render_template('index.html', class_name=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)