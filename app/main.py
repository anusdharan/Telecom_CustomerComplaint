from flask import Flask, request, render_template
from services import fetch_prediction as fp

app = Flask(__name__)
@app.route('/')
def my_form():
      return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
        complaint = request.form['complaint']
        prediction = fp.process_text(complaint)
        return render_template('model_predictions.html', result = prediction)

if __name__ == '__main__':
      app.run(debug=True)
