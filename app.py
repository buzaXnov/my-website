import ast
import torch
from flask import Flask, render_template, request
from utils.model import preprocess_symbol_vector, load_model

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


def recognize_symbol(strokes, model):
    """Recognize the symbol from the given strokes."""
    symbols = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    strokes = preprocess_symbol_vector(strokes)
    result = model.predict(torch.tensor(strokes[None, ...]))

    return symbols[result.item()]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/greek-symbols', methods=['POST', 'GET'])
def greek_symbols():
    model = load_model()

    if request.method == 'GET':
        return render_template('greek_symbols.html')
    elif request.method == 'POST':
        # Retrieve the coordinates of the strokes from the request data
        strokes = request.form['strokes']

        strokes = ast.literal_eval(strokes)

        return f"Predicted symbol is {recognize_symbol(strokes, model)}"

        # return render_template('greek_symbols.html', symbol=symbol)


@app.route('/greek-feedback', methods=['POST'])
def greek_feedback():
    feedback = request.get_json().get('feedback')
    # TODO: Do something with the feedback data, such as store it in a database
    return 'Feedback received'


if __name__ == '__main__':
    app.run(debug=True)