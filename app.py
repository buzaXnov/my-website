import ast
import torch
import ast
import torch
from flask import Flask, render_template, request, url_for
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

        return recognize_symbol(strokes, model)


if __name__ == '__main__':
    app.run(debug=True)

# turn this string of a list of lists into a list of lists with floats; give me a func for that

