from flask import Flask, render_template, request, url_for

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


def recognize_symbol():
    return "Greek Symbol"


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/greek-symbols', methods=['POST', 'GET'])
def greek_symbols():
    if request.method == 'GET':
        return render_template('greek_symbols.html')
    elif request.method == 'POST':
        # Retrieve the coordinates of the strokes from the request data
        strokes = request.form['strokes']
        # TODO: Pass the strokes to your trained neural network for recognition
        # and return the results
        results = recognize_symbol()
        return strokes


if __name__ == '__main__':
    app.run(debug=True)
