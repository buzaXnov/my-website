from flask import Flask, render_template, request

app = Flask(__name__)

def recognie_symbol():
    return "Greek Symbol"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def greek_symbols():
    # Retrieve the coordinates of the strokes from the request data
    strokes = request.form['strokes']
    # TODO: Pass the strokes to your trained neural network for recognition
    # and return the results
    results = recognie_symbol()
    return strokes

if __name__ == '__main__':
    app.run(debug=True)
