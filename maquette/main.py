from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def main():

    return render_template('index.html', expected_result="feu", predictions=[["feu",80],["non_feu",20]])


if __name__ == "__main__":
    app.run(debug = True)
