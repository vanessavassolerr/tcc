from flask import Flask
import test2

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Olá, mundo!'

@app.route('/enviar_contagem', methods=['POST'])
def enviar_contagem_carros():
    carros = test2.carros
    return f'O número enviado foi: {carros}'

if __name__ == '__main__':
    app.run()
