import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
from flask import Flask, request, jsonify
import contagem

app = Flask(__name__)

# Configurações do MQTT Broker
#broker.mqtt-dashboard.com alternativo
# arduino e broker tem q estar no mesmo wifi
mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883
mqtt_topic = "esp32/topic"

@app.route('/')
def teste():
    return 'Teste'

@app.route('/contagem', methods=['POST'])
def contagem():
    try:
        veiculos = contagem.carros_ultimos_15s
        # Enviar a resposta para o ESP32 via MQTT
        publish.single(mqtt_topic, veiculos, hostname=mqtt_broker, port=mqtt_port)
        print(veiculos)
        return jsonify({"Veiculos contados nos ultimos 15 segundos:": veiculos})
    except Exception as e:
        return f'Erro: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)