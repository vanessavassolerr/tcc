import paho.mqtt.publish as publish

mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883
mqtt_topic = "esp32/topic"

publish.single(mqtt_topic, "Teste", hostname=mqtt_broker, port=mqtt_port)
print("Mensagem enviada")
