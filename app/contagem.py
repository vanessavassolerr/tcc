import numpy as np
import cv2
import sys
from time import sleep

VIDEO = "tcc/video/visao_guarita_v1.mp4"  # Caminho do vídeo
algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']


# Variáveis de configuração

w_min = 45  # largura minima do retangulo
h_min = 45  # altura minima do retangulo
carros = 0


#------------------------------------------------------------------------------------

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation
    
def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120, 
                                                        decisionThreshold=0.8)
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 100, nmixtures = 5,
                                                        backgroundRatio = 0.7, 
                                                        noiseSigma = 0)
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True,
                                                varThreshold=100)
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, 
                                                 detectShadows=True)
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, 
                                                        useHistory =True,
                                                        maxPixelStability=15*60,
                                                        isParallel=True)
    print('Detector inválido')
    sys.exit(1)
    
    
#------------------------


def centroide(x, y, w, h):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param w: largura do objeto
    :param h: altura do objeto
    :return: tupla q retorna as coordenadas do centro do carro
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


class Veiculo:
    def __init__(self, posicao):
        self.posicao = posicao
        self.frames_desde_ultima_vista = 0
       

detec = []
veiculos = []

def set_info(detec):
    global carros
    global veiculos

    for veiculo in veiculos:
        veiculo.frames_desde_ultima_vista += 1
    for centro in detec:
        if (298) < centro[1] < (313):  # condição de linha de contagem
            novo_veiculo = True
            # Verificar se o centro está perto de um veículo existente
            for veiculo in veiculos:
                distancia = np.linalg.norm(np.array(centro) - np.array(veiculo.posicao))
                if distancia < 120 and veiculo.frames_desde_ultima_vista < 30:  # 30 é um valor de exemplo
                    novo_veiculo = False
                    veiculo.posicao = centro  # Atualiza a posição do veículo
                    veiculo.frames_desde_ultima_vista = 0  # Reseta o contador
                    break
            
            if novo_veiculo:
                carros += 1
                veiculos.append(Veiculo(centro))
                print("Carros detectados até o momento: " + str(carros))


def show_info(frame):
    text = f'Carros: {carros}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
   

cap = cv2.VideoCapture(VIDEO) # lendo o vídeo
fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec do vídeo

algorithm_type = algorithm_types[1]
background_subtractor = Subtractor(algorithm_type)  # pega o fundo e subtrai do que se movendo

while True:
    ok, frame = cap.read() # pega cada frame do vídeo
    
    if not ok:
        break
    
    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')
    
    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (510, 250), (640, 280), (255, 107, 25), 2) 

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)
        if not validar_contorno:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
        centro = centroide(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 3, (0, 0, 255), -1)
        print(centro)
    


    set_info(detec)
    show_info(frame)



    if cv2.waitKey(1) == 27: #ESC
        break

cv2.destroyAllWindows()
cap.release()
#Total de veiculos que passaram: 85
#Contagem: 73