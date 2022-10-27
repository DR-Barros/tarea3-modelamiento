from random import randint, random
import glfw
from OpenGL.GL import *
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
from grafica.assets_path import getAssetPath

#clase para guardar el control de la aplicación
class Controller:
    def __init__(self):
        self.camType = 0
        self.position = np.array([0, -5, 0.5])
        self.theta = 0
        self.phi = 0
        self.front = np.array([0, 1, 0])
        self.aceleartion = 1
        #doblar  toma valores -1 (izquierda), 0, 1 (derecha)
        self.doblar = 0
        self.view = tr.lookAt(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 1]))
        #backflip
        self.backflip = 0
        self. backfliped = False
        #camara jupiter
        self.jupiter = 0
        #Terminar programa
        self.close = 0
        self.closed = False
    def derecha(self):
        self.doblar = 1
        self.phi = self.phi + np.pi/100 *np.cos(self.theta)/abs(np.cos(self.theta))
        self.front = np.array([
            np.sin(self.phi)*np.cos(self.theta),
            np.cos(self.phi)*np.cos(self.theta),
            np.sin(self.theta)
        ])
    def izquierda(self):
        self.doblar = -1
        self.phi = self.phi - np.pi/100 * np.cos(self.theta)/abs(np.cos(self.theta))
        self.front = np.array([
            np.sin(self.phi)*np.cos(self.theta),
            np.cos(self.phi)*np.cos(self.theta),
            np.sin(self.theta)
        ])
    def arriba(self):
        self.theta = self.theta + 0.03
        self.front = np.array([
            np.sin(self.phi)*np.cos(self.theta),
            np.cos(self.phi)*np.cos(self.theta),
            np.sin(self.theta)
        ])
        if self.theta > 2*np.pi:
            self.theta = 0 
    def abajo(self):
        self.theta = self.theta - 0.03
        self.front = np.array([
            np.sin(self.phi)*np.cos(self.theta),
            np.cos(self.phi)*np.cos(self.theta),
            np.sin(self.theta)
        ])
        if self.theta < 0:
            self.theta = 2*np.pi
    def camera(self, time, step):
        if self.camType == 0:
            up = np.array([0, 0, np.cos(self.theta)])
            self.view = tr.lookAt(
                controller.position - controller.front*0.01 + np.array([0, 0, 0.001*np.cos(self.theta)]),
                controller.position + np.array([0, 0, 0.001*np.cos(self.theta)]),
                up/np.linalg.norm(up)
            )
        elif self.camType == 1:
            posicionFront = np.array([0.08, -0.05, 0.02])
            posicionAt = np.array([
                destroyerCurve[(step*4)%12800, 0],
                destroyerCurve[(step*4)%12800, 1],
                destroyerCurve[(step*4)%12800, 2]
            ])
            self.view = tr.lookAt(
                posicionAt + posicionFront,
                posicionAt,
                np.array([0, 0, 1])
            )
        elif self.camType == 2:
            #cometa
            Rcometa = (10- 2*np.cos(np.pi*time/250))
            self.view = tr.lookAt(
                np.array([Rcometa*np.cos(np.pi*time/250)*1.01, -1.05, Rcometa*np.sin(np.pi*time/250)*1.01]),
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
            )
        elif self.camType ==3:
            #tierra
            self.view = tr.lookAt(
                np.array([-4*np.cos(time/365), -4*np.sin(time/365), 0.1]),
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
            )
        elif self.camType ==4:
            #saturno
            self.view = tr.lookAt(
                np.array([28*np.cos(time/10767+np.pi/4), 28*np.sin(time/10767+np.pi/4), 0.8795*np.cos(time/10767)+0.3]),
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
            )
        
        elif self.camType ==5:
            posicionAt = np.array([
                convoy3Curve[(step*10)%22000, 0],
                convoy3Curve[(step*10)%22000, 1],
                convoy3Curve[(step*10)%22000, 2]
            ])
            self.view = tr.lookAt(posicionAt*1.05 + np.array([0,0, 0.1]), posicionAt, np.array([0, 0, 1]))
        return self.view
        
#iniciamos el controlador
controller = Controller()

# funcion para facilitar inicialización
def createGPUShape(shape, pipeline):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

#Funcion crear esfera
def crearEsfera(N, r,g,b):
    vertices = []
    indices = []
    angulo = 2 * np.pi
    n = int(N/2)-1
    for i in range(N):
        indices += [N*n, i*n, ((i+1)*n)%(N*n)]
        rand = random()/8
        omega = i/N * angulo
        for j in range(n):
            theta = (j+1)/N * angulo
            vertices += [
                    np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta), abs(r), abs(g), abs(b)
                ]
        for j in range(n-1):
            indices += [
                n*i+j, n*i+1+j, (n*(i+1)+j)%(N*n),
                n*i+1+j, (n*(i+1)+j)%(N*n), (n*(i+1)+1+j)%(N*n),
            ]
        indices += [N*n+1, (i*n-1)%(N*n), ((i+1)*n-1)%(N*n)]
    
    vertices += [
        0, 0, 1, r, g, b,
        0, 0, -1, r, g, b 
    ] 
    return bs.Shape(vertices, indices)

#Funcion crear esfera para phong
def esferaPhong(N):
    vertices = []
    faces = []
    angulo = 2 * np.pi
    n = int(N/2)
    for i in range(N+1):
        omega = i/N * angulo
        if i == N:
            omega = 0
        for j in range(n+1):
            theta = j/(N)* angulo
            vertices += [
                [np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta),
                i/N, j/n,
                np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta)],
            ]
            if i != N:
                faces +=[
                    [(n+1)*i+j, (n+1)*i+1+j, ((n+1)*(i+1)+j)%((N+1)*(n+1))],
                    [(n+1)*i+1+j, ((n+1)*(i+1)+j)%((N+1)*(n+1)), ((n+1)*(i+1)+1+j)%((N+1)*(n+1))]
                ]
    indices = []
    vertexData = []
    index = 0
    for face in faces:
        vertex = vertices[face[0]]
        vertexData += vertex
        vertex = vertices[face[1]]
        vertexData += vertex
        vertex = vertices[face[2]]
        vertexData += vertex

        indices += [index, index + 1, index + 2]
        index += 3  
    
    return bs.Shape(vertexData, indices)

#Funcion crear anillo
def crearAnillo(N, R):
    vertices = []
    faces = []
    for i in range(N):
        theta = i/N * 2*np.pi
        theta2 = (i+1)/N * 2*np.pi
        vertices += [
            [R*np.cos(theta), R*np.sin(theta), 0, 0, 0, 0, 0, 1],
            [2*R*np.cos(theta), 2*R*np.sin(theta), 0, 1, 0, 0, 0, 1],
            [R*np.cos(theta2), R*np.sin(theta2), 0, 0, 1, 0, 0, 1],
            [2*R*np.cos(theta2), 2*R*np.sin(theta2), 0, 1, 1, 0, 0, 1],
        ]
        faces += [
            [4*i, 4*i+1, 4*i+2],
            [4*i+3, 4*i+1, 4*i+2],
        ]
    indices = []
    vertexData = []
    index = 0
    for face in faces:
        vertex = vertices[face[0]]
        vertexData += vertex
        vertex = vertices[face[1]]
        vertexData += vertex
        vertex = vertices[face[2]]
        vertexData += vertex

        indices += [index, index + 1, index + 2]
        index += 3  
    
    return bs.Shape(vertexData, indices)

#Funcion crear cometa
def cometaPhong(N):
    vertices = []
    faces = []
    angulo = 2 * np.pi
    n = int(N/2)
    for i in range(N+1):
        omega = i/N * angulo
        if i == N:
            omega = 0
        for j in range(n+2):
            theta = j/(N)* angulo/2
            if j != n+1:
                vertices += [
                    [np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta),
                    i/N, j/(n*5),
                    np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta)],
                ]
            else:
                vertices += [
                    [0, 0, -3,
                    i/N, 1,
                    0, 0, -1],
                ]

            if i != N:
                faces +=[
                    [(n+2)*i+j, (n+2)*i+1+j, ((n+2)*(i+1)+j)%((N+1)*(n+2))],
                    [(n+2)*i+1+j, ((n+2)*(i+1)+j)%((N+1)*(n+2)), ((n+2)*(i+1)+1+j)%((N+1)*(n+2))]
                ]
    indices = []
    vertexData = []
    index = 0
    for face in faces:
        vertex = vertices[face[0]]
        vertexData += vertex
        vertex = vertices[face[1]]
        vertexData += vertex
        vertex = vertices[face[2]]
        vertexData += vertex

        indices += [index, index + 1, index + 2]
        index += 3  
    
    return bs.Shape(vertexData, indices)

#Funcion cubo phong para estela
def cuboPhong(r, g, b):
    vertices =[
        #cara 1
        0.5, 0.5, -0.5, r, g, b, 0, 0, -1,
        0.5, -0.5, -0.5, r, g, b, 0, 0, -1,
        -0.5, 0.5, -0.5, r, g, b, 0, 0, -1,
        -0.5, -0.5, -0.5, r, g, b, 0, 0, -1,
        #cara 2
        0.5, 0.5, 0.5, r, g, b, 0, 0, 1,
        0.5, -0.5, 0.5, r, g, b, 0, 0, 1,
        -0.5, 0.5, 0.5, r, g, b, 0, 0, 1,
        -0.5, -0.5, 0.5, r, g, b, 0, 0, 1,
        #cara 3
        0.5, -0.5, 0.5, r, g, b, 0, -1, 0,
        0.5, -0.5, -0.5, r, g, b, 0, -1, 0,
        -0.5, -0.5, 0.5, r, g, b, 0, -1, 0,
        -0.5, -0.5, -0.5, r, g, b, 0, -1, 0,
        #cara 4
        0.5, 0.5, 0.5, r, g, b, 0, 1, 0,
        0.5, 0.5, -0.5, r, g, b, 0, 1, 0,
        -0.5, 0.5, 0.5, r, g, b, 0, 1, 0,
        -0.5, 0.5, -0.5, r, g, b, 0, 1, 0,
        #cara 5
        -0.5, 0.5, 0.5, r, g, b, -1, 0, 0,
        -0.5, 0.5, -0.5, r, g, b, -1, 0, 0,
        -0.5, -0.5, 0.5, r, g, b, -1, 0, 0,
        -0.5, -0.5, -0.5, r, g, b, -1, 0, 0,
        #cara 6
        0.5, 0.5, 0.5, r, g, b, 1, 0, 0,
        0.5, 0.5, -0.5, r, g, b, 1, 0, 0,
        0.5, -0.5, 0.5, r, g, b, 1, 0, 0,
        0.5, -0.5, -0.5, r, g, b, 1, 0, 0
    ]
    indices = [
        #cara1
        0, 1, 2,
        1, 2, 3,
        #cara2
        4, 5, 6,
        5, 6, 7,
        #cara3
        8, 9, 10,
        9, 10, 11,
        #cara4
        12, 13, 14,
        13, 14, 15,
        #cara5
        16, 17, 18,
        17, 18, 19,
        #cara6
        20, 21, 22,
        21, 22, 23
    ]
    return bs.Shape(vertices, indices)

#curvas de bezier
def bezier(P0, P1, P2, P3):
    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)

    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    
    return np.matmul(G, Mb)

def evaluarCurva(Matriz, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = np.array([[1, ts[i], ts[i]**2, ts[i]**3]]).T
        curve[i, 0:3] = np.matmul(Matriz, T).T
        
    return curve

#Creamos las curvas para las naves
#Curvas tie (destroyer)
tie1_1 = bezier(
    np.array([[-0.03, 0, 0]]).T, 
    np.array([[-0.03, 0.03*4/3, 0]]).T, 
    np.array([[0.03, 0.03*4/3, 0]]).T, 
    np.array([[0.03, 0, 0]]).T
    )
tie1_2 = bezier(
    np.array([[0.03, 0, 0]]).T, 
    np.array([[0.03, -0.03*4/3, 0]]).T, 
    np.array([[-0.03, -0.03*4/3, 0]]).T, 
    np.array([[-0.03, 0., 0]]).T
    )
tie1Curve1 = evaluarCurva(tie1_1, 150)
tie1Curve2 = evaluarCurva(tie1_2, 150)
tie1Curve = np.concatenate((tie1Curve1, tie1Curve2), axis = 0)
destroyer_1 = bezier(
    np.array([[4, 2, 1]]).T,
    np.array([[4, 64/126, 1]]).T,
    np.array([[4, -64/126, 1]]).T,
    np.array([[4, -2, 1]]).T
)
destroyer_2 = bezier(
    np.array([[4, -2, 1]]).T,
    np.array([[4, -2-8/3, 1]]).T,
    np.array([[-4, -2-8/3, 1]]).T,
    np.array([[-4, -2, 1]]).T
)
destroyer_3 = bezier(
    np.array([[-4, -2, 1]]).T,
    np.array([[-4, -64/126, 1]]).T,
    np.array([[-4, 64/126, 1]]).T,
    np.array([[-4, 2, 1]]).T
)
destroyer_4 = bezier(
    np.array([[-4, 2, 1]]).T,
    np.array([[-4, 2+8/3, 1]]).T,
    np.array([[4, 2+8/3, 1]]).T,
    np.array([[4, 2, 1]]).T
)
destroyerCurve1 = evaluarCurva(destroyer_1, 1600)
destroyerCurve2 = evaluarCurva(destroyer_2, 4800)
destroyerCurve3 = evaluarCurva(destroyer_3, 1600)
destroyerCurve4 = evaluarCurva(destroyer_4, 4800)
destroyerCurve = np.concatenate((destroyerCurve1, destroyerCurve2, destroyerCurve3, destroyerCurve4), axis = 0)

# Curvas convoy tierra
kontos_1 = bezier(
    np.array([[-0.229, 0, 0]]).T, 
    np.array([[-0.229, -0.229*4/3, 0]]).T, 
    np.array([[0.229, -0.229*4/3, 0]]).T, 
    np.array([[0.229, 0, 0]]).T
    )
kontos_2 = bezier(
    np.array([[0.229, 0, 0]]).T, 
    np.array([[0.229, 0.229*4/3, 0]]).T, 
    np.array([[-0.229, 0.229*4/3, 0]]).T, 
    np.array([[-0.229, 0., 0]]).T
    )
kontosCurve1 = evaluarCurva(kontos_1, 1000)
kontosCurve2 = evaluarCurva(kontos_2, 1000)
kontosCurve = np.concatenate((kontosCurve1, kontosCurve2), axis = 0)

#Curvas convoy 2
convoy2_1 = bezier(
    np.array([[-1.1, 0, 0]]).T, 
    np.array([[-1.1, -1.1*4/3, 0]]).T, 
    np.array([[1.1, -1.1*4/3, 0]]).T, 
    np.array([[1.1, 0, 0]]).T
    )
convoy2_2 = bezier(
    np.array([[1.1, 0, 0]]).T, 
    np.array([[1.1, 1.1*4/3, 0]]).T, 
    np.array([[-1.1, 1.1*4/3, 0]]).T, 
    np.array([[-1.1, 0., 0]]).T
    )
convoy2Curve1 = evaluarCurva(convoy2_1, 1000)
convoy2Curve2 = evaluarCurva(convoy2_2, 1000)
convoy2Curve = np.concatenate((convoy2Curve1, convoy2Curve2), axis = 0)

#Curvas convoy 3
convoy3_recta1 = bezier(
    np.array([[-2, -3, -1]]).T,
    np.array([[-64/126, -3, -1]]).T,
    np.array([[64/126, -3, -1]]).T,
    np.array([[2, -3, -1]]).T 
)
convoy3_curva1 = bezier(
    np.array([[2, -3, -1]]).T,
    np.array([[2+0.55, -3, -1]]).T,
    np.array([[3, -2-0.55, -1]]).T,
    np.array([[3, -2, -1]]).T 
)
convoy3_recta2 = bezier(
    np.array([[3, -2, -1]]).T,
    np.array([[3, -64/126, -1]]).T,
    np.array([[3, 64/126, -1]]).T,
    np.array([[3, 2, -1]]).T 
)
convoy3_curva2 = bezier(
    np.array([[3, 2, -1]]).T,
    np.array([[3, 2+0.55, -1]]).T,
    np.array([[2+0.55, 3, -1]]).T,
    np.array([[2, 3, -1]]).T 
)
convoy3_recta3 = bezier(
    np.array([[2, 3, -1]]).T,
    np.array([[64/126, 3, -1]]).T,
    np.array([[-64/126, 3, -1]]).T,
    np.array([[-2, 3, -1]]).T 
)
convoy3_curva3 = bezier(
    np.array([[-2, 3, -1]]).T,
    np.array([[-2-0.55, 3, -1]]).T,
    np.array([[-3, 2+0.55, -1]]).T,
    np.array([[-3, 2, -1]]).T 
)
convoy3_recta4 = bezier(
    np.array([[-3, 2, -1]]).T,
    np.array([[-3, 64/126, -1]]).T,
    np.array([[-3, -64/126, -1]]).T,
    np.array([[-3, -2, -1]]).T 
)
convoy3_curva4 = bezier(
    np.array([[-3, -2, -1]]).T,
    np.array([[-3, -2-0.55, -1]]).T,
    np.array([[-2-0.55, -3, -1]]).T,
    np.array([[-2, -3, -1]]).T 
)
convoy3Curve1 = evaluarCurva(convoy3_recta1, 4000)
convoy3Curve2 = evaluarCurva(convoy3_curva1, 1500)
convoy3Curve3 = evaluarCurva(convoy3_recta2, 4000)
convoy3Curve4 = evaluarCurva(convoy3_curva2, 1500)
convoy3Curve5 = evaluarCurva(convoy3_recta3, 4000)
convoy3Curve6 = evaluarCurva(convoy3_curva3, 1500)
convoy3Curve7 = evaluarCurva(convoy3_recta4, 4000)
convoy3Curve8 = evaluarCurva(convoy3_curva4, 1500)
convoy3Curve = np.concatenate((convoy3Curve1, convoy3Curve2, convoy3Curve3, convoy3Curve4, convoy3Curve5, convoy3Curve6, convoy3Curve7, convoy3Curve8), axis = 0)
#nave2
convoy3_1_recta1 = bezier(
    np.array([[-2, -3.08, -1]]).T,
    np.array([[-64/126, -3.08, -1]]).T,
    np.array([[64/126, -3.08, -1]]).T,
    np.array([[2, -3.08, -1]]).T 
)
convoy3_1_curva1 = bezier(
    np.array([[2, -3.08, -1]]).T,
    np.array([[2+0.594, -3.08, -1]]).T,
    np.array([[3.08, -2-0.594, -1]]).T,
    np.array([[3.08, -2, -1]]).T 
)
convoy3_1_recta2 = bezier(
    np.array([[3.08, -2, -1]]).T,
    np.array([[3.08, -64/126, -1]]).T,
    np.array([[3.08, 64/126, -1]]).T,
    np.array([[3.08, 2, -1]]).T 
)
convoy3_1_curva2 = bezier(
    np.array([[3.08, 2, -1]]).T,
    np.array([[3.08, 2+0.594, -1]]).T,
    np.array([[2+0.594, 3.08, -1]]).T,
    np.array([[2, 3.08, -1]]).T 
)
convoy3_1_recta3 = bezier(
    np.array([[2, 3.08, -1]]).T,
    np.array([[64/126, 3.08, -1]]).T,
    np.array([[-64/126, 3.08, -1]]).T,
    np.array([[-2, 3.08, -1]]).T 
)
convoy3_1_curva3 = bezier(
    np.array([[-2, 3.08, -1]]).T,
    np.array([[-2-0.594, 3.08, -1]]).T,
    np.array([[-3.08, 2+0.594, -1]]).T,
    np.array([[-3.08, 2, -1]]).T 
)
convoy3_1_recta4 = bezier(
    np.array([[-3.08, 2, -1]]).T,
    np.array([[-3.08, 64/126, -1]]).T,
    np.array([[-3.08, -64/126, -1]]).T,
    np.array([[-3.08, -2, -1]]).T 
)
convoy3_1_curva4 = bezier(
    np.array([[-3.08, -2, -1]]).T,
    np.array([[-3.08, -2-0.594, -1]]).T,
    np.array([[-2-0.594, -3.08, -1]]).T,
    np.array([[-2, -3.08, -1]]).T 
)
convoy3_1Curve1 = evaluarCurva(convoy3_1_recta1, 4000)
convoy3_1Curve2 = evaluarCurva(convoy3_1_curva1, 1500)
convoy3_1Curve3 = evaluarCurva(convoy3_1_recta2, 4000)
convoy3_1Curve4 = evaluarCurva(convoy3_1_curva2, 1500)
convoy3_1Curve5 = evaluarCurva(convoy3_1_recta3, 4000)
convoy3_1Curve6 = evaluarCurva(convoy3_1_curva3, 1500)
convoy3_1Curve7 = evaluarCurva(convoy3_1_recta4, 4000)
convoy3_1Curve8 = evaluarCurva(convoy3_1_curva4, 1500)
convoy3_1Curve = np.concatenate((convoy3_1Curve1, convoy3_1Curve2, convoy3_1Curve3, convoy3_1Curve4, convoy3_1Curve5, convoy3_1Curve6, convoy3_1Curve7, convoy3_1Curve8), axis = 0)
#nave 3
convoy3_2_recta1 = bezier(
    np.array([[-2, -2.93, -1]]).T,
    np.array([[-64/126, -2.93, -1]]).T,
    np.array([[64/126, -2.93, -1]]).T,
    np.array([[2, -2.93, -1]]).T 
)
convoy3_2_curva1 = bezier(
    np.array([[2, -2.93, -1]]).T,
    np.array([[2+0.5115, -2.93, -1]]).T,
    np.array([[2.93, -2-0.5115, -1]]).T,
    np.array([[2.93, -2, -1]]).T 
)
convoy3_2_recta2 = bezier(
    np.array([[2.93, -2, -1]]).T,
    np.array([[2.93, -64/126, -1]]).T,
    np.array([[2.93, 64/126, -1]]).T,
    np.array([[2.93, 2, -1]]).T 
)
convoy3_2_curva2 = bezier(
    np.array([[2.93, 2, -1]]).T,
    np.array([[2.93, 2+0.5115, -1]]).T,
    np.array([[2+0.5115, 2.93, -1]]).T,
    np.array([[2, 2.93, -1]]).T 
)
convoy3_2_recta3 = bezier(
    np.array([[2, 2.93, -1]]).T,
    np.array([[64/126, 2.93, -1]]).T,
    np.array([[-64/126, 2.93, -1]]).T,
    np.array([[-2, 2.93, -1]]).T 
)
convoy3_2_curva3 = bezier(
    np.array([[-2, 2.93, -1]]).T,
    np.array([[-2-0.5115, 2.93, -1]]).T,
    np.array([[-2.93, 2+0.5115, -1]]).T,
    np.array([[-2.93, 2, -1]]).T 
)
convoy3_2_recta4 = bezier(
    np.array([[-2.93, 2, -1]]).T,
    np.array([[-2.93, 64/126, -1]]).T,
    np.array([[-2.93, -64/126, -1]]).T,
    np.array([[-2.93, -2, -1]]).T 
)
convoy3_2_curva4 = bezier(
    np.array([[-2.93, -2, -1]]).T,
    np.array([[-2.93, -2-0.5115, -1]]).T,
    np.array([[-2-0.5115, -2.93, -1]]).T,
    np.array([[-2, -2.93, -1]]).T 
)
convoy3_2Curve1 = evaluarCurva(convoy3_2_recta1, 4000)
convoy3_2Curve2 = evaluarCurva(convoy3_2_curva1, 1500)
convoy3_2Curve3 = evaluarCurva(convoy3_2_recta2, 4000)
convoy3_2Curve4 = evaluarCurva(convoy3_2_curva2, 1500)
convoy3_2Curve5 = evaluarCurva(convoy3_2_recta3, 4000)
convoy3_2Curve6 = evaluarCurva(convoy3_2_curva3, 1500)
convoy3_2Curve7 = evaluarCurva(convoy3_2_recta4, 4000)
convoy3_2Curve8 = evaluarCurva(convoy3_2_curva4, 1500)
convoy3_2Curve = np.concatenate((convoy3_2Curve1, convoy3_2Curve2, convoy3_2Curve3, convoy3_2Curve4, convoy3_2Curve5, convoy3_2Curve6, convoy3_2Curve7, convoy3_2Curve8), axis = 0)
#Crear sistema solar scene graph
def createSystem(pipeline):
    mercurioShape = createGPUShape(esferaPhong(100), pipeline)
    mercurioShape.texture = es.textureSimpleSetup(getAssetPath("mercurio.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    venusShape = createGPUShape(esferaPhong(100), pipeline)
    venusShape.texture = es.textureSimpleSetup(getAssetPath("venus.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    tierraShape = createGPUShape(esferaPhong(100), pipeline)
    tierraShape.texture = es.textureSimpleSetup(getAssetPath("tierra.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    lunaShape = createGPUShape(esferaPhong(100), pipeline)
    lunaShape.texture = es.textureSimpleSetup(getAssetPath("luna.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    marteShape = createGPUShape(esferaPhong(100), pipeline)
    marteShape.texture = es.textureSimpleSetup(getAssetPath("marte.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    jupiterShape = createGPUShape(esferaPhong(100), pipeline)
    jupiterShape.texture = es.textureSimpleSetup(getAssetPath("jupiter.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    saturnoShape = createGPUShape(esferaPhong(100), pipeline)
    saturnoShape.texture = es.textureSimpleSetup(getAssetPath("saturno.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    anilloShape = createGPUShape(crearAnillo(100, 1), pipeline)
    anilloShape.texture = es.textureSimpleSetup(getAssetPath("anillo saturno.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    uranoShape = createGPUShape(esferaPhong(100), pipeline)
    uranoShape.texture = es.textureSimpleSetup(getAssetPath("urano.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    neptunoShape = createGPUShape(esferaPhong(100), pipeline)
    neptunoShape.texture = es.textureSimpleSetup(getAssetPath("neptuno.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    plutonShape = createGPUShape(esferaPhong(100), pipeline)
    plutonShape.texture = es.textureSimpleSetup(getAssetPath("pluton.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    cometaShape = createGPUShape(cometaPhong(100), pipeline)
    cometaShape.texture = es.textureSimpleSetup(getAssetPath("cometa.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    mercurioNode = sg.SceneGraphNode("mercurioNode")
    mercurioNode.transform = tr.uniformScale(0.1)
    mercurioNode.childs += [mercurioShape]
    mercurioTranslation = sg.SceneGraphNode("mercurioTranslation")
    mercurioTranslation.transform = tr.translate(-1.5, 0, 0)
    mercurioTranslation.childs += [mercurioNode]

    venusNode = sg.SceneGraphNode("venusNode")
    venusNode.transform = tr.uniformScale(0.2)
    venusNode.childs += [venusShape]
    venusTranslation = sg.SceneGraphNode("venusTranslation")
    venusTranslation.transform = tr.translate(-2.2, 0, 0)
    venusTranslation.childs += [venusNode]

    tierraNode = sg.SceneGraphNode("tierraNode")
    tierraNode.transform = tr.uniformScale(0.22)
    tierraNode.childs += [tierraShape]
    lunaNode = sg.SceneGraphNode("lunaNode")
    lunaNode.transform = tr.uniformScale(0.08)
    lunaNode.childs += [lunaShape]
    lunaTranslation = sg.SceneGraphNode("lunaTranslation")
    lunaTranslation.transform = tr.translate(0.5, 0, 0)
    lunaTranslation.childs += [lunaNode]
    sistemaTierraLuna = sg.SceneGraphNode("sistemaTierraLuna")
    sistemaTierraLuna.transform = tr.translate(-3, 0, 0)
    sistemaTierraLuna.childs += [tierraNode, lunaTranslation]

    marteNode = sg.SceneGraphNode("marteNode")
    marteNode.transform = tr.uniformScale(0.2)
    marteNode.childs += [marteShape]
    marteTranslation = sg.SceneGraphNode("marteTranslation")
    marteTranslation.transform = tr.translate(-4.5, 0, 0)
    marteTranslation.childs += [marteNode]

    jupiterNode = sg.SceneGraphNode("jupiterNode")
    jupiterNode.transform = tr.uniformScale(0.52)
    jupiterNode.childs += [jupiterShape]
    jupiterTranslation = sg.SceneGraphNode("jupiterTranslation")
    jupiterTranslation.transform = tr.translate(-13, 0, 0)
    jupiterTranslation.childs += [jupiterNode]

    saturnoNode =sg.SceneGraphNode("saturnoNode")
    saturnoNode.transform = tr.uniformScale(0.50)
    saturnoNode.childs += [saturnoShape]
    anilloNode = sg.SceneGraphNode("anilloNode")
    anilloNode.transform = tr.uniformScale(0.54)
    anilloNode.childs += [anilloShape]
    sistemaSaturno = sg.SceneGraphNode("sistemaSaturno")
    sistemaSaturno.transform = tr.translate(-23.8, 0, 0)
    sistemaSaturno.childs += [
        saturnoNode, 
        anilloNode
    ]

    uranoNode = sg.SceneGraphNode("uranoNode")
    uranoNode.transform = tr.uniformScale(0.4)
    uranoNode.childs += [uranoShape]
    uranoTranslation = sg.SceneGraphNode("uranoTranslation")
    uranoTranslation.transform = tr.translate(-48.7, 0, 0)
    uranoTranslation.childs += [uranoNode]

    neptunoNode = sg.SceneGraphNode("neptunoNode")
    neptunoNode.transform = tr.uniformScale(0.4)
    neptunoNode.childs += [neptunoShape]
    neptunoTranslation = sg.SceneGraphNode("neptunoTranslation")
    neptunoTranslation.transform = tr.translate(-75.1, 0, 0)
    neptunoTranslation.childs += [neptunoNode]

    plutonNode = sg.SceneGraphNode("plutonNode")
    plutonNode.transform = tr.uniformScale(0.08)
    plutonNode.childs += [plutonShape]
    plutonTranslation = sg.SceneGraphNode("plutonTranslation")
    plutonTranslation.transform = tr.translate(-98.6, 0, 0)
    plutonTranslation.childs += [plutonNode]

    cometaNode =sg.SceneGraphNode("cometaNode")
    cometaNode.transform = tr.uniformScale(0.01)
    cometaNode.childs += [cometaShape]
    cometaTranslation =sg.SceneGraphNode("cometaTranslation")
    cometaTranslation.transform = tr.identity()
    cometaTranslation.childs += [cometaNode]

    systemNode = sg.SceneGraphNode("sistemaSolar")
    systemNode.childs +=[
        mercurioTranslation,
        venusTranslation,
        sistemaTierraLuna,
        marteTranslation,
        jupiterTranslation,
        sistemaSaturno,
        uranoTranslation,
        neptunoTranslation,
        plutonTranslation,
        cometaTranslation
    ]

    return systemNode

#Crear fondo de estrellas
def createStars(pipeline):
    estrellasShape = createGPUShape(crearEsfera(25, 1, 1, 0), pipeline)
    sceneNode = sg.SceneGraphNode("fondo")
    contador = 0
    estrellas = []
    while contador < 500:
        distancia = randint(110, 300)
        theta = random() * np.pi
        omega = (random() - 1)*2 * np.pi
        name = str(contador)
        estrellas += [sg.SceneGraphNode(name)]
        estrellas[contador].transform = tr.matmul([
            tr.translate(distancia*np.sin(theta)*np.cos(omega), distancia*np.sin(theta)*np.sin(omega), distancia*np.cos(theta)),
            tr.uniformScale(random())
        ])
        estrellas[contador].childs += [estrellasShape]
        sceneNode.childs += [estrellas[contador]]
        contador += 1
    

    return sceneNode

#Crear batalla de naves scene graph
def createFighter(pipeline):
    corvetteShape = createGPUShape(bs.readOFF(getAssetPath('Costum_Corvette.off'), (0.6 , 0.3 ,0.6)), pipeline)
    fromSPShape = createGPUShape(bs.readOFF(getAssetPath('FromSP.off'), (0.3 , 0.9 ,0.3)), pipeline)
    destroyerShape = createGPUShape(bs.readOFF(getAssetPath('Imperial_star_destroyer.off'), (0.5 , 0.5 ,0.5)), pipeline)
    kontosShape = createGPUShape(bs.readOFF(getAssetPath('Kontos.off'), (0.3 , 0.5 ,0.3)), pipeline)
    nabooFighterShape = createGPUShape(bs.readOFF(getAssetPath('NabooFighter.off'), (0.9 , 0.3 ,0.3)), pipeline)
    tieUVShape = createGPUShape(bs.readOFF(getAssetPath('tie_UV.off'), (0.1 , 0.1 ,0.1)), pipeline)
    triFighterShape = createGPUShape(bs.readOFF(getAssetPath('Tri_Fighter.off'), (0.3 , 0.3 ,0.9)), pipeline)
    xWingShape = createGPUShape(bs.readOFF(getAssetPath('XJ5 X-wing starfighter.off'), (0.9 , 0.9 ,0.9)), pipeline)
    estelaBlueShape = createGPUShape(cuboPhong(0,0,1), pipeline)
    estelaGreenShape = createGPUShape(cuboPhong(0,1,0), pipeline)
    estelaRedShape = createGPUShape(cuboPhong(1,0,0), pipeline)

    #Crear nodos por cada shape, con la escala de las naves

    corvetteNode = sg.SceneGraphNode("corvetteNode")
    corvetteNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.05)
    ])
    corvetteNode.childs += [corvetteShape]

    fromSPNode = sg.SceneGraphNode("fromSPNode")
    fromSPNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.05)
    ])
    fromSPNode.childs += [fromSPShape]

    destroyerNode = sg.SceneGraphNode("destroyerNode")
    destroyerNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.05)
    ])
    destroyerNode.childs += [destroyerShape]

    tieUVNode = sg.SceneGraphNode("tieUVNode")
    tieUVNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.005)
    ])
    tieUVNode.childs += [tieUVShape]
    
    kontosNode = sg.SceneGraphNode("kontosNode")
    kontosNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.05)
    ])
    kontosNode.childs += [kontosShape]

    nabooNode = sg.SceneGraphNode("nabooNode")
    nabooNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.005)
    ])
    nabooNode.childs += [nabooFighterShape]

    triNode = sg.SceneGraphNode("triNode")
    triNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.05)
    ])
    triNode.childs += [triFighterShape]    

    xWingNode = sg.SceneGraphNode("xWingNode")
    xWingNode.transform = tr.matmul([
        tr.rotationX(np.pi/2),
        tr.uniformScale(0.005)
    ])
    xWingNode.childs += [xWingShape]

    estelaBlueNode =sg.SceneGraphNode("estelaBlueNode")
    estelaBlueNode.transform = tr.identity()
    estelaBlueNode.childs += [estelaBlueShape]

    estelaGreenNode =sg.SceneGraphNode("estelaGreenNode")
    estelaGreenNode.transform = tr.identity()
    estelaGreenNode.childs += [estelaGreenShape]

    estelaRedNode =sg.SceneGraphNode("estelaRedNode")
    estelaRedNode.transform = tr.identity()
    estelaRedNode.childs += [estelaRedShape]

    #convoy tierra
    earthconvoy = sg.SceneGraphNode("earthConvoy")
    earthconvoy.transform = tr.translate(0.25, 0, 0)
    earthconvoy.childs += [kontosNode]
    earthOrbit = sg.SceneGraphNode("earthOrbit")
    earthOrbit.transform = tr.identity()
    earthOrbit.childs += [earthconvoy]
    earthMove = sg.SceneGraphNode("earthMove")
    earthMove.transform =tr.identity()
    earthMove.childs += [earthOrbit] 
    estelaKontos = []
    for i in range(25):
        estelaKontos += [sg.SceneGraphNode("estelaKontos"+str(i))]
        estelaKontos[i].transform = tr.uniformScale(0.05)
        estelaKontos[i].childs += [estelaRedNode]
        earthOrbit.childs += [estelaKontos[i]]

    #Convoy 1
    #Tie
    tieUVTraslation = sg.SceneGraphNode("tieUVTraslation")
    tieUVTraslation.transform = tr.translate(0.03, 0, 0)
    tieUVTraslation.childs += [tieUVNode]
    #crear destructor con un Tie rotando al rededor
    destroyer1 = sg.SceneGraphNode("destroyer1")
    destroyer1.transform = tr.identity()
    destroyer1.childs += [destroyerNode, tieUVTraslation]
    estelaTie = []
    for i in range(25):
        estelaTie += [sg.SceneGraphNode("estelaTie"+str(i))]
        estelaTie[i].transform = tr.uniformScale(0.05)
        estelaTie[i].childs += [estelaBlueNode]
        destroyer1.childs += [estelaTie[i]]
    estelaDestroyer = []
    convoy = sg.SceneGraphNode("convoy")
    convoy.transform = tr.identity()
    convoy.childs += [destroyer1]
    for i in range(25):
        estelaDestroyer += [sg.SceneGraphNode("estelaDestroyer"+str(i))]
        estelaDestroyer[i].transform = tr.uniformScale(0.05)
        estelaDestroyer[i].childs += [estelaGreenNode]
        convoy.childs += [estelaDestroyer[i]]
    
    #convoy 2
    corvette1 = sg.SceneGraphNode("corvette1")
    corvette1.transform = tr.identity()
    corvette1.childs += [corvetteNode]

    corvette2 = sg.SceneGraphNode("corvette2")
    corvette2.transform = tr.translate(-0.04, 0.04, -0.02)
    corvette2.childs += [corvetteNode]

    corvette3 = sg.SceneGraphNode("corvette3")
    corvette3.transform = tr.translate(0.04, 0.04, -0.02)
    corvette3.childs += [corvetteNode]

    convoy2 = sg.SceneGraphNode("convoy2")
    convoy2.transform = tr.identity()
    convoy2.childs += [corvette1, corvette2, corvette3]

    #convoy 3

    fromSP = sg.SceneGraphNode("fromSP")
    fromSP.transform = tr.identity()
    fromSP.childs += [fromSPNode]
    fromSPtranslate = sg.SceneGraphNode("fromSPtranslate")
    fromSPtranslate.transform = tr.translate(0, -0.04, 0)
    fromSPtranslate.childs += [fromSP]
    

    tri = sg.SceneGraphNode("tri")
    tri.transform = tr.identity()
    tri.childs += [triNode]
    triTranslate = sg.SceneGraphNode("triTranslate")
    triTranslate.transform = tr.translate(0, 0.04, 0)
    triTranslate.childs += [tri]

    naboo = sg.SceneGraphNode("naboo")
    naboo.transform = tr.identity()
    naboo.childs += [nabooNode]
    nabooTranslate = sg.SceneGraphNode("nabooTranslate")
    nabooTranslate.transform = tr.translate(0.04, 0, 0)
    nabooTranslate.childs += [naboo]

    convoy3 = sg.SceneGraphNode("convoy3")
    convoy3.transform = tr.identity()
    convoy3.childs += [fromSPtranslate, triTranslate, nabooTranslate]

    usuario =sg.SceneGraphNode("usuario")
    usuario.transform = tr.identity()
    usuario.childs += [xWingNode]

    sceneNode = sg.SceneGraphNode("naves")
    sceneNode.childs += [usuario, convoy, earthMove, convoy2, convoy3]

    #estelas
    #convoy 2
    estelacorvette1 = []
    for i in range(25):
        estelacorvette1 += [sg.SceneGraphNode("estelacorvette1"+str(i))]
        estelacorvette1[i].transform = tr.uniformScale(0.05)
        estelacorvette1[i].childs += [estelaBlueNode]
        sceneNode.childs += [estelacorvette1[i]]
    estelacorvette2 = []
    for i in range(25):
        estelacorvette2 += [sg.SceneGraphNode("estelacorvette2"+str(i))]
        estelacorvette2[i].transform = tr.uniformScale(0.05)
        estelacorvette2[i].childs += [estelaRedNode]
        sceneNode.childs += [estelacorvette2[i]]
    estelacorvette3 = []
    for i in range(25):
        estelacorvette3 += [sg.SceneGraphNode("estelacorvette3"+str(i))]
        estelacorvette3[i].transform = tr.uniformScale(0.05)
        estelacorvette3[i].childs += [estelaGreenNode]
        sceneNode.childs += [estelacorvette3[i]]


    return sceneNode

#Funcion que maneja el uso de teclas
def on_key(window, key, scancode, action, mods):
    #Si no se presiona una tecla no hace nada
    if action != glfw.PRESS:
        return
    
    global controller

    #Cierra la aplicación con escape
    if not controller.closed:
        if key == glfw.KEY_0:
            controller.camType = 0
        if key == glfw.KEY_1:
            controller.camType = 1
        if key == glfw.KEY_2:
            controller.camType = 2
        if key == glfw.KEY_3:
            controller.camType = 3
        if key == glfw.KEY_4:
            controller.camType = 4
        if key == glfw.KEY_5:
            controller.camType = 5
        #Backflip
        if key == glfw.KEY_E and not controller.backfliped:
            controller.backfliped = True
            controller.backflip += 0.001
        if key == glfw.KEY_Q:
            controller.backfliped = True
    
        if key == glfw.KEY_ESCAPE:
            controller.closed = True


def main():

    #inicializa glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)
    
    width = 1500
    height = 1000

    window = glfw.create_window(width, height, "Tarea 2", None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)
    
    glfw.make_context_current(window)

    #Conecta la funcion callback 'on_key' para manejar los eventos del teclado
    glfw.set_key_callback(window, on_key)

    #Ensamblando el shader program
    MVPpipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimplePhongShaderProgram()
    textpipeline = ls.SimpleTexturePhongShaderProgram()


    #mandar a OpenGL a usar el shader program
    glUseProgram(pipeline.shaderProgram)

    #setiando el color de fondo
    glClearColor(0.05, 0.05, 0.15, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    #Proyeccion
    proyeccion = tr.perspective(60, float(1500)/float(1000), 0.001, 200)

    #Crear shapes en la GPU memory
    gpuAxis = createGPUShape(bs.createAxis(7), MVPpipeline)
    sistemaSolar = createSystem(textpipeline)
    sol = createGPUShape(crearEsfera(100, 1, 1, 0), MVPpipeline)
    fondo = createStars(MVPpipeline)
    figther = createFighter(pipeline)


    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 1.0, 0.7, 0.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 0, 0, 0)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

    glUseProgram(textpipeline.shaderProgram)

    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "Kd"), 1.0, 0.7, 0.0)
    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "Ks"), 0, 0, 0)

    glUniform3f(glGetUniformLocation(textpipeline.shaderProgram, "lightPosition"), 0, 0, 0)
    
    glUniform1ui(glGetUniformLocation(textpipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(textpipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(textpipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(textpipeline.shaderProgram, "quadraticAttenuation"), 0.01)
    

    #cuenta los pasos en el programa
    step = 0

    while not glfw.window_should_close(window):

        #usar GLFW para chequear input events
        glfw.poll_events()

        #contador de tiempo
        time = 10 * glfw.get_time()
        step +=1

        #iputs nave usuario
        global controller
        if controller.camType == 0:
            if glfw.get_key(window,glfw.KEY_SPACE) == glfw.PRESS:
                if controller.aceleartion <= 10:
                    controller.aceleartion += 0.05
                if glfw.get_key(window,glfw.KEY_D) == glfw.PRESS:
                    controller.derecha()
                elif glfw.get_key(window,glfw.KEY_A) == glfw.PRESS:
                    controller.izquierda()
                if glfw.get_key(window,glfw.KEY_W) == glfw.PRESS:
                    controller.arriba()
                elif glfw.get_key(window,glfw.KEY_S) == glfw.PRESS:
                    controller.abajo()
                #Modo turbo con left shift
                if glfw.get_key(window,glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
                    controller.position = controller.position +controller.front*0.005*controller.aceleartion
                else:
                    controller.position = controller.position +controller.front*0.0005*controller.aceleartion
        elif controller.camType == 5:
            if glfw.get_key(window,glfw.KEY_W) == glfw.PRESS:
                controller.jupiter += 0.05
                if controller.jupiter > 1:
                    controller.jupiter = 1
            elif glfw.get_key(window,glfw.KEY_S) == glfw.PRESS:
                controller.jupiter -= 0.05
                if controller.jupiter < 0:
                    controller.jupiter = 0

        #Backflip
        if controller.backfliped:
            if controller.backflip > 0:
                controller.backflip += np.pi/25
                if controller.backflip >= 2*np.pi:
                    controller.backfliped = False
                    controller.backflip = 0
            else:
                controller.backflip -= np.pi/25
                if controller.backflip <= -2*np.pi:
                    controller.backfliped = False
                    controller.backflip = 0

        #Chekea si hay que cerrar el programa
        if controller.closed:
            if controller.camType == 0:
                controller.position = controller.position +controller.front*5
                controller.close += 1
                if controller.close >= 40:
                    glfw.set_window_should_close(window, True)
            else:
                glfw.set_window_should_close(window, True)

        view = controller.camera(time, step)

        #View y proyeccion
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

        glUseProgram(textpipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(textpipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textpipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

        glUseProgram(MVPpipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(MVPpipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(MVPpipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


        #traslaciones planeta
        tierraTransform = tr.matmul([
            tr.rotationZ(time/365),
            tr.translate(-3, 0, 0)
        ])
        jupiterTransform = tr.matmul([
            tr.rotationX(-0.05*np.pi),
            tr.rotationZ(time/3942),
            tr.translate(0, 13, 0)
        ])


        #Modelos
        #Naves

        #convoy tierra
        earthMove = sg.findNode(figther, "earthMove")
        earthMove.transform = tierraTransform
        earthOrbit = sg.findNode(figther, "earthOrbit")
        earthOrbit.transform =tr.matmul([
            tr.rotationX(np.pi/10),
            
        ])
        earthConvoy = sg.findNode(figther, "earthConvoy")
        earthConvoy.transform = tr.matmul([
            tr.translate(
                kontosCurve[(step*10)%2000, 0],
                kontosCurve[(step*10)%2000, 1],
                kontosCurve[(step*10)%2000, 2],
            ),
            tr.rotationZ(np.pi*((step*10)%2000)/1000),
            tr.rotationY(-np.pi/2)
        ])
        estelaKontos = []
        for i in range(25):
            estelaKontos += [sg.findNode(figther, "estelaKontos"+str(i))]
            estelaKontos[i].transform = tr.matmul([
                tr.translate(
                    kontosCurve[(step*10 - 50- i*2)%2000, 0],
                    kontosCurve[(step*10 - 50- i*2)%2000, 1],
                    kontosCurve[(step*10 - 50- i*2)%2000, 2],
                ),
                tr.rotationZ(np.pi*((step*10)%2000)/1000),
                tr.uniformScale(0.005*(25-i/2)/25)
            ])


        #Convoy 
        destroyer1 = sg.findNode(figther, "destroyer1")
        destroyerPos = 4*step 
        tiePosition = sg.findNode(figther, "tieUVTraslation")
        tiePosition.transform = tr.matmul([
            tr.translate(
                tie1Curve[step%300, 0],
                tie1Curve[step%300, 1],
                tie1Curve[step%300, 2],
            ),
            tr.rotationZ(-np.pi*(step%300)/150)
        ])
        estelaTie = []
        for i in range(25):
            estelaTie += [sg.findNode(figther, "estelaTie"+str(i))]
            estelaTie[i].transform = tr.matmul([
                tr.translate(
                    tie1Curve[(step- 5 - i)%300, 0],
                    tie1Curve[(step- 5 - i)%300, 1],
                    tie1Curve[(step- 5 - i)%300, 2],
                ),
                tr.rotationZ(-np.pi*((step- 5 - i)%300)/150),
                tr.uniformScale(0.001*(25-i/2)/25)
            ])
        estelaDestroyer = []
        for i in range(25):
            estelaDestroyer += [sg.findNode(figther, "estelaDestroyer"+str(i))]
            if (destroyerPos-20 -i)%12800<1600:
                estelaDestroyer[i].transform = tr.matmul([
                    tr.translate(
                        destroyerCurve[(destroyerPos-20 -i)%12800, 0],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 1],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 2]
                    ),
                    tr.uniformScale(0.005*(25-i/2)/25)
                ])
                
            elif (destroyerPos-20 -i)%12800<6400:
                estelaDestroyer[i].transform = tr.matmul([
                    tr.translate(
                        destroyerCurve[(destroyerPos-20 -i)%12800, 0],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 1],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 2]
                    ),
                    tr.rotationZ(-np.pi*(((destroyerPos-20 -i)%12800)-1600)/4800),
                    tr.rotationY(-np.sin(np.pi*(((destroyerPos-20)%12800)-1600)/4800)*np.pi/4),
                    tr.uniformScale(0.005*(25-i/2)/25)
                ])
            elif (destroyerPos-20 -i)%12800<8000:
                estelaDestroyer[i].transform = tr.matmul([
                    tr.translate(
                        destroyerCurve[(destroyerPos-20 -i)%12800, 0],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 1],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 2]
                    ),
                    tr.rotationZ(-np.pi),
                    tr.uniformScale(0.005*(25-i/2)/25)
                ])
            else:
                estelaDestroyer[i].transform = tr.matmul([
                    tr.translate(
                        destroyerCurve[(destroyerPos-20 -i)%12800, 0],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 1],
                        destroyerCurve[(destroyerPos-20 -i)%12800, 2]
                    ),
                    tr.rotationZ(-np.pi -np.pi*(((destroyerPos-20 -i)%12800)-8000)/4800),
                    tr.rotationY(-np.sin(np.pi*(((destroyerPos-20)%12800)-8000)/4800)*np.pi/4),
                    tr.uniformScale(0.005*(25-i/2)/25)
                ])
        if destroyerPos%12800<1600:
            destroyer1.transform = tr.translate(
                destroyerCurve[destroyerPos%12800, 0],
                destroyerCurve[destroyerPos%12800, 1],
                destroyerCurve[destroyerPos%12800, 2]
            )
        elif destroyerPos%12800<6400:
            destroyer1.transform = tr.matmul([
                tr.translate(
                    destroyerCurve[destroyerPos%12800, 0],
                    destroyerCurve[destroyerPos%12800, 1],
                    destroyerCurve[destroyerPos%12800, 2]
                ),
                tr.rotationZ(-np.pi*((destroyerPos%12800)-1600)/4800),
                tr.rotationY(-np.sin(np.pi*((destroyerPos%12800)-1600)/4800)*np.pi/4)
            ])
        elif destroyerPos%12800<8000:
            destroyer1.transform = tr.matmul([
                tr.translate(
                    destroyerCurve[destroyerPos%12800, 0],
                    destroyerCurve[destroyerPos%12800, 1],
                    destroyerCurve[destroyerPos%12800, 2]
                ),
                tr.rotationZ(-np.pi)
            ])
        else:
            destroyer1.transform = tr.matmul([
                tr.translate(
                    destroyerCurve[destroyerPos%12800, 0],
                    destroyerCurve[destroyerPos%12800, 1],
                    destroyerCurve[destroyerPos%12800, 2]
                ),
                tr.rotationZ(-np.pi -np.pi*((destroyerPos%12800)-8000)/4800),
                tr.rotationY(-np.sin(np.pi*((destroyerPos%12800)-8000)/4800)*np.pi/4)
            ])

        #Convoy2
        convoy2 = sg.findNode(figther, "convoy2")
        convoy2.transform = tr.matmul([
            tr.translate(
                convoy2Curve[step%2000, 0],
                convoy2Curve[step%2000, 1],
                convoy2Curve[step%2000, 2]
            ),
            tr.rotationZ(np.pi*(step%2000)/1000)
        ])
        estelacorvette1 = []
        for i in range(25):
            estelacorvette1 += [sg.findNode(figther, "estelacorvette1"+str(i))]
            estelacorvette1[i].transform = tr.matmul([
                tr.translate(
                    convoy2Curve[(step- 10 - i)%2000, 0],
                    convoy2Curve[(step- 10 - i)%2000, 1],
                    convoy2Curve[(step- 10 - i)%2000, 2],
                ),
                tr.rotationZ(-np.pi*((step- 10 - i)%2000)/1000),
                tr.rotationY(np.pi/4),
                tr.uniformScale(0.005*(25-i/2)/25)
            ])
        #No cumplen con lo esperado, crear sus propias rutas para cada nave
        """ estelacorvette2 = []
        for i in range(25):
            estelacorvette2 += [sg.findNode(figther, "estelacorvette2"+str(i))]
            estelacorvette2[i].transform = tr.matmul([
                tr.translate(
                    convoy2Curve[(step- 10 - i)%2000, 0]-0.04,
                    convoy2Curve[(step- 10 - i)%2000, 1]+0.04,
                    convoy2Curve[(step- 10 - i)%2000, 2]-0.02,
                ),
                tr.rotationZ(-np.pi*((step- 10 - i)%2000)/1000),
                tr.rotationY(np.pi/4),
                tr.uniformScale(0.005*(25-i/2)/25)
            ])
        estelacorvette3 = []
        for i in range(25):
            estelacorvette3 += [sg.findNode(figther, "estelacorvette3"+str(i))]
            estelacorvette3[i].transform = tr.matmul([
                tr.translate(
                    convoy2Curve[(step- 10 - i)%2000, 0]+0.04,
                    convoy2Curve[(step- 10 - i)%2000, 1]+0.04,
                    convoy2Curve[(step- 10 - i)%2000, 2]-0.02,
                ),
                tr.rotationZ(-np.pi*((step- 10 - i)%2000)/1000),
                tr.rotationY(np.pi/4),
                tr.uniformScale(0.005*(25-i/2)/25)
            ]) """
        corvette = sg.findNode(figther, "corvetteNode")
        corvette.transform =  tr.matmul([
            tr.rotationY(np.pi/4),
            tr.rotationX(np.pi/2),
            tr.uniformScale(0.05)
        ])

        #Convoy3
        convoy3Pos = step*10
        if convoy3Pos%22000 < 4000:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi/2)
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi/2)
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(np.pi/2)
            ])
            print("recta1")
        elif convoy3Pos%22000 < 5500:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi/2 + np.pi*(convoy3Pos%22000-4000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-4000)/1500))
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi/2 + np.pi*(convoy3Pos%22000-4000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-4000)/1500))
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(np.pi/2 + np.pi*(convoy3Pos%22000-4000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-4000)/1500))
            ])
            print("curva1")
        elif convoy3Pos%22000 < 9500:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi)
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi)
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(np.pi)
            ])
            print("recta2")
        elif convoy3Pos%22000 < 11000:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi + np.pi*(convoy3Pos%22000-9500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-9500)/1500))
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi + np.pi*(convoy3Pos%22000-9500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-9500)/1500))
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(np.pi + np.pi*(convoy3Pos%22000-9500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-9500)/1500))
            ])
            print("curva2")
        elif convoy3Pos%22000 < 15000:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2)
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2)
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2)
            ])
            print("recta3")
        elif convoy3Pos%22000 < 16500:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2 + np.pi*(convoy3Pos%22000-15000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-15000)/1500))
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2 + np.pi*(convoy3Pos%22000-15000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-15000)/1500))
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(3*np.pi/2 + np.pi*(convoy3Pos%22000-15000)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-15000)/1500))
            ])
            print("curva3")
        elif convoy3Pos%22000 < 20500:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                )
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                )
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                )
            ])
            print("recta4")
        else:
            fromSPtransform = tr.matmul([
                tr.translate(
                    convoy3_1Curve[convoy3Pos%22000, 0],
                    convoy3_1Curve[convoy3Pos%22000, 1],
                    convoy3_1Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi*(convoy3Pos%22000-20500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-20500)/1500))
            ])
            triTransform = tr.matmul([
                tr.translate(
                    convoy3_2Curve[convoy3Pos%22000, 0],
                    convoy3_2Curve[convoy3Pos%22000, 1],
                    convoy3_2Curve[convoy3Pos%22000, 2]
                ),
                tr.rotationZ(np.pi*(convoy3Pos%22000-20500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-20500)/1500))
            ])
            nabooTransform = tr.matmul([
                tr.translate(
                    convoy3Curve[(convoy3Pos)%22000, 0],
                    convoy3Curve[(convoy3Pos)%22000, 1],
                    convoy3Curve[(convoy3Pos)%22000, 2]
                ),
                tr.rotationZ(np.pi*(convoy3Pos%22000-20500)/3000),
                tr.rotationY(np.pi/4*np.sin(np.pi*(convoy3Pos%22000-20500)/1500))
            ])
            print("curva4")
        fromSP = sg.findNode(figther, "fromSP")
        fromSP.transform = fromSPtransform
        tri = sg.findNode(figther, "tri")
        tri.transform = triTransform
        naboo = sg.findNode(figther, "naboo")
        naboo.transform = nabooTransform
        
        #nave usuario
        usuarioNave = sg.findNode(figther, "usuario")
        if controller.doblar == 0:
            usuarioTransform = tr.matmul([
                tr.translate(
                    controller.position[0],
                    controller.position[1],
                    controller.position[2],
                ),
                tr.rotationZ(-controller.phi),
                tr.rotationX(controller.theta+controller.backflip),
            ])
        #doblar derecha
        elif controller.doblar == 1:
            usuarioTransform = tr.matmul([
                tr.translate(
                    controller.position[0],
                    controller.position[1],
                    controller.position[2],
                ),
                tr.rotationZ(-controller.phi),
                tr.rotationX(controller.theta+controller.backflip),
                tr.rotationY(np.pi/8)
            ])
            controller.doblar = 0
        #doblar izquierda
        else:
            usuarioTransform = tr.matmul([
                tr.translate(
                    controller.position[0],
                    controller.position[1],
                    controller.position[2],
                ),
                tr.rotationZ(-controller.phi),
                tr.rotationX(controller.theta+controller.backflip),
                tr.rotationY(-np.pi/8)
            ])
            controller.doblar = 0
        usuarioNave.transform = usuarioTransform


        


        #Sistema Solar
        mercurio = sg.findNode(sistemaSolar, "mercurioNode")
        mercurio.transform = tr.matmul([
            tr.rotationX(-0.1*np.pi),
            tr.rotationZ(time/0.16),
            tr.uniformScale(0.1)
        ])
        mercuriotraslacion = sg.findNode(sistemaSolar, "mercurioTranslation")
        mercuriotraslacion.transform = tr.matmul([
            tr.rotationX(0.05*np.pi),
            tr.rotationZ(time/58.5),
            tr.translate(-1.5, 0, 0)
        ])

        venus = sg.findNode(sistemaSolar, "venusNode")
        venus.transform = tr.matmul([
            tr.rotationX(0.15*np.pi),
            tr.rotationZ(time/243),
            tr.uniformScale(0.2)
        ])
        venustraslacion = sg.findNode(sistemaSolar, "venusTranslation")
        venustraslacion.transform = tr.matmul([
            tr.rotationX(-0.01*np.pi),
            tr.rotationZ(time/224),
            tr.translate(2.2, 0, 0)
        ])
        
        tierra = sg.findNode(sistemaSolar, "tierraNode")
        tierra.transform = tr.matmul([
            tr.rotationX(-0.1*np.pi),
            tr.rotationZ(time),
            tr.uniformScale(0.22)

        ])
        luna = sg.findNode(sistemaSolar, "lunaNode")
        luna.transform = tr.matmul([
            tr.rotationZ(time/29.5),
            tr.uniformScale(0.08)
        ])
        lunatraslacion = sg.findNode(sistemaSolar, "lunaTranslation")
        lunatraslacion.transform = tr.matmul([
            tr.rotationZ(time/29.5),
            tr.translate(0.5, 0, 0)
        ])
        sistTierraLuna = sg.findNode(sistemaSolar, "sistemaTierraLuna")
        sistTierraLuna.transform = tierraTransform

        marte = sg.findNode(sistemaSolar, "marteNode")
        marte.transform = tr.matmul([
            tr.rotationX(0.1*np.pi),
            tr.rotationZ(time),
            tr.uniformScale(0.2)
        ])
        martetraslacion = sg.findNode(sistemaSolar, "marteTranslation")
        martetraslacion.transform = tr.matmul([
            tr.rotationX(0.05*np.pi),
            tr.rotationZ(time/668),
            tr.translate(0, -4.5, 0)
        ])

        jupiter = sg.findNode(sistemaSolar, "jupiterNode")
        jupiter.transform = tr.matmul([
            tr.rotationX(0.2*np.pi),
            tr.rotationZ(2.4 * time),
            tr.uniformScale(0.52)
        ])
        jupitertraslacion = sg.findNode(sistemaSolar, "jupiterTranslation")
        jupitertraslacion.transform = jupiterTransform

        saturno = sg.findNode(sistemaSolar, "saturnoNode")
        saturno.transform = tr.matmul([
            tr.rotationX(0.1*np.pi),
            tr.rotationZ(2.4 * time),
            tr.uniformScale(0.5)
        ])
        anillo= sg.findNode(sistemaSolar, "anilloNode")
        anillo.transform = tr.matmul([
            tr.rotationX(-0.1*np.pi),
            tr.rotationZ(2 * time),
            tr.uniformScale(0.54)
        ])

        saturnotraslacion = sg.findNode(sistemaSolar, "sistemaSaturno")
        saturnotraslacion.transform = tr.matmul([
            tr.rotationX(0.01*np.pi),
            tr.rotationZ(time/10767),
            tr.translate(23.8*np.cos(np.pi/4), 23.8*np.sin(np.pi/4), 0)
        ])

        urano = sg.findNode(sistemaSolar, "uranoNode")
        urano.transform = tr.matmul([
            tr.rotationX(-0.15*np.pi),
            tr.rotationZ(1.4 * time),
            tr.uniformScale(0.4)
        ])
        uranotraslacion = sg.findNode(sistemaSolar, "uranoTranslation")
        uranotraslacion.transform = tr.matmul([
            tr.rotationX(-0.01*np.pi),
            tr.rotationZ(time/30660),
            tr.translate(-48.7*np.cos(np.pi/4), 48.7*np.cos(np.pi/4), 0)
        ])

        neptuno = sg.findNode(sistemaSolar, "neptunoNode")
        neptuno.transform = tr.matmul([
            tr.rotationX(-0.25*np.pi),
            tr.rotationZ(1.5 * time),
            tr.uniformScale(0.4)
        ])
        neptunotraslacion = sg.findNode(sistemaSolar, "neptunoTranslation")
        neptunotraslacion.transform = tr.matmul([
            tr.rotationX(0.03*np.pi),
            tr.rotationZ(time/60225),
            tr.translate(75.1, 0, 0)
        ])

        pluton = sg.findNode(sistemaSolar, "plutonNode")
        pluton.transform = tr.matmul([
            tr.rotationX(0.1*np.pi),
            tr.rotationZ(time/367),
            tr.uniformScale(0.08)
        ])
        plutontraslacion = sg.findNode(sistemaSolar, "plutonTranslation")
        plutontraslacion.transform = tr.matmul([
            tr.rotationX(0.02*np.pi),
            tr.rotationZ(time/90520),
            tr.translate(0, 98.6, 0)
        ])

        Rcometa = (10- 2*np.cos(np.pi*time/250))
        cometaNode = sg.findNode(sistemaSolar, "cometaNode")
        cometaNode.transform = tr.matmul([
            tr.rotationY(-np.pi*time/250),
            tr.rotationZ(time/50), 
            tr.uniformScale(0.01)
        ])
        cometa = sg.findNode(sistemaSolar, "cometaTranslation")
        cometa.transform = tr.translate(Rcometa*np.cos(np.pi*time/250), -1, Rcometa*np.sin(np.pi*time/250))
        

        #graficar escena
        glUseProgram(pipeline.shaderProgram)
        sg.drawSceneGraphNode(figther, pipeline, "model")

        glUseProgram(textpipeline.shaderProgram)
        sg.drawSceneGraphNode(sistemaSolar, textpipeline, "model")


        glUseProgram(MVPpipeline.shaderProgram)

        sg.drawSceneGraphNode(fondo, MVPpipeline, "model")

        glUniformMatrix4fv(glGetUniformLocation(MVPpipeline.shaderProgram, "model"), 1, GL_TRUE, tr.rotationZ(time/27))
        MVPpipeline.drawCall(sol)
        
        glUniformMatrix4fv(glGetUniformLocation(MVPpipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        MVPpipeline.drawCall(gpuAxis, GL_LINES)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)



    # freeing GPU memory
    gpuAxis.clear()
    sistemaSolar.clear()
    sol.clear()
    fondo.clear()
    figther.clear()

    glfw.terminate()

if __name__ == "__main__":
    main()