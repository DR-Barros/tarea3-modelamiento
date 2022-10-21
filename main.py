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
    def camera(self, time):
        if self.camType == 0:
            up = np.array([0, 0, np.cos(self.theta)])
            self.view = tr.lookAt(
                controller.position - controller.front*0.01 + np.array([0, 0, 0.001*np.cos(self.theta)]),
                controller.position + np.array([0, 0, 0.001*np.cos(self.theta)]),
                up/np.linalg.norm(up)
            )
        elif self.camType == 1:
            cX = 4
            cY = cX*2
            posicionFront = np.array([0.08, -0.05, 0.02]) 
            if time%1600 < 200:
                posicionAt = np.array([cX, 2-cY*(time%1600)/200, 2])
                posicionFront = np.array([0.08, -0.05, 0.02])
            elif time%1600 < 800:
                posicionAt = np.array([cX*np.cos(np.pi*((time%1600)-200)/600), 2-cY-cX*np.sin(np.pi*((time%1600)-200)/600), 2])
                #posicionFront = np.array([0.08*np.cos(np.pi*((time%1600)-200)/600), -0.05 -0.05*np.sin(np.pi*((time%1600)-200)/600), 0.02])
            elif time%1600<1000:
                posicionAt = np.array([-cX, 2-cY +cY*(time%1600-800)/200, 2])
                #posicionFront = np.array([-0.08, -0.05, 0.02])
            else:
                posicionAt = np.array([-cX*np.cos(np.pi*((time%1600)-1000)/600), 2+cX*np.sin(np.pi*((time%1600)-1000)/600), 2])
                #posicionFront = np.array([-0.08*np.cos(np.pi*((time%1600)-1000)/600), -0.05 -0.05*np.sin(np.pi*((time%1600)-200)/600), 0.02])

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
            self.view = tr.lookAt(
                np.array([(15-self.jupiter*1.5)*np.cos(time/3942+np.pi/2), (15-self.jupiter*1.5)*np.sin(time/3942+np.pi/2), -2.3465*np.cos(time/3942)+0.4*(1-self.jupiter*0.5)]),
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
            )
        elif self.camType ==6:
            N=4
            C3time = 2000/(4*N-2)
            if time%2000 < C3time*(N-2):
                posicionAt = np.array([(N-2)*(time%2000)/(C3time*(N-2))-N/2+1, N/2, -1])
            elif time%2000 < C3time*(N-0.5):
                posicionAt = np.array([
                    N/2-1 + np.sin(np.pi*(time%2000-C3time*(N-2))/(3*C3time)), 
                    N/2-1 + np.cos(np.pi*(time%2000-C3time*(N-2))/(3*C3time)), 
                    -1
                ])
            elif time%2000 < C3time*(2*N-2.5):
                posicionAt = np.array([
                    N/2, 
                    N/2-1-(N-2)*(time%2000-C3time*(N-0.5))/(C3time*(N-2)), 
                    -1
                ])
            elif time%2000 < C3time*(2*N-1):
                posicionAt = np.array([
                    N/2 -1 + np.cos(np.pi*(time%2000-C3time*(2*N-2.5))/(3*C3time)), 
                    -N/2 +1 - np.sin(np.pi*(time%2000-C3time*(2*N-2.5))/(3*C3time)),  
                    -1
                ])
            elif time%2000 < C3time*(3*N-3):
                posicionAt = np.array([
                    N/2-1-(N-2)*(time%2000-C3time*(2*N-1))/(C3time*(N-2)), 
                    -N/2, 
                    -1
                ])
            elif time%2000 < C3time*(3*N-1.5):
                posicionAt = np.array([
                    -N/2 +1 - np.sin(np.pi*(time%2000-C3time*(3*N-3))/(3*C3time)), 
                    -N/2 +1 - np.cos(np.pi*(time%2000-C3time*(3*N-3))/(3*C3time)), 
                    -1
                ])
            elif time%2000 < C3time*(4*N-3.5):
                posicionAt = np.array([
                    -N/2,
                    -N/2+1+(N-2)*(time%2000-C3time*(3*N-1.5))/(C3time*(N-2)), 
                    -1
                ])
            else:
                posicionAt = np.array([
                    -N/2 +1 - np.cos(np.pi*(time%2000-C3time*(4*N-3.5))/(3*C3time)), 
                    N/2 -1 + np.sin(np.pi*(time%2000-C3time*(4*N-3.5))/(3*C3time)), 
                    -1
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
def esferaPhong(N, r, g, b):
    vertices = []
    faces = []
    angulo = 2 * np.pi
    n = int(N/2)
    for i in range(N+1):
        omega = i/N * angulo
        if i == N:
            omega = 0
            print(0)
        for j in range(n+1):
            theta = j/(N)* angulo
            vertices += [
                [np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta),
                abs(r), abs(g), abs(b),
                np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta)],
            ]
            if i != N:
                faces +=[
                    [n*i+j, n*i+1+j, (n*(i+1)+j)%(N*n)],
                    [n*i+1+j, (n*(i+1)+j)%(N*n), (n*(i+1)+1+j)%(N*n)]
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
def crearAnillo(N, R, r, g, b):
    vertices = []
    indices = []
    for i in range(N):
        theta = i/N * 2*np.pi
        vertices += [
            R*np.cos(theta), R*np.sin(theta), 0, r,g,b,
            (R+0.1)*np.cos(theta), (R+0.1)*np.sin(theta), 0, r,g,b,
        ]
        indices += [
            i, i+1, i+2,
            (i+N)%(N*2), (i+N+1)%(N*2), (i+N+2)%(N*2)
        ]
    return bs.Shape(vertices, indices)

#Funcion crear cometa
def crearCometa(N, r, g, b):
    vertices = []
    indices = []
    angulo = 2 * np.pi
    n = int(N/2)-1
    for i in range(N):
        indices += [N*n, i*n, ((i+1)*n)%(N*n)]
        rand = random()/8
        omega = i/N * angulo
        for j in range(n):
            theta = (j+1)/N * angulo*9/16
            vertices += [
                    np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta), abs(r-rand), abs(g-rand), abs(b-rand)
                ]
        for j in range(n-1):
            indices += [
                n*i+j, n*i+1+j, (n*(i+1)+j)%(N*n),
                n*i+1+j, (n*(i+1)+j)%(N*n), (n*(i+1)+1+j)%(N*n),
            ]
        indices += [N*n+1, (i*n-1)%(N*n), ((i+1)*n-1)%(N*n)]
    
    vertices += [
        0, 0, 1, r, g, b,
        0, 0, -4, r, g, b 
    ] 
    return bs.Shape(vertices, indices)

#post procesado de shapes para usar en lightning pipelines
def postPL(Shape, color):
    vertices = Shape.vertices
    numVertices = len(vertices)//6
    vert2 = []
    for i in range(len(vertices)//6):
        vert2 += [vertices[i*6], vertices[i*6+1], vertices[i*6+2]]
    indices = Shape.indices
    numFaces = len(indices)//3
    vertices = np.asarray(vert2)
    vertices = np.reshape(vertices, (numVertices, 3))
    normals = np.zeros((numVertices,3), dtype=np.float32)
    faces = []
    for i in range(numFaces):
        aux = [indices[i*3], indices[i*3+1], indices[i*3+2]]
        faces += [aux[0:]]
        
        vecA = [vertices[aux[1]][0] - vertices[aux[0]][0], vertices[aux[1]][1] - vertices[aux[0]][1], vertices[aux[1]][2] - vertices[aux[0]][2]]
        vecB = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]

        res = np.cross(vecA, vecB)
        normals[aux[0]][0] += res[0]  
        normals[aux[0]][1] += res[1]  
        normals[aux[0]][2] += res[2]  

        normals[aux[1]][0] += res[0]  
        normals[aux[1]][1] += res[1]  
        normals[aux[1]][2] += res[2]  

        normals[aux[2]][0] += res[0]  
        normals[aux[2]][1] += res[1]  
        normals[aux[2]][2] += res[2]  
    #print(faces)
    norms = np.linalg.norm(normals,axis=1)
    normals = normals/norms[:,None]

    color = np.asarray(color)
    color = np.tile(color, (numVertices, 1))

    vertexData = np.concatenate((vertices, color), axis=1)
    vertexData = np.concatenate((vertexData, normals), axis=1)

    #print(vertexData.shape)

    indices = []
    vertexDataF = []
    index = 0

    for face in faces:
        vertex = vertexData[face[0],:]
        vertexDataF += vertex.tolist()
        vertex = vertexData[face[1],:]
        vertexDataF += vertex.tolist()
        vertex = vertexData[face[2],:]
        vertexDataF += vertex.tolist()
        
        indices += [index, index + 1, index + 2]
        index += 3        



    return bs.Shape(vertexDataF, indices)

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

#Crear sistema solar scene graph
def createSystem(pipeline):
    #mercurioShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0, 1 ,0)), pipeline)
    mercurioShape = createGPUShape(esferaPhong(100, 0, 1, 0), pipeline)
    #venusShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0, 1 ,1)), pipeline)
    venusShape = createGPUShape(esferaPhong(100, 0, 1, 1), pipeline)
    tierraShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0, 0 ,1)), pipeline)
    tierraShape = createGPUShape(esferaPhong(100, 0, 0, 1), pipeline)
    lunaShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (1, 1 ,1)), pipeline)
    marteShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (1, 0 ,0)), pipeline)
    jupiterShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (1, 0 ,1)), pipeline)
    saturnoShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0.5, 1 ,0.5)), pipeline)
    anillo1Shape = createGPUShape(postPL(crearAnillo(100, 1, 1, 1, 1), (1, 1, 1)), pipeline)
    anillo2Shape = createGPUShape(postPL(crearAnillo(100, 1.15, 0.8, 0.8, 0.8),(0.8,0.8,0.8)), pipeline)
    anillo3Shape = createGPUShape(postPL(crearAnillo(100, 1.3, 0.6, 0.6, 0.6), (0.6,0.6,0.6)), pipeline)
    uranoShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (1 ,0.5, 0)), pipeline)
    neptunoShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0.5, 0.8, 0)), pipeline)
    plutonShape = createGPUShape(bs.readOFF(getAssetPath('sphere.off'), (0.1, 0.8, 0.5)), pipeline)
    cometaShape = createGPUShape(postPL(crearCometa(100, 136/255, 9/255, 123/255), (136/255, 9/255, 123/255)), pipeline)

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
    anillo1Node = sg.SceneGraphNode("anillo1Node")
    anillo1Node.transform = tr.uniformScale(0.54)
    anillo1Node.childs += [anillo1Shape]
    anillo2Node = sg.SceneGraphNode("anillo2Node")
    anillo2Node.transform = tr.uniformScale(0.54)
    anillo2Node.childs += [anillo2Shape]
    anillo3Node = sg.SceneGraphNode("anillo3Node")
    anillo3Node.transform = tr.uniformScale(0.54)
    anillo3Node.childs += [anillo3Shape]
    anillosNode =sg.SceneGraphNode("anillosNode")
    anillosNode.transform = tr.identity()
    anillosNode.childs += [anillo1Node, anillo2Node, anillo3Node]
    sistemaSaturno = sg.SceneGraphNode("sistemaSaturno")
    sistemaSaturno.transform = tr.translate(-23.8, 0, 0)
    sistemaSaturno.childs += [saturnoNode, anillosNode]

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
    while contador < 1000:
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
    tieUVShape = createGPUShape(bs.readOFF(getAssetPath('tie_UV.off'), (0.3 , 0.3 ,0.3)), pipeline)
    triFighterShape = createGPUShape(bs.readOFF(getAssetPath('Tri_Fighter.off'), (0.3 , 0.3 ,0.9)), pipeline)
    xWingShape = createGPUShape(bs.readOFF(getAssetPath('XJ5 X-wing starfighter.off'), (0.9 , 0.9 ,0.9)), pipeline)

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

    #jupiter convoy
    jupiterconvoy = sg.SceneGraphNode("jupiterConvoy")
    jupiterconvoy.transform = tr.translate(0.53, 0, 0)
    navesJupiter= []
    contador = 0
    while contador < 50:
        posX = random()
        posY = random()
        posZ = random()
        num = str(contador)
        navesJupiter += [sg.SceneGraphNode(num+"Xwing")]
        navesJupiter += [sg.SceneGraphNode(num+"Tie")]
        navesJupiter[2*contador].transform = tr.translate(0.3*(posX-0.5), 0.3*posY, 0.05*posZ)
        navesJupiter[2*contador+1].transform = tr.translate(0.3*(posX-0.5), -0.3*posY, 0.05*posZ)
        navesJupiter[2*contador].childs += [tieUVNode]
        navesJupiter[2*contador+1].childs += [xWingNode]
        jupiterconvoy.childs += [navesJupiter[2*contador], navesJupiter[2*contador+1]]
        contador += 1
    jupiterOrbit = sg.SceneGraphNode("jupiterOrbit")
    jupiterOrbit.transform = tr.identity()
    jupiterOrbit.childs += [jupiterconvoy]
    jupiterMove = sg.SceneGraphNode("jupiterMove")
    jupiterMove.transform = tr.identity()
    jupiterMove.childs += [jupiterOrbit]

    

    #Convoy
    #Tie
    tieUVTraslation = sg.SceneGraphNode("tieUVTraslation")
    tieUVTraslation.transform = tr.translate(0.03, 0, 0)
    tieUVTraslation.childs += [tieUVNode]

    tieUVRotation = sg.SceneGraphNode("tieUVRotation")
    tieUVRotation.transform = tr.rotationZ(0)
    tieUVRotation.childs += [tieUVTraslation]


    #crear destructores con un Tie rotando al rededor
    destroyer1 = sg.SceneGraphNode("destroyer1")
    destroyer1.transform = tr.identity()
    destroyer1.childs += [destroyerNode, tieUVRotation]

    destroyer2 = sg.SceneGraphNode("destroyer2")
    destroyer2.transform = tr.translate(-0.05, -0.02, -0.02)
    destroyer2.childs += [destroyerNode, tieUVRotation]

    destroyer3 = sg.SceneGraphNode("destroyer3")
    destroyer3.transform = tr.translate(0.05, -0.02, -0.02)
    destroyer3.childs += [destroyerNode, tieUVRotation]

    convoy = sg.SceneGraphNode("convoy")
    convoy.transform = tr.identity()
    convoy.childs += [destroyer1, destroyer2, destroyer3]
    
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

    naboo1 = sg.SceneGraphNode("naboo1")
    naboo1.transform = tr.identity()
    naboo1.childs += [nabooNode]
    naboo1Translate = sg.SceneGraphNode("naboo1Translate")
    naboo1Translate.transform = tr.translate(0.04, 0, 0)
    naboo1Translate.childs += [naboo1]

    naboo2 = sg.SceneGraphNode("naboo2")
    naboo2.transform = tr.identity()
    naboo2.childs += [nabooNode]
    naboo2Translate = sg.SceneGraphNode("naboo2Translate")
    naboo2Translate.transform = tr.translate(0.04, 0.06, 0)
    naboo2Translate.childs += [naboo2]

    naboo3 = sg.SceneGraphNode("naboo3")
    naboo3.transform = tr.identity()
    naboo3.childs += [nabooNode]
    naboo3Translate = sg.SceneGraphNode("naboo3Translate")
    naboo3Translate.transform = tr.translate(0.04, -0.06, 0)
    naboo3Translate.childs += [naboo3]

    convoy3 = sg.SceneGraphNode("convoy3")
    convoy3.transform = tr.identity()
    convoy3.childs += [fromSPtranslate, triTranslate, naboo1Translate, naboo2Translate, naboo3Translate]

    usuario =sg.SceneGraphNode("usuario")
    usuario.transform = tr.identity()
    usuario.childs += [xWingNode]

    sceneNode = sg.SceneGraphNode("naves")
    sceneNode.childs += [usuario, convoy, earthMove, convoy2, jupiterMove, convoy3]

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
        if key == glfw.KEY_6:
            controller.camType = 6
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
    sistemaSolar = createSystem(pipeline)
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

    while not glfw.window_should_close(window):

        #usar GLFW para chequear input events
        glfw.poll_events()

        #contador de tiempo
        time = 10 * glfw.get_time()

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

        view = controller.camera(time)

        #View y proyeccion
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

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
        
        glUseProgram(pipeline.shaderProgram)
        #Naves

        #convoy tierra
        earthMove = sg.findNode(figther, "earthMove")
        earthMove.transform = tierraTransform
        earthOrbit = sg.findNode(figther, "earthOrbit")
        earthOrbit.transform =tr.matmul([
            tr.rotationX(np.pi/10),
            tr.rotationZ(-time/20)
        ])
        earthConvoy = sg.findNode(figther, "earthConvoy")
        earthConvoy.transform = tr.matmul([
            tr.translate(0.229, 0, 0),
            tr.rotationY(np.pi/2)
        ])

        #Convoy tierra
        jupiterMove = sg.findNode(figther, "jupiterMove")
        jupiterMove.transform = jupiterTransform
        jupiterOrbit = sg.findNode(figther, "jupiterOrbit")
        jupiterOrbit.transform =tr.matmul([tr.rotationZ(time/40)])
        jupiterConvoy = sg.findNode(figther, "jupiterConvoy")
        jupiterConvoy.transform = tr.matmul([
            tr.translate(0.53, 0, 0),
            tr.rotationY(np.pi/2)
        ])


        #Convoy 
        convoy = sg.findNode(figther, "convoy")
        destroyer1 = sg.findNode(figther, "destroyer1")
        destroyer2 = sg.findNode(figther, "destroyer2")
        destroyer3 = sg.findNode(figther, "destroyer3")
        tieRotation = sg.findNode(figther, "tieUVRotation")
        tieRotation.transform = tr.rotationZ(time/10)
        cX = 4
        cY = cX*2
        if time%1600<200:
            convoy.transform = tr.translate(cX, 2-cY*(time%1600)/200, 2)
        elif time%1600<800:
            convoy.transform = tr.matmul([
                tr.translate(cX*np.cos(np.pi*((time%1600)-200)/600), 2-cY-cX*np.sin(np.pi*((time%1600)-200)/600), 2),
                tr.rotationZ(-np.pi*((time%1600)-200)/600)
            ])
            destroyer1.transform = tr.rotationY(-np.sin(np.pi*((time%1600)-200)/600)*np.pi/4)
            destroyer2.transform = tr.matmul([
                tr.translate(-0.05, -0.02, -0.02),
                tr.rotationY(-np.sin(np.pi*((time%1600)-200)/600)*np.pi/4)
            ])
            destroyer3.transform = tr.matmul([
                tr.translate(0.05, -0.02, -0.02),
                tr.rotationY(-np.sin(np.pi*((time%1600)-200)/600)*np.pi/4)
            ])    
        elif time%1600<1000:
            convoy.transform = tr.matmul([
                tr.translate(-cX, 2-cY +cY*(time%1600-800)/200, 2),
                tr.rotationZ(-np.pi)
            ])
        else:
            convoy.transform = tr.matmul([
                tr.translate(-cX*np.cos(np.pi*((time%1600)-1000)/600), 2+cX*np.sin(np.pi*((time%1600)-1000)/600), 2),
                tr.rotationZ(-np.pi-np.pi*((time%1600)-1000)/600)
            ])
            destroyer1.transform = tr.rotationY(-np.sin(np.pi*((time%1600)-1000)/600)*np.pi/4)
            destroyer2.transform = tr.matmul([
                tr.translate(-0.05, -0.02, -0.02),
                tr.rotationY(-np.sin(np.pi*((time%1600)-1000)/600)*np.pi/4)
            ])
            destroyer3.transform = tr.matmul([
                tr.translate(0.05, -0.02, -0.02),
                tr.rotationY(-np.sin(np.pi*((time%1600)-1000)/600)*np.pi/4)
            ])

        #Convoy2
        convoy2 = sg.findNode(figther, "convoy2")
        convoy2.transform = tr.matmul([
            tr.rotationZ(-time/20),
            tr.translate(1.1, 0, 0)
        ])
        corvette = sg.findNode(figther, "corvetteNode")
        corvette.transform =  tr.matmul([
            tr.rotationY(np.pi/4),
            tr.rotationX(np.pi/2),
            tr.uniformScale(0.05)
        ])

        #Convoy3
        N=4
        C3time = 2000/(4*N-2)
        if time%2000 < C3time*(N-2):
            convoy3Transform = tr.translate((N-2)*(time%2000)/(C3time*(N-2))-N/2+1, N/2, -1)
            navesTransform = tr.rotationZ(np.pi/2)
        elif time%2000 < C3time*(N-0.5):
            convoy3Transform = tr.matmul([
                tr.translate(
                    N/2-1 + np.sin(np.pi*(time%2000-C3time*(N-2))/(3*C3time)), 
                    N/2-1 + np.cos(np.pi*(time%2000-C3time*(N-2))/(3*C3time)), 
                    -1
                ),
                tr.rotationZ(- np.pi*(time%2000-C3time*(N-2))/(3*C3time))
            ])
            
            navesTransform = tr.matmul([
                tr.rotationX(np.pi/4*np.sin(np.pi*(time%2000-C3time*(N-2))/(1.5*C3time))),
                tr.rotationZ(np.pi/2),
            ])
        elif time%2000 < C3time*(2*N-2.5):
            convoy3Transform = tr.matmul([    
                tr.translate(
                    N/2, 
                    N/2-1-(N-2)*(time%2000-C3time*(N-0.5))/(C3time*(N-2)), 
                    -1
                ),
                tr.rotationZ(-np.pi/2)
            ])
            navesTransform = tr.rotationZ(np.pi/2)
        elif time%2000 < C3time*(2*N-1):
            convoy3Transform = tr.matmul([
                tr.translate(
                    N/2 -1 + np.cos(np.pi*(time%2000-C3time*(2*N-2.5))/(3*C3time)), 
                    -N/2 +1 - np.sin(np.pi*(time%2000-C3time*(2*N-2.5))/(3*C3time)),  
                    -1
                ), 
                tr.rotationZ(-np.pi/2-np.pi*(time%2000-C3time*(2*N-2.5))/(3*C3time))
            ])
            navesTransform = tr.matmul([
                tr.rotationX(np.pi/4*np.sin(np.pi*(time%2000-C3time*(2*N-2.5))/(1.5*C3time))),
                tr.rotationZ(np.pi/2),
            ])
        elif time%2000 < C3time*(3*N-3):
            convoy3Transform = tr.matmul([
                tr.translate(
                    N/2-1-(N-2)*(time%2000-C3time*(2*N-1))/(C3time*(N-2)), 
                    -N/2, 
                    -1
                ),
                tr.rotationZ(-np.pi)
            ])
            navesTransform = tr.rotationZ(np.pi/2)
        elif time%2000 < C3time*(3*N-1.5):
            convoy3Transform = tr.matmul([
                tr.translate(
                    -N/2 +1 - np.sin(np.pi*(time%2000-C3time*(3*N-3))/(3*C3time)), 
                    -N/2 +1 - np.cos(np.pi*(time%2000-C3time*(3*N-3))/(3*C3time)), 
                    -1
                ),
                tr.rotationZ(-np.pi-np.pi*(time%2000-C3time*(3*N-3))/(3*C3time))
            ])
            navesTransform = tr.matmul([
                tr.rotationX(np.pi/4*np.sin(np.pi*(time%2000-C3time*(3*N-3))/(1.5*C3time))),
                tr.rotationZ(np.pi/2),
            ])
        elif time%2000 < C3time*(4*N-3.5):
            convoy3Transform = tr.matmul([
                tr.translate(
                    -N/2,
                    -N/2+1+(N-2)*(time%2000-C3time*(3*N-1.5))/(C3time*(N-2)), 
                    -1
                ),
                tr.rotationZ(-3*np.pi/2)
            ])
            navesTransform = tr.rotationZ(np.pi/2)
        else:
            convoy3Transform = tr.matmul([
                tr.translate(
                    -N/2 +1 - np.cos(np.pi*(time%2000-C3time*(4*N-3.5))/(3*C3time)), 
                    N/2 -1 + np.sin(np.pi*(time%2000-C3time*(4*N-3.5))/(3*C3time)), 
                    -1
                ),
                tr.rotationZ(-3*np.pi/2-np.pi*(time%2000-C3time*(4*N-3.5))/(3*C3time))
            ])
            navesTransform = tr.matmul([
                tr.rotationX(np.pi/4*np.sin(np.pi*(time%2000-C3time*(4*N-3.5))/(1.5*C3time))),
                tr.rotationZ(np.pi/2),
            ])
        convoy3 = sg.findNode(figther, "convoy3")
        convoy3.transform = convoy3Transform
        fromSP = sg.findNode(figther, "fromSP")
        fromSP.transform = navesTransform
        tri = sg.findNode(figther, "tri")
        tri.transform = navesTransform
        naboo1 = sg.findNode(figther, "naboo1")
        naboo1.transform = navesTransform
        naboo2 = sg.findNode(figther, "naboo2")
        naboo2.transform = navesTransform
        naboo3 = sg.findNode(figther, "naboo3")
        naboo3.transform = navesTransform
        

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
        anillos= sg.findNode(sistemaSolar, "anillosNode")
        anillos.transform = tr.matmul([
            tr.rotationX(-0.1*np.pi),
            tr.rotationZ(2 * time),
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
            tr.rotationZ(time), 
            tr.uniformScale(0.01)
        ])
        cometa = sg.findNode(sistemaSolar, "cometaTranslation")
        cometa.transform = tr.translate(Rcometa*np.cos(np.pi*time/250), -1, Rcometa*np.sin(np.pi*time/250))
        



        #graficar escena
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
        sg.drawSceneGraphNode(figther, pipeline, "model")

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0, 0, 0)
        sg.drawSceneGraphNode(sistemaSolar, pipeline, "model")


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
