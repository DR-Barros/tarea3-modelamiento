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
    n = int(N/2)
    for i in range(N):
        indices += [N*n, i*n, ((i+1)*n)%(N*n)]
        omega = i/N * angulo
        for j in range(n):
            theta = ((j)/(N-1) * angulo)
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



#post procesado de shapes para usar en lightning pipelines
def postPL(Shape, color):
    Vertices = Shape.vertices
    numVertices = len(Vertices)//6
    vert2 = []
    for i in range(len(Vertices)//6):
        vert2 += [Vertices[i*6], Vertices[i*6+1], Vertices[i*6+2]]
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

def on_key(window, key, scancode, action, mods):
    #Si no se presiona una tecla no hace nada
    if action != glfw.PRESS:
        return

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


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
    view = tr.lookAt(
        np.array([5, 5, 3]),
        np.array ([0,0, 0]),
        np.array([0,0,1])
    )

    #Crear shapes en la GPU memory
    #planeta = createGPUShape(postPL(crearEsfera(10, 1, 1, 1), [1,1,1]), pipeline)
    planeta = createGPUShape(esferaPhong(100, 1, 1, 1), pipeline)


    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 1.0, 0.7, 0.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0, 0, 0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 1, 1, 1)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

    while not glfw.window_should_close(window):
        #usar GLFW para chequear input events
        glfw.poll_events()


        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawCall(planeta)


        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)



    # freeing GPU memory
    planeta.clear()

    glfw.terminate()

if __name__ == "__main__":
    main()
